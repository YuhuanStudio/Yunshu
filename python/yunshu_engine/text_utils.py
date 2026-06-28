"""Output text cleaning utilities.

Removes special tokens from model output that may leak through tokenization
edge cases.
"""

from __future__ import annotations

import contextlib
import re


class StopHoldbackBuffer:
    """Withholds the tail of a streamed text that could be the start of a stop
    string, so a multi-token stop string never leaks its prefix to the client.

    SSE is append-only: once a chunk is sent it can't be retracted. If a stop
    string like ``"\\n\\n"`` arrives as two separate ``"\\n"`` tokens, naively
    emitting each token streams the first ``"\\n"`` before the match completes.

    The rule: after appending new text, hold back the longest *suffix* of the
    buffer that is a *prefix* of any stop string; emit the rest. When a stop
    completes, ``take_stopped()`` trims the matched stop and returns the
    preceding text. At end-of-generation (no stop) ``flush()`` emits whatever
    remains (it cannot extend into a stop).
    """

    def __init__(self, stop_strings: list[str] | None) -> None:
        self._stops = [s for s in (stop_strings or []) if s]
        self._buf = ""
        # No stop strings → never hold anything back (fast path).
        self._active = bool(self._stops)
        # Per-token logprob tracking for feed_lp(). Each entry is
        # [remaining_chars, lp_entry, lp_emitted] for a token still holding
        # un-emitted characters in _buf, front == oldest. Sum of remaining_chars
        # always equals len(_buf). Only populated/consumed via feed_lp(); the
        # plain feed() path never touches it.
        self._pend: list[list] = []

    def _first_stop_pos(self) -> int:
        """Index of the first COMPLETE stop occurrence in the buffer, or -1.

        Catches a stop that lands mid-segment (e.g. one token decodes to ``"aSTOPb"``
        for stop ``"STOP"``): the suffix-only holdback misses it (suffix ``"b"`` isn't
        a stop prefix) and would emit the whole thing, leaking the stop — the
        non-streaming path's find()-based backstop, which streaming previously lacked."""
        if not isinstance(self._buf, str):
            return -1  # defensive: a non-str buffer has no string stop occurrence
        first = -1
        for s in self._stops:
            i = self._buf.find(s)
            if i >= 0 and (first < 0 or i < first):
                first = i
        return first

    def _hold_len(self) -> int:
        """Longest suffix of the buffer that is a prefix of some stop string."""
        hold = 0
        for s in self._stops:
            kmax = min(len(self._buf), len(s))
            for k in range(kmax, 0, -1):
                if self._buf[-k:] == s[:k]:
                    if k > hold:
                        hold = k
                    break
        return hold

    def contains_stop(self) -> bool:
        """Whether a COMPLETE stop string is currently held in the buffer (the engine
        should fire the stop and finalize via take_stopped())."""
        return self._active and self._first_stop_pos() >= 0

    def feed(self, text: str) -> str:
        """Append ``text``; return the prefix safe to emit now."""
        if not self._active:
            return text
        if text:
            self._buf += text
        # A complete stop already in the buffer: emit only what's BEFORE it and hold
        # the stop + trailing (never leak it). The engine fires via contains_stop().
        sp = self._first_stop_pos()
        if sp >= 0:
            out = self._buf[:sp]
            self._buf = self._buf[sp:]
            return out
        hold = self._hold_len()
        if hold == 0:
            out, self._buf = self._buf, ""
            return out
        if hold >= len(self._buf):
            return ""
        out = self._buf[:-hold]
        self._buf = self._buf[-hold:]
        return out

    def feed_lp(self, text: str, lp_entry) -> list[tuple[str, object]]:
        """Logprob-aware variant of :meth:`feed`.

        Streaming logprobs attach one logprob entry per token to the chunk that
        carries that token's text. The plain :meth:`feed` concatenates text and
        loses token boundaries, so it cannot be used when ``logprobs`` is on
        without dropping or misaligning entries — which is exactly why the
        hold-back buffer used to be disabled under logprobs, leaving multi-token
        stop prefixes to leak (SSE is append-only).

        ``feed_lp`` keeps a per-token queue so it can:
          * hold back any buffer suffix that is a prefix of a stop string
            (so a multi-token stop never leaks its prefix), and
          * emit each token's ``lp_entry`` **exactly once**, in the chunk that
            first reveals any of that token's text.

        Returns a list of ``(emit_text, lp_entry_or_None)`` chunks safe to emit
        now. A token's ``lp_entry`` rides the chunk containing its first
        released character; any later-released characters of the same token (it
        was split across the hold boundary) carry ``None`` so the entry is never
        duplicated. Fully held tokens keep their entry pending until their text
        is released (or discarded by :meth:`take_stopped`).

        The held tail emitted by :meth:`flush`/:meth:`take_stopped` carries no
        per-token logprobs — same as the pre-existing non-logprob hold-back
        path. That tail is at most ``max(len(stop)) - 1`` characters.
        """
        if not self._active:
            return [(text, lp_entry)] if text else []
        # Record every token (even a zero-length detokenizer segment) so its
        # logprob keeps its place in the queue and is emitted in order — the
        # pre-hold-back path emitted empty-text logprob chunks too, and dropping
        # them would lose per-token logprobs.
        self._pend.append([len(text), lp_entry, False])
        if text:
            self._buf += text
        # A complete stop in the buffer: release only what's BEFORE it (never leak the
        # stop or trailing) — same mid-segment guard as feed(); engine fires via
        # contains_stop(). Otherwise hold the suffix that could be a stop prefix.
        sp = self._first_stop_pos()
        n_release = sp if sp >= 0 else (len(self._buf) - self._hold_len())
        if n_release <= 0:
            return []
        released = self._buf[:n_release]
        self._buf = self._buf[n_release:]
        out: list[tuple[str, object]] = []
        pos = 0
        r = n_release
        while r > 0 and self._pend:
            seg = self._pend[0]
            take = seg[0] if seg[0] <= r else r
            chunk = released[pos : pos + take]
            lp: object = None
            if not seg[2]:
                lp = seg[1]
                seg[2] = True
            out.append((chunk, lp))
            pos += take
            r -= take
            seg[0] -= take
            if seg[0] == 0:
                self._pend.pop(0)
        return out

    def take_stopped(self) -> str:
        """A stop fired: return the held-back text up to the FIRST stop occurrence
        (anywhere — end OR mid-buffer), dropping the stop and everything after it.
        Handles both ``"doneSTOP"`` (suffix) and ``"aSTOPb"`` (mid-segment)."""
        buf = self._buf
        self._buf = ""
        self._pend = []
        first = -1
        for s in self._stops:
            i = buf.find(s)
            if i >= 0 and (first < 0 or i < first):
                first = i
        if first >= 0:
            return buf[:first]
        for s in self._stops:  # fall back to suffix-prefix (partial tail)
            if buf.endswith(s):
                return buf[: -len(s)]
        return buf

    def flush(self) -> str:
        """End of generation — emit whatever is held back, but if a complete stop is
        held (engine didn't fire), drop it and everything after so it can't leak."""
        buf, self._buf = self._buf, ""
        self._pend = []
        first = -1
        for s in self._stops:
            i = buf.find(s)
            if i >= 0 and (first < 0 or i < first):
                first = i
        return buf[:first] if first >= 0 else buf

# Pattern matching common special tokens that should be removed from output.
# These tokens sometimes appear in output due to tokenizer quirks.
SPECIAL_TOKENS_PATTERN = re.compile(
    r"<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|"
    r"<\|end\|>|<\|eot_id\|>|<\|start_header_id\|>|<\|end_header_id\|>|"
    r"</s>|<s>|<pad>|\[PAD\]|\[SEP\]|\[CLS\]"
)


def clean_special_tokens(text: str) -> str:
    """Remove special tokens from model output."""
    if not text:
        return text
    return SPECIAL_TOKENS_PATTERN.sub("", text).strip()


def cache_tokenizer_vocab(tok) -> None:
    """Memoize the underlying HF tokenizer's get_vocab() (call once at engine start).

    PERF: mlx-lm builds a FRESH streaming detokenizer per request
    (``tokenizer.detokenizer`` property → new instance), whose ``__init__`` reads
    ``tokenizer.vocab`` — transformers implements that as ``get_vocab()`` with NO
    caching, rebuilding the full ~150k-entry dict from the Rust tokenizer (~49ms) twice
    per request. Profiling a 0.8B model's served TTFT showed get_vocab dominating at
    ~98ms of 184ms (the model forward was only ~8ms); caching it cut TTFT to 78ms (−58%).
    The vocab is immutable for a loaded tokenizer, so cache it once. Per-request
    detokenizer buffers are untouched (never pooled — reset() leaks byte buffers); only
    the immutable vocab lookup is shared.
    """
    import logging
    _log = logging.getLogger(__name__)
    try:
        hf = getattr(tok, "_tokenizer", None) or tok
        if not hasattr(hf, "get_vocab"):
            return
        # Idempotent: don't re-wrap an already-cached get_vocab.
        if getattr(hf.get_vocab, "_yunshu_cached", False):
            return
        _base = hf.get_vocab
        _cached = _base()

        def _cached_get_vocab(*args, **kwargs):
            # The .vocab property + detokenizer path call with_added_tokens=True (default).
            if kwargs.get("with_added_tokens", True) and not args:
                return _cached
            return _base(*args, **kwargs)

        _cached_get_vocab._yunshu_cached = True
        hf.get_vocab = _cached_get_vocab
        _log.debug("cached tokenizer vocab (%d entries) — skips per-request rebuild",
                   len(_cached))
    except Exception:
        _log.debug("tokenizer vocab cache setup failed", exc_info=True)


def chunk_prompt_tokens(
    token_ids: list[int],
    chunk_size: int = 2048,
) -> list[list[int]]:
    """Split prompt tokens into chunks for chunked prefill.

    Chunked prefill breaks long prompts into manageable pieces that
    fit within GPU memory constraints. Each chunk is processed
    sequentially, accumulating KV cache across chunks.

    Args:
        token_ids: Full prompt token IDs.
        chunk_size: Maximum tokens per chunk (default from prefill_step_size).

    Returns:
        List of token ID chunks.
    """
    if not token_ids:
        return []

    chunks = []
    for i in range(0, len(token_ids), chunk_size):
        chunks.append(token_ids[i : i + chunk_size])
    return chunks


def estimate_prefill_memory(
    num_tokens: int,
    num_layers: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    dtype_size: int = 2,
    num_attention_heads: int = 32,
) -> int:
    """Estimate peak memory usage during prefill (bytes).

    For head_dim <= 128 (most models): O(n) tiled SDPA
    For head_dim > 128: O(n^2) full attention matrix

    Returns estimated bytes needed for the prefill operation.
    """
    kv_memory = num_tokens * num_layers * num_kv_heads * head_dim * dtype_size * 2

    if head_dim > 128:
        # Full attention matrix materialized in float32
        attention = num_attention_heads * num_tokens * num_tokens * 4
        attention += num_attention_heads * num_tokens * head_dim * 4
    else:
        # Tiled/fused — O(n) memory
        attention = num_attention_heads * num_tokens * head_dim * 4

    return kv_memory + attention


def should_chunk_prefill(
    num_tokens: int,
    available_memory: int,
    model_config: dict | None = None,
    safety_factor: float = 0.8,
) -> bool:
    """Determine if a prompt should be chunked for prefill.

    Returns True if the prompt is too large for single-pass prefill.
    """
    if model_config is None:
        model_config = {}

    estimate = estimate_prefill_memory(
        num_tokens,
        num_layers=model_config.get("num_layers", 32),
        num_kv_heads=model_config.get("num_kv_heads", 8),
        head_dim=model_config.get("head_dim", 128),
        num_attention_heads=model_config.get("num_attention_heads", 32),
    )

    return estimate * safety_factor > available_memory


def get_eos_token_ids(tokenizer) -> list[int]:
    """Extract EOS token IDs from a tokenizer."""
    eos_ids: set[int] = set()
    if hasattr(tokenizer, "eos_token_id"):
        eid = tokenizer.eos_token_id
        if isinstance(eid, (list, tuple)):
            eos_ids.update(eid)
        elif eid is not None:
            eos_ids.add(eid)
    if hasattr(tokenizer, "eos_token_ids"):
        eids = tokenizer.eos_token_ids
        if isinstance(eids, (list, tuple)):
            eos_ids.update(eids)
        elif isinstance(eids, int):
            eos_ids.add(eids)
    return list(eos_ids)


# ── raw-bytes recovery for logprobs ────────────────────────────────────────────
def _build_gpt2_byte_decoder() -> dict:
    """The GPT-2 byte-level BPE surface-char → raw-byte map (the standard
    bytes_to_unicode, inverted)."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for c, b in zip(cs, bs, strict=True)}


_GPT2_BYTE_DECODER = _build_gpt2_byte_decoder()


def _is_byte_level_tokenizer(tokenizer) -> bool:
    """True iff the tokenizer is GPT-2/tiktoken-style byte-level BPE (Qwen, Llama-3,
    GPT) — i.e. its surface chars are the GPT-2 byte→unicode proxies.

    This discriminator is REQUIRED for token_id_to_bytes. Both byte-level
    BPE and sentencepiece can produce the identical surface char (e.g. 'é'), but it
    means raw byte [233] for byte-level (a multi-byte fragment) versus the character
    U+00E9 = UTF-8 [195,169] for sentencepiece. The surface alone can't disambiguate,
    so the GPT-2 reverse-byte-map branch must run ONLY for byte-level tokenizers;
    applying it to sentencepiece corrupted logprobs.bytes for accented/Latin-1 output.
    The reliable signal is a ByteLevel component in the fast tokenizer's
    decoder/pre_tokenizer; sentencepiece uses Metaspace/Replace, never ByteLevel.
    Memoized on the tokenizer instance (immutable post-load).
    """
    cached = getattr(tokenizer, "_yunshu_byte_level", None)
    if cached is not None:
        return cached
    result = False
    try:
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is not None:
            for comp in (getattr(backend, "decoder", None),
                         getattr(backend, "pre_tokenizer", None)):
                if comp is not None and "ByteLevel" in repr(comp):
                    result = True
                    break
    except Exception:
        result = False
    with contextlib.suppress(Exception):
        tokenizer._yunshu_byte_level = result
    return result


def token_id_to_bytes(tokenizer, token_id: int, fallback_str: str | None = None) -> list[int]:
    """Recover the RAW UTF-8 bytes a single token id represents.

    OpenAI's logprobs `bytes` field exists so clients can reassemble a
    multi-byte character (CJK / Japanese / emoji / accented) that byte-level BPE
    (Qwen/Llama-3/GPT) split across several tokens. `tokenizer.decode([tid])` for a
    lone byte-fragment returns U+FFFD (�), so `decode([tid]).encode("utf-8")` emits the
    REPLACEMENT-char bytes [239,191,189] and the original bytes are lost — the client
    can never reconstruct the character. Recover the real bytes from the GPT-2 surface
    form via the reverse byte map; fall back to the decoded string for non-byte-level
    tokenizers (sentencepiece <0xXX> byte tokens are handled explicitly).
    """
    try:
        surface = tokenizer.convert_ids_to_tokens(token_id)
    except Exception:
        surface = None
    if isinstance(surface, str) and surface:
        # sentencepiece byte-fallback token "<0xE8>" — checked FIRST, since its chars are
        # all printable ASCII and would otherwise pass the GPT-2 map check below and be
        # mis-decoded to the literal string's bytes.
        if len(surface) == 6 and surface.startswith("<0x") and surface.endswith(">"):
            try:
                return [int(surface[3:-1], 16)]
            except ValueError:
                pass
        # GPT-2 / tiktoken byte-level BPE (Qwen, Llama-3, GPT): every surface char maps
        # to exactly one raw byte. This is exact for ASCII, space-prefixed, AND split
        # multi-byte fragments. Gate on _is_byte_level_tokenizer — a
        # sentencepiece surface 'é' (U+00E9) is ALSO all-in-the-map but means the
        # character UTF-8 [195,169], NOT byte [233]; only byte-level tokenizers may
        # take this branch (see _is_byte_level_tokenizer).
        if _is_byte_level_tokenizer(tokenizer) and all(c in _GPT2_BYTE_DECODER for c in surface):
            return [_GPT2_BYTE_DECODER[c] for c in surface]
    # Fallback (non-byte-level tokenizers, or recovery failed): the decoded string's
    # UTF-8 (correct for whole, non-split tokens — the common case there).
    s = fallback_str
    if s is None:
        try:
            s = tokenizer.decode([token_id])
        except Exception:
            s = ""
    return list(s.encode("utf-8")) if s else []
