from __future__ import annotations

"""Engine text extracted from batched_engine.

Runtime dependencies stay on the compatibility facade so existing patches apply.
"""


def _resolve_think_token_ids(tokenizer):
    """Resolve the single-token ids for the <think> / </think> markers, or (None, None).

    The old call sites encoded "<think" / "</think" WITHOUT the closing '>',
    which for the canonical thinking models (Qwen3/Qwen3.5/DeepSeek-R1) tokenizes to TWO
    tokens (e.g. Qwen3.5 `</think` → [510, 26003]) while the model actually emits the
    SINGLE special token `</think>` (with bracket, e.g. 248069). So the `len == 1` guard
    failed, think_end_token became None, the streaming reasoning-state machine never
    engaged, and the ENTIRE chain-of-thought (plus the literal markup) leaked into
    delta.content with reasoning_tokens=0 — defeating the streaming fixes on the
    DEFAULT path for the most common reasoning models. Encode the BRACKETED form with
    add_special_tokens=False (so a BOS-prepending tokenizer doesn't inflate the length).
    """

    def _enc(s: str):
        try:
            return tokenizer.encode(s, add_special_tokens=False)
        except TypeError:
            # Some wrappers don't accept the kwarg — fall back, then strip a leading BOS.
            ids = tokenizer.encode(s)
            bos = getattr(tokenizer, "bos_token_id", None)
            if bos is not None and len(ids) > 1 and ids[0] == bos:
                ids = ids[1:]
            return ids
        except Exception:
            return None

    try:
        _ts = _enc("<think>")
        _te = _enc("</think>")
        if _ts and _te and len(_ts) == 1 and len(_te) == 1:
            return _ts[0], _te[0]
        # Gemma-4 reasoning uses channel SPECIAL TOKENS instead of <think>:
        # <|channel>thought…<channel|> are SINGLE ids (100/101) that decode to ''
        # under skip_special_tokens — so they vanish from the text before any
        # string-based <think> logic sees them, and the whole chain-of-thought
        # leaks into content with reasoning_tokens=0 (observed on gemma-4 via the
        # chat() path). The streaming reasoning machine is TOKEN-ID based, so
        # mapping <|channel>→start and <channel|>→end makes it segment the channel
        # reasoning correctly (content stays clean, reasoning → reasoning_content).
        # Only reached when <think>/</think> are NOT single tokens, so canonical
        # thinking models are unaffected.
        _co = _enc("<|channel>")
        _cc = _enc("<channel|>")
        if _co and _cc and len(_co) == 1 and len(_cc) == 1:
            return _co[0], _cc[0]
    except Exception:
        _engine.logger.debug("thinking token resolve failed", exc_info=True)
    return None, None


def _recover_channel_reasoning(tokens, tokenizer, output_text):
    """Recover Gemma-4-style channel reasoning that the detokenizer stripped.

    Gemma-4 emits reasoning inside <|channel>thought…<channel|> blocks whose
    delimiters are SINGLE special tokens that decode to '' under
    skip_special_tokens — so they vanish from ``output_text`` before any
    <think>-based splitter (the engine reasoning_parser AND the gateway's
    extract_thinking) can see them, and the whole chain-of-thought leaks into
    content. _resolve_think_token_ids already maps these channel ids onto the
    think-start/end slots, so re-segment the RAW token list by those ids and
    rebuild output_text in canonical <think>…</think> form, which the gateway's
    extract_thinking strips (gemma's ENGINE parser leaves it) → content ends
    clean + reasoning_content populated.

    Gated on the marker decoding to '' under skip_special_tokens, so canonical
    <think> models (whose markers survive in output_text and are already handled)
    are untouched. Returns (output_text, reason_token_ids); reason ids empty when
    no recovery happened.
    """
    tk_start, tk_end = _engine._resolve_think_token_ids(tokenizer)
    if tk_start is None or tk_start not in tokens:
        return output_text, []
    try:
        _survives = tokenizer.decode([tk_start], skip_special_tokens=True).strip()
    except TypeError:
        _survives = tokenizer.decode([tk_start]).strip()
    if _survives:
        # Marker survives decode → canonical <think> model, already handled.
        return output_text, []
    # Segment the raw tokens into ORDERED runs (gemma-4 can interleave several
    # channel blocks with plain content: thought₁ → content₁ → thought₂ → …).
    # Each reasoning block is decoded + label-stripped INDEPENDENTLY — joining
    # all reason ids first (the old approach) stripped the "thought" label only
    # once, so blocks 2+ left an embedded "thought" in the reasoning, and the
    # interleaved structure was lost.
    content_ids: list[int] = []
    reason_ids: list[int] = []
    reason_parts: list[str] = []
    depth = 0
    run: list[int] = []
    run_is_reason = False

    def _flush_run(ids: list[int], is_reason: bool) -> None:
        if not ids:
            return
        if is_reason:
            reason_ids.extend(ids)
            txt = _engine._clean_special_tokens(tokenizer.decode(ids)).strip()
            if txt.startswith("thought"):  # drop this block's channel label
                txt = txt[len("thought") :].lstrip(" \n")
            if txt:
                reason_parts.append(txt)
        else:
            content_ids.extend(ids)

    for t in tokens:
        if t == tk_start:
            _flush_run(run, run_is_reason)
            run = []
            depth += 1
            run_is_reason = depth > 0
            continue
        if tk_end is not None and t == tk_end and depth > 0:
            _flush_run(run, run_is_reason)
            run = []
            depth -= 1
            run_is_reason = depth > 0
            continue
        run.append(t)
    _flush_run(run, run_is_reason)

    if not reason_ids:
        return output_text, []
    # content fragments were separated by channel blocks in the stream — join as
    # emitted (no synthetic spacing); reasoning blocks join on newlines.
    content = _engine._clean_special_tokens(tokenizer.decode(content_ids)).strip()
    reason = "\n".join(reason_parts)
    new_text = f"<think>{reason}</think>{content}" if reason else content
    return new_text, (reason_ids if reason else [])


def _template_supports_tools(tokenizer) -> bool:
    """True if the tokenizer's chat template natively renders a ``tools`` variable
    (Qwen3, Llama-3.1, Hermes, Mistral, GLM, …). When False, callers fall back to the
    generic injected tool system-prompt. Cheap string check on the Jinja template."""
    tmpl = getattr(tokenizer, "chat_template", None)
    return isinstance(tmpl, str) and "tools" in tmpl


def _clean_special_tokens(text: str) -> str:
    """Remove special tokens from output ."""
    if not text:
        return ""
    import re

    text = re.sub(r"<\|im_end\|>", "", text)
    text = re.sub(r"<\|endoftext\|>", "", text)
    text = re.sub(r"<\|end\|>", "", text)
    return text


# Resolve the facade after definitions to also support direct module imports.
from . import batched_engine as _engine
