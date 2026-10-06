"""Prompt cache intent and diagnostic rendering; never modify the model's prompt.

Markers ride through the gateway's existing conversion, then live only in a
shadow request. The renderer must reproduce the unmarked prompt byte for byte.
The full tokenization determines seams; a token straddling a seam is replayed.
"""

from __future__ import annotations

import copy
import json
import uuid
from bisect import bisect_right


def ttl(control: dict) -> int:
    if control.get("type") != "ephemeral" or control.get("ttl", "5m") not in (
        "5m",
        "1h",
    ):
        raise ValueError("cache_control requires type=ephemeral and ttl=5m or 1h")
    return 3600 if control.get("ttl") == "1h" else 300


def mark_anthropic(req) -> dict:
    """Add diagnostic markers before conversion; caller strips them before serving."""
    markers: dict[str, int] = {}
    tools = {}
    marker_ends = {}
    write_markers = set()
    write_tools = set()
    source_blocks = [
        b
        for c in [req.system, *(m.content for m in req.messages)]
        if isinstance(c, list)
        for b in c
        if isinstance(b, dict)
    ]
    active = bool(
        getattr(req, "cache_control", None)
        or any(b.get("cache_control") for b in source_blocks)
        or any(getattr(t, "cache_control", None) for t in (req.tools or []))
    )
    nonce = "YUNSHUCACHE" + uuid.uuid4().hex

    def blocks(content):
        content = copy.deepcopy(content)
        if not active:
            return content
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            return content
        for block in content:
            control = block.get("cache_control")
            if not control and block.get("type") not in (
                "text",
                "tool_use",
                "tool_result",
                "image",
                "document",
            ):
                continue
            marker = nonce + str(len(markers)) + "END"
            duration = ttl(control) if control else 300
            if control:
                write_markers.add(marker)
            # These fields are preserved by the existing converter. Other blocks
            # are resolved after rendering through their serialized representation.
            key = "text" if block.get("type") == "text" else None
            if key and not block.get(key) and not control:
                continue
            if (
                block.get("type") == "document"
                and (block.get("source") or {}).get("type") == "text"
            ):
                block["_yunshu_cache_marker"] = marker
                markers[marker] = duration
            elif key:
                block[key] = block.get(key, "") + marker
                markers[marker] = duration
            elif block.get("type") == "tool_result" and isinstance(
                block.get("content"), str
            ):
                block["content"] += marker
                markers[marker] = duration
            elif block.get("type") == "tool_use":
                block["_yunshu_cache_marker"] = marker
                markers[marker] = duration
                marker_ends[marker] = "tool_use"
            elif block.get("type") == "thinking":
                block["thinking"] = block.get("thinking", "") + marker
                markers[marker] = duration
            elif block.get("type") == "image":
                block["_yunshu_cache_marker"] = marker
                markers[marker] = duration
            elif block.get("type") == "tool_result" and isinstance(
                block.get("content"), list
            ):
                inner = block["content"]
                last = next(
                    (b for b in reversed(inner) if b.get("type") == "text"), None
                )
                if last is None or inner[-1] is not last:
                    if not control:
                        continue
                    raise ValueError(
                        "image-only tool_result cache endpoint is not supported"
                    )
                last["text"] += marker
                markers[marker] = duration
            else:
                raise ValueError(
                    "cache_control block has no supported rendered endpoint"
                )
        return content

    automatic = getattr(req, "cache_control", None)
    if automatic:
        # Automatic caching moves to the last eligible source content block;
        # an identical explicit marker there uses the same write slot.
        duration = ttl(automatic)
        for message in reversed(req.messages):
            content = copy.deepcopy(message.content)
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            if not isinstance(content, list):
                continue
            candidate = next(
                (
                    b
                    for b in reversed(content)
                    if b.get("type") in ("text", "document", "tool_result")
                ),
                None,
            )
            if candidate is None:
                continue
            existing = candidate.get("cache_control")
            if existing and ttl(existing) != duration:
                raise ValueError(
                    "automatic and final explicit cache_control TTL differ"
                )
            candidate["cache_control"] = automatic
            index = req.messages.index(message)
            req.messages[index] = message.model_copy(update={"content": content})
            break
    req.system = blocks(req.system)
    req.messages = [
        m.model_copy(update={"content": blocks(m.content)}) for m in req.messages
    ]
    for i, tool in enumerate(req.tools or []):
        control = getattr(tool, "cache_control", None)
        if control:
            tools[i] = ttl(control)
            write_tools.add(i)
        elif active:
            tools[i] = 300
    if len(write_markers) + len(write_tools) > 4:
        raise ValueError("at most four cache_control breakpoints are supported")
    return {
        "markers": markers,
        "marker_ends": marker_ends,
        "write_markers": write_markers,
        "write_tools": write_tools,
        "tools": tools,
        "points": [],
        "written": 0,
        "resolved": False,
    }


def strip_markers(value, markers):
    if isinstance(value, str):
        for marker in markers:
            value = value.replace(marker, "")
        return value
    if isinstance(value, list):
        return [strip_markers(v, markers) for v in value]
    if isinstance(value, dict):
        return {k: strip_markers(v, markers) for k, v in value.items()}
    return value


def tool_boundaries(
    prompt: str, tools: list[dict], controls: dict[int, int]
) -> list[tuple[int, int]]:
    """Find complete template-rendered JSON definitions, including closing braces.

    Accept either OpenAI's function wrapper or the function itself; compare
    parsed objects so escaping, whitespace and key order cannot fool us.
    """
    decoder = json.JSONDecoder()
    found = {}
    cursor = 0
    for i, tool in enumerate(tools):
        for at in range(cursor, len(prompt)):
            if prompt[at] != "{":
                continue
            try:
                obj, n = decoder.raw_decode(prompt[at:])
            except ValueError:
                continue
            if obj == tool or obj == tool.get("function"):
                cursor = at + n
                if i in controls:
                    found[i] = (cursor, controls[i])
                break
        else:
            if i in controls:
                raise ValueError(
                    "tool cache breakpoint is absent from rendered template"
                )
    return list(found.values())


def token_boundaries(prompt, char_points, tokenizer, ids, *, return_map=False):
    """Map rendered character endpoints against the actual full token IDs."""
    # A fast tokenizer's full-prompt offsets handle byte-fallback and non-ASCII
    # token seams. Verify its IDs against the engine's tokenization first.
    raw = getattr(tokenizer, "_tokenizer", tokenizer)
    offsets = None
    if callable(raw):
        try:
            enc = raw(prompt, add_special_tokens=False, return_offsets_mapping=True)
            plain = list(enc["input_ids"])
            shift = len(ids) - len(plain)
            if shift in (0, 1) and list(ids[shift:]) == plain:
                offsets = [(0, 0)] * shift + list(enc["offset_mapping"])
        except (TypeError, ValueError, NotImplementedError, KeyError):
            pass
    points = []
    endpoint_map = {}
    for char_end, duration in char_points:
        if offsets is not None:
            # Prefix only: a special token with offset (0,0) later in the
            # sequence must never jump over intervening uncached content.
            n = 0
            for i, (start, end) in enumerate(offsets):
                if end > char_end or start > char_end:
                    break
                n = i + 1
        else:
            # No substring count: retain only IDs identical to the full render.
            try:
                prefix = tokenizer.encode(prompt[:char_end], add_special_tokens=False)
            except TypeError:
                prefix = tokenizer.encode(prompt[:char_end])
            n = 0
            for a, b in zip(ids, prefix, strict=False):
                if a != b:
                    break
                n += 1
        if 0 < n < len(ids):
            points.append((n, duration))
            endpoint_map[char_end] = (n, duration)
    points = sorted(set(points))
    return (points, endpoint_map) if return_map else points


def rendered_boundaries(
    prompt,
    marked_prompt,
    markers,
    tokenizer,
    ids,
    *,
    tools=None,
    tool_controls=None,
    marker_ends=None,
    selection=None,
):
    chars = []
    write_chars = set()
    marker_chars = {}
    positions = {m: marked_prompt.find(m) for m in markers}
    for marker, duration in markers.items():
        at = positions[marker]
        if at < 0 or marked_prompt.count(marker) != 1:
            raise ValueError(
                "cache breakpoint was removed or duplicated by rendered template"
                f" (marker {list(markers).index(marker) + 1} of {len(markers)}, "
                f"found {marked_prompt.count(marker)} times)"
            )
        end = at
        if (marker_ends or {}).get(marker) == "tool_use":
            close = marked_prompt.find("</tool_call>", at)
            if close >= 0:
                end = close + len("</tool_call>")
            else:
                decoder = json.JSONDecoder()
                candidates = []
                for start in range(at):
                    if marked_prompt[start] != "{":
                        continue
                    try:
                        _, length = decoder.raw_decode(marked_prompt[start:])
                    except ValueError:
                        continue
                    if start + length > at:
                        candidates.append(start + length)
                if not candidates:
                    raise ValueError(
                        "tool_use cache endpoint is absent from rendered template"
                    )
                end = max(candidates)
        end -= sum(len(m) for m, pos in positions.items() if 0 <= pos < end)
        chars.append((end, duration))
        marker_chars[marker] = end
        if selection is None or marker in selection.get("write_markers", markers):
            write_chars.add(end)
    stripped = strip_markers(marked_prompt, markers)
    if stripped != prompt:
        raise ValueError("cache diagnostic render differs from original prompt")
    if len(chars) != len(markers):
        raise ValueError("cache breakpoint was removed by the rendered template")
    tool_chars = tool_boundaries(prompt, tools or [], tool_controls or {})
    chars.extend(tool_chars)
    selected_tools = (selection or {}).get("write_tools", tool_controls or {})
    write_chars.update(
        n
        for n, _ in tool_boundaries(
            prompt,
            tools or [],
            {i: d for i, d in (tool_controls or {}).items() if i in selected_tools},
        )
    )
    points, endpoint_map = token_boundaries(
        prompt, chars, tokenizer, ids, return_map=True
    )
    if selection is None:
        return points
    writes = sorted({endpoint_map[n] for n in write_chars if n in endpoint_map})
    if selection.get("protocol") == "openai":
        lookup = {n for n, _ in writes}
        if selection.get("mode") == "implicit":
            lookup.update(
                endpoint_map[marker_chars[m]][0]
                for m in selection["eligible_markers"][-21:]
                if marker_chars[m] in endpoint_map
            )
        return {"points": points, "writes": writes, "lookup_points": sorted(lookup)}
    # Numerical seams stay stable when a moving breakpoint changes its lookup
    # window. Only requested endpoints incur a checkpoint clone/write.
    lookup = set()
    positions = sorted({n for n, _ in points})
    for n, _ in writes:
        index = positions.index(n)
        lookup.update(positions[max(0, index - 20) : index + 1])
    return {"points": points, "writes": writes, "lookup_points": sorted(lookup)}


def canonical_step_end(start: int, limit: int, step: int, boundaries=()) -> int:
    """Same absolute numerical spans for a cold request and a restored suffix."""
    end = min(limit, (start // step + 1) * step)
    idx = bisect_right(boundaries, start)
    if idx < len(boundaries):
        end = min(end, boundaries[idx])
    return end


def openai_plan(messages, options=None):
    """OpenAI explicit/implicit endpoints; old automatic APC stays the default."""
    options = options or {}
    mode = options.get("mode", "implicit")
    if mode not in ("explicit", "implicit"):
        raise ValueError("prompt_cache_options.mode must be explicit or implicit")
    marked = any(
        p.get("prompt_cache_breakpoint")
        for m in messages
        for p in (m.get("content") if isinstance(m.get("content"), list) else [])
        if isinstance(p, dict)
    )
    if not marked and not options.get("mode"):
        return None
    shadow = copy.deepcopy(messages)
    markers: dict[str, int] = {}
    writes: set[str] = set()
    eligible: list[str] = []
    head = None
    leading = True
    nonce = "YUNSHUOPENAICACHE" + uuid.uuid4().hex
    duration = options.get("ttl") or "30m"
    durations = {"30m": 1800, "24h": 86400, "in_memory": 300}
    if duration not in durations:
        raise ValueError(
            "prompt_cache_options.ttl must be 30m (legacy retention: in_memory or 24h)"
        )
    seconds = durations[duration]
    for message in shadow:
        content = message.get("content")
        last = None
        if isinstance(content, str) and content:
            marker = nonce + str(len(markers)) + "END"
            message["content"] = content + marker
            markers[marker] = seconds
            last = marker
        elif isinstance(content, list):
            for block in content:
                explicit = bool(block.get("prompt_cache_breakpoint"))
                if block.get("type") not in ("text", "input_text", "output_text"):
                    if explicit:
                        raise ValueError(
                            "OpenAI prompt_cache_breakpoint requires a text block"
                        )
                    continue
                marker = nonce + str(len(markers)) + "END"
                block["text"] = block.get("text", "") + marker
                markers[marker] = seconds
                last = marker
                if explicit:
                    writes.add(marker)
        if leading and message.get("role") in ("system", "developer"):
            head = last or head
        else:
            leading = False
        if last and message.get("role") in ("user", "tool", "system", "developer"):
            eligible.append(last)
    if mode == "implicit" and eligible:
        writes.add(eligible[-1])
        # Hybrid state cannot be sliced from the latest user endpoint back to
        # a system head. Keep the existing local APC head policy under budgets.
        if head and len(writes) < 4:
            writes.add(head)
    if len(writes) > 4:
        raise ValueError(
            "at most four prompt cache writes are supported (implicit uses one slot)"
        )
    return {
        "messages": shadow,
        "markers": markers,
        "tools": {},
        "write_markers": writes,
        "write_tools": set(),
        "eligible_markers": eligible,
        "protocol": "openai",
        "mode": mode,
        "points": [],
        "written": 0,
        "resolved": False,
    }


def expanded_boundaries(plain, expanded, points, media_ids):
    """Map verified processor repetitions without guessing placeholder lengths.

    Non-media tokens must be identical. A media run maps only its start and
    end; there is no checkpoint inside an encoded image/audio/video span.
    """
    mapping = {0: 0}
    i = j = 0
    while i < len(plain) and j < len(expanded):
        token = plain[i]
        if token != expanded[j]:
            raise ValueError(
                "processor tokens differ outside a verified media expansion"
            )
        if token in media_ids:
            a = i
            while i < len(plain) and plain[i] == token:
                i += 1
            while j < len(expanded) and expanded[j] == token:
                j += 1
            # Endpoint only: the processor's actual run length is authoritative.
            mapping[i] = j
            if i == a:
                raise ValueError("empty processor media span")
        else:
            i += 1
            j += 1
            mapping[i] = j
    if i != len(plain) or j != len(expanded):
        raise ValueError("processor tokenization does not match the rendered template")
    if any(n not in mapping for n, _ in points):
        raise ValueError("cache breakpoint cuts a processor media span")
    return [(mapping[n], duration) for n, duration in points]


def trim_safe_shadow(value, markers):
    """Let a template trim the same trailing whitespace as the real prompt."""
    if isinstance(value, str):
        for marker in markers:
            if value.endswith(marker):
                before = value[: -len(marker)]
                text = before.rstrip()
                value = text + marker + before[len(text) :]
        return value
    if isinstance(value, list):
        return [trim_safe_shadow(v, markers) for v in value]
    if isinstance(value, dict):
        return {k: trim_safe_shadow(v, markers) for k, v in value.items()}
    return value
