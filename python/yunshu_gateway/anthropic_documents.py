"""Bounded document ingestion and source-checked Anthropic citation round trips.

PDFs expose their text layer and page images to vision models. Image-only PDFs
require a vision model instead of silently sending base64 to a text model. Citation markers are accepted only for
exact, in-range source spans, and are translated to the public citation schema.
"""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, field

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

_MAX_BYTES = 32 * 1024 * 1024
_MAX_CHARS = 2 * 1024 * 1024
_MARK = re.compile(r"\[\[cite:(\d+):(\d+):(\d+)\]\]")


@dataclass
class Document:
    index: int
    title: str | None
    text: str
    kind: str
    spans: list[tuple[int, int]]
    enabled: bool
    images: list[dict] = field(default_factory=list)

    def citation(self, start, end):
        if not (0 <= start < end <= len(self.text)):
            raise HTTPException(502, "Model returned an out-of-range citation")
        base = {
            "document_index": self.index,
            "document_title": self.title,
            "cited_text": self.text[start:end],
        }
        if self.kind == "text":
            return {
                **base,
                "type": "char_location",
                "start_char_index": start,
                "end_char_index": end,
            }
        matching = [i for i, (a, b) in enumerate(self.spans) if a < end and b > start]
        if not matching:
            raise HTTPException(502, "Citation does not cover document content")
        if self.kind == "pdf":
            return {
                **base,
                "type": "page_location",
                "start_page_number": matching[0] + 1,
                "end_page_number": matching[-1] + 2,
            }
        return {
            **base,
            "type": "content_block_location",
            "start_block_index": matching[0],
            "end_block_index": matching[-1] + 1,
        }


def pdf_text(data):
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted or len(reader.pages) > 100:
            raise ValueError("PDF must be unencrypted and contain at most 100 pages")
        texts = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise HTTPException(400, f"Cannot read PDF: {exc}") from exc
    return texts


def pdf_images(data):
    import pypdfium2 as pdfium

    images, total = [], 0
    try:
        with pdfium.PdfDocument(data) as pdf:
            if len(pdf) > 100:
                raise ValueError("PDF must contain at most 100 pages")
            for page_index in range(len(pdf)):
                page = pdf[page_index]
                try:
                    scale = min(1.5, 1536 / max(page.get_size()))
                    bitmap = page.render(scale=scale)
                    try:
                        image = bitmap.to_pil()
                        buffer = io.BytesIO()
                        image.save(buffer, format="PNG")
                        image.close()
                    finally:
                        bitmap.close()
                finally:
                    page.close()
                png = buffer.getvalue()
                total += len(png)
                if total > _MAX_BYTES:
                    raise HTTPException(413, "Rendered PDF images exceed 32 MiB")
                images.extend(
                    [
                        {"type": "text", "text": f"[PDF page {page_index + 1}]"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(png).decode(),
                            },
                        },
                    ]
                )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Cannot render PDF: {exc}") from exc
    return images


async def read_document(block, index, render_pdf_images=False):
    source = block.get("source") or {}
    kind = source.get("type")
    media = source.get("media_type")
    images = []
    if kind == "url":
        import tempfile
        from pathlib import Path

        from yunshu_engine import netguard

        with tempfile.TemporaryDirectory(prefix="yunshu_document_") as directory:
            path = Path(directory) / "document.pdf"
            await netguard.download_to_file(
                source.get("url", ""),
                str(path),
                max_bytes=_MAX_BYTES,
                timeout=30,
                allow_private=False,
            )
            data = path.read_bytes()
        kind, media = "base64", "application/pdf"
    elif kind == "base64":
        encoded = source.get("data") or ""
        if not isinstance(encoded, str) or len(encoded) > (_MAX_BYTES * 4 // 3 + 4):
            raise HTTPException(413, "Document exceeds 32 MiB")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, "Invalid document base64") from exc
    if kind == "text" and media == "text/plain":
        parts, doc_kind = [source.get("data", "")], "text"
    elif kind == "content":
        content = source.get("content", [])
        if isinstance(content, str):
            parts = [content]
        elif isinstance(content, list) and all(
            isinstance(p, dict) and p.get("type") in ("text", "image") for p in content
        ):
            parts = [
                p.get("text", "") if p.get("type") == "text" else "" for p in content
            ]
            images = [p for p in content if p.get("type") == "image"]
        else:
            raise HTTPException(400, "Document content supports text and image blocks")
        doc_kind = "content"
    elif kind == "base64" and media == "application/pdf":
        import asyncio

        parts, doc_kind = await asyncio.to_thread(pdf_text, data), "pdf"
        if render_pdf_images:
            images = await asyncio.to_thread(pdf_images, data)
        elif not any(p.strip() for p in parts):
            raise HTTPException(400, "Image-only PDF requires a vision model")
    else:
        raise HTTPException(
            400, "Unsupported document source; use text/plain, PDF, or text content"
        )
    if not all(isinstance(p, str) for p in parts):
        raise HTTPException(400, "Document text must be a string")
    text, spans = "", []
    for part in parts:
        start = len(text)
        text += part
        spans.append((start, len(text)))
        if len(text) > _MAX_CHARS:
            raise HTTPException(413, "Extracted document exceeds 2 million characters")
    return Document(
        index,
        block.get("title"),
        text,
        doc_kind,
        spans,
        bool((block.get("citations") or {}).get("enabled")),
        images,
    )


def attach_citations(content, documents):
    out = []
    for block in content:
        if block.get("type") != "text":
            out.append(block)
            continue
        text = block.get("text", "")
        cursor = 0
        for match in _MARK.finditer(text):
            index, start, end = map(int, match.groups())
            if index >= len(documents) or not documents[index].enabled:
                raise HTTPException(
                    502, "Model referenced an unavailable citation source"
                )
            citation = documents[index].citation(start, end)
            span = text[cursor : match.start()]
            if span:
                out.append(
                    {
                        **block,
                        "type": "text",
                        "text": span,
                        "citations": [*(block.get("citations") or []), citation],
                    }
                )
            elif out and out[-1].get("type") == "text":
                out[-1].setdefault("citations", []).append(citation)
            else:
                raise HTTPException(502, "Citation has no preceding response text")
            cursor = match.end()
        if cursor < len(text) or cursor == 0:
            out.append({**block, "text": text[cursor:]})
    return out


def has_documents(value):
    if isinstance(value, list):
        return any(has_documents(v) for v in value)
    if isinstance(value, dict):
        return value.get("type") == "document" or has_documents(value.get("content"))
    return False


async def prepare_documents(req, render_pdf_images=False):
    from .routers.anthropic import AnthropicMessage

    docs = []

    async def render(blocks):
        out = []
        for block in blocks:
            if not isinstance(block, dict):
                out.append(block)
                continue
            if block.get("type") == "tool_result" and isinstance(
                block.get("content"), list
            ):
                out.append({**block, "content": await render(block["content"])})
                continue
            if block.get("type") != "document":
                out.append(block)
                continue
            try:
                doc = await read_document(block, len(docs), render_pdf_images)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(400, f"Cannot load document: {exc}") from exc
            docs.append(doc)
            text = f"[Document {doc.index}, title={json.dumps(doc.title)}, context={json.dumps(block.get('context'))}]\n{doc.text}"
            if doc.enabled and doc.text:
                text += f"\nTo cite this source, append [[cite:{doc.index}:START:END]] to the relevant statement. START and END are exact zero-based character offsets in the document text, end exclusive."
            out.append({"type": "text", "text": text})
            out.extend(doc.images)
            out.append(
                {
                    "type": "text",
                    "text": f"[End document {doc.index}]",
                    **(
                        {"cache_control": block["cache_control"]}
                        if "cache_control" in block
                        else {}
                    ),
                }
            )
        return out

    messages = [
        AnthropicMessage(
            role=m.role,
            content=await render(m.content)
            if isinstance(m.content, list)
            else m.content,
        )
        for m in req.messages
    ]
    return req.model_copy(update={"messages": messages}), docs


async def create_documents(req, request, inner):
    from .engine import get_engine, get_model_manager
    from .routers.anthropic import _resolve_engine

    vision = bool(getattr(get_engine(), "has_vision", False))
    if get_engine() is None and get_model_manager() is not None:
        engine, _ = await _resolve_engine(req.model)
        vision = bool(getattr(engine, "has_vision", False))
    adapted, docs = await prepare_documents(req, vision)
    if req.stream and not any(d.enabled for d in docs):
        # No citation to attach: stream the generation as it happens.
        return await inner(adapted, request)
    adapted = adapted.model_copy(update={"stream": False})
    response = await inner(adapted, request)
    if response.status_code != 200:
        return response
    body = json.loads(response.body)
    body["content"] = attach_citations(body.get("content", []), docs)
    if not req.stream:
        return JSONResponse(body)
    return StreamingResponse(replay(body), media_type="text/event-stream")


async def replay(body):
    def event(kind, **data):
        return f"event: {kind}\ndata: {json.dumps({'type': kind, **data}, ensure_ascii=False)}\n\n"

    yield event(
        "message_start",
        message={
            **body,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {**body["usage"], "output_tokens": 0},
        },
    )
    for index, block in enumerate(body["content"]):
        kind = block["type"]
        initial = dict(block)
        if kind == "text":
            initial.update(text="", citations=[])
        elif kind == "tool_use":
            initial["input"] = {}
        elif kind == "thinking":
            initial.update(thinking="", signature="")
        yield event("content_block_start", index=index, content_block=initial)
        if kind == "text":
            yield event(
                "content_block_delta",
                index=index,
                delta={"type": "text_delta", "text": block["text"]},
            )
            for citation in block.get("citations", []):
                yield event(
                    "content_block_delta",
                    index=index,
                    delta={"type": "citations_delta", "citation": citation},
                )
        elif kind == "tool_use":
            yield event(
                "content_block_delta",
                index=index,
                delta={
                    "type": "input_json_delta",
                    "partial_json": json.dumps(block["input"]),
                },
            )
        elif kind == "thinking":
            yield event(
                "content_block_delta",
                index=index,
                delta={"type": "thinking_delta", "thinking": block["thinking"]},
            )
            yield event(
                "content_block_delta",
                index=index,
                delta={
                    "type": "signature_delta",
                    "signature": block.get("signature", ""),
                },
            )
        yield event("content_block_stop", index=index)
    yield event(
        "message_delta",
        delta={
            "stop_reason": body["stop_reason"],
            "stop_sequence": body.get("stop_sequence"),
        },
        usage=body["usage"],
    )
    yield event("message_stop")
