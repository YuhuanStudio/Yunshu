# Upstream (inspired): jundot/omlx (Apache-2.0) PR #4315 design only (code not read): /v1/systemone beside a Clef decision engine
from __future__ import annotations

"""Decisions: typed questions answered with probabilities.

``POST /v1/decisions``  OpenAI's Decisions API (wire format from openai-python 3.26 types).
``POST /v1/systemone``  TypeSafe Jev / System One wire (map-keyed questions, Choice/Score/Noul).

Both translate into one internal ``DecisionRequest`` (yunshu_engine.decision_engine) and are
answered by a ``DecisionEngine``. Stateless, non-streaming, no generation: output_tokens is 0.
"""
import base64
import binascii
import logging
import re
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    ValidationError,
    model_validator,
)

from yunshu_engine.decision_engine import (
    MAX_IMAGES,
    DecisionError,
    DecisionRequest,
    DecisionResult,
    Option,
    Question,
    question_ids,
)

from .models import _check_model_access, _check_permission

logger = logging.getLogger(__name__)

router = APIRouter(tags=["decisions"])

_MAX_QUESTIONS = 128
_MAX_OPTIONS = 64
_MAX_TEXT_CHARS = 1_000_000
_DATA_URL = re.compile(r"^data:image/[A-Za-z0-9.+-]+;base64,(.*)$", re.DOTALL)


def _text(v: str, field: str) -> str:
    if len(v) > _MAX_TEXT_CHARS:
        raise ValueError(f"{field}: longer than {_MAX_TEXT_CHARS} characters")
    return v


def decode_image(data_url: str, where: str) -> bytes:
    """Bytes of a base64 data-URL image. External URLs and file ids are refused, as on OpenAI."""
    m = _DATA_URL.match(data_url.strip())
    if not m:
        raise ValueError(
            f"{where}: only inline base64 images (data:image/...;base64,...) are supported; "
            "external URLs and file ids are not"
        )
    try:
        raw = base64.b64decode(m.group(1), validate=False)
    except (binascii.Error, ValueError):
        raise ValueError(f"{where}: invalid base64") from None
    if not raw:
        raise ValueError(f"{where}: empty image")
    return raw


# ── OpenAI wire ──────────────────────────────────────────────────────────────────────────


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InputText(_Strict):
    type: Literal["input_text"]
    text: str


class InputImage(_Strict):
    type: Literal["input_image"]
    image_url: str
    detail: Literal["low", "high", "auto", "original"] | None = None


class InputMessage(_Strict):
    role: Literal["user"]
    type: Literal["message"] | None = None
    content: str | list[Annotated[InputText | InputImage, Field(discriminator="type")]]


class PredicateQ(_Strict):
    type: Literal["predicate"]
    instructions: str
    name: str | None = None


class ChoiceItem(_Strict):
    value: StrictStr | StrictBool
    description: str | None = None


class ChoiceQ(_Strict):
    type: Literal["choice"]
    instructions: str
    name: str | None = None
    choices: list[ChoiceItem]


class ScoreLevel(_Strict):
    label: str
    description: str | None = None


class ScoreQ(_Strict):
    type: Literal["score"]
    instructions: str
    name: str | None = None
    levels: list[ScoreLevel]


class DecisionCreate(BaseModel):
    # extra top-level keys are ignored (SDK extra_body); question objects are strict
    model: str
    input: str | list[InputMessage]
    questions: list[
        Annotated[PredicateQ | ChoiceQ | ScoreQ, Field(discriminator="type")]
    ]
    safety_identifier: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def _check(self):
        if not self.model.strip():
            raise ValueError("model: field is required and cannot be empty")
        if not self.questions:
            raise ValueError("questions: at least one question is required")
        if len(self.questions) > _MAX_QUESTIONS:
            raise ValueError(
                f"questions: at most {_MAX_QUESTIONS} questions per request"
            )
        return self


def _choice_id(value: str | bool) -> str:
    return ("true" if value else "false") if isinstance(value, bool) else value


def _level_text(level: ScoreLevel) -> str:
    # The head was trained on level DESCRIPTIONS; the label stands in when there is none.
    return level.description or level.label


def to_internal(body: DecisionCreate) -> DecisionRequest:
    if isinstance(body.input, str):
        texts, images = [body.input], []
    else:
        texts, images = [], []
        for mi, msg in enumerate(body.input):
            parts = [msg.content] if isinstance(msg.content, str) else msg.content
            for pi, part in enumerate(parts):
                if isinstance(part, str):
                    texts.append(part)
                elif part.type == "input_text":
                    texts.append(part.text)
                else:
                    images.append(
                        decode_image(part.image_url, f"input[{mi}].content[{pi}]")
                    )
    if len(images) > MAX_IMAGES:
        raise ValueError(f"input: at most {MAX_IMAGES} images per request")
    state = _text("\n".join(texts), "input")
    if not state.strip() and not images:
        raise ValueError("input: cannot be empty")

    questions: list[Question] = []
    for qi, q in enumerate(body.questions):
        where = f"questions[{qi}]"
        if isinstance(q, PredicateQ):
            opts = (Option("true", True), Option("false", False))
        elif isinstance(q, ChoiceQ):
            if len(q.choices) < 2:
                raise ValueError(f"{where}.choices: at least 2 choices are required")
            if len(q.choices) > _MAX_OPTIONS:
                raise ValueError(f"{where}.choices: at most {_MAX_OPTIONS} choices")
            ids = [_choice_id(c.value) for c in q.choices]
            if len(set(ids)) != len(ids):
                raise ValueError(
                    f"{where}.choices: values must be distinct (the model sees booleans as "
                    "'true'/'false', so a string 'true' and a boolean true collide)"
                )
            opts = tuple(
                Option(i, c.value, c.description)
                for i, c in zip(ids, q.choices, strict=True)
            )
        else:
            if len(q.levels) < 2:
                raise ValueError(f"{where}.levels: at least 2 levels are required")
            if len(q.levels) > _MAX_OPTIONS:
                raise ValueError(f"{where}.levels: at most {_MAX_OPTIONS} levels")
            if len({lv.label for lv in q.levels}) != len(q.levels):
                raise ValueError(f"{where}.levels: labels must be distinct")
            opts = tuple(
                Option(str(i), lv.label, _level_text(lv))
                for i, lv in enumerate(q.levels)
            )
        questions.append(
            Question(
                q.type, q.name, _text(q.instructions, f"{where}.instructions"), opts
            )
        )
    question_ids(questions)  # unique names, or a 400 before the engine is touched
    return DecisionRequest(body.model, state, questions, images)


def decision_response(req: DecisionRequest, res: DecisionResult) -> dict[str, Any]:
    """The ``Decision`` object: answers in question order, then usage."""
    answers: list[dict[str, Any]] = []
    for q, probs in zip(req.questions, res.probabilities, strict=True):
        if probs is None:
            answers.append({"type": "refusal", "name": q.name})
        elif q.kind == "predicate":
            answers.append(
                {"type": "predicate", "name": q.name, "probability": probs[0]}
            )
        elif q.kind == "choice":
            best = max(range(len(probs)), key=probs.__getitem__)
            answers.append(
                {
                    "type": "choice",
                    "name": q.name,
                    "choice": q.options[best].value,
                    "confidence": probs[best],
                    "probabilities": [
                        {"value": o.value, "probability": p}
                        for o, p in zip(q.options, probs, strict=True)
                    ],
                }
            )
        else:
            answers.append(
                {
                    "type": "score",
                    "name": q.name,
                    "score": sum(i * p for i, p in enumerate(probs)),
                    "confidence": max(probs),
                    "probabilities": [
                        {"label": o.value, "value": i, "probability": p}
                        for i, (o, p) in enumerate(zip(q.options, probs, strict=True))
                    ],
                }
            )
    return {"answers": answers, "model": req.model, "usage": _usage(res)}


def _usage(res: DecisionResult) -> dict[str, Any]:
    return {
        "input_tokens": res.input_tokens,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": res.input_tokens,
    }


# ── System One (Jev) wire ────────────────────────────────────────────────────────────────


class SystemOneQuestion(_Strict):
    type: Literal["noul", "choice", "score"]
    instructions: str | None = None
    criteria: dict[str, str | None] | list[str] | None = None


class SystemOneRequest(BaseModel):
    model: str
    state: Any
    questions: dict[str, SystemOneQuestion]
    images: list[str] | None = None

    @model_validator(mode="after")
    def _check(self):
        if not self.model.strip():
            raise ValueError("model: field is required and cannot be empty")
        if not self.questions:
            raise ValueError("questions: at least one question is required")
        if len(self.questions) > _MAX_QUESTIONS:
            raise ValueError(
                f"questions: at most {_MAX_QUESTIONS} questions per request"
            )
        return self


def systemone_to_internal(body: SystemOneRequest) -> DecisionRequest:
    images = []
    for i, img in enumerate(body.images or []):
        url = img if img.startswith("data:") else f"data:image/png;base64,{img}"
        images.append(decode_image(url, f"images[{i}]"))
    if len(images) > MAX_IMAGES:
        raise ValueError(f"images: at most {MAX_IMAGES} images per request")
    questions: list[Question] = []
    for qid, q in body.questions.items():
        where = f"questions.{qid}"
        crit = q.criteria
        if q.type == "noul":
            if isinstance(crit, list):
                raise ValueError(
                    f"{where}.criteria: a noul takes {{'true': ..., 'false': ...}}"
                )
            extra = set(crit or {}) - {"true", "false"}
            if extra:
                raise ValueError(f"{where}.criteria: unknown keys {sorted(extra)}")
            opts = (
                Option("true", True, (crit or {}).get("true")),
                Option("false", False, (crit or {}).get("false")),
            )
            kind = "predicate"
        elif q.type == "choice":
            if not isinstance(crit, dict) or len(crit) < 2:
                raise ValueError(
                    f"{where}.criteria: a choice needs a map of at least 2 options"
                )
            opts = tuple(Option(k, k, v) for k, v in crit.items())
            kind = "choice"
        else:
            if isinstance(crit, dict):
                crit = list(crit.values())
            if not crit or len(crit) < 2:
                raise ValueError(
                    f"{where}.criteria: a score needs a list of at least 2 levels"
                )
            if any(c is None for c in crit):
                raise ValueError(f"{where}.criteria: score levels must be strings")
            opts = tuple(Option(str(i), str(c), str(c)) for i, c in enumerate(crit))
            kind = "score"
        if len(opts) > _MAX_OPTIONS:
            raise ValueError(f"{where}.criteria: at most {_MAX_OPTIONS} options")
        questions.append(Question(kind, qid, q.instructions or qid, opts))
    return DecisionRequest(body.model, body.state, questions, images)


def systemone_response(req: DecisionRequest, res: DecisionResult) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    for q, probs in zip(req.questions, res.probabilities, strict=True):
        qid = q.name or ""
        if probs is None:
            answers[qid] = {"type": "refusal"}
        elif q.kind == "predicate":
            answers[qid] = {"type": "noul", "noul": round(probs[0], 4)}
        elif q.kind == "choice":
            best = max(range(len(probs)), key=probs.__getitem__)
            answers[qid] = {
                "type": "choice",
                "choice": q.options[best].id,
                "confidence": round(probs[best], 4),
                "probabilities": {
                    o.id: round(p, 4) for o, p in zip(q.options, probs, strict=True)
                },
            }
        else:
            levels = [str(i) for i in range(len(probs))]
            answers[qid] = {
                "type": "score",
                "score": round(sum(i * p for i, p in enumerate(probs)), 4),
                "confidence": round(max(probs), 4),
                "legend": {
                    lv: o.description for lv, o in zip(levels, q.options, strict=True)
                },
                "probabilities": {
                    lv: round(p, 4) for lv, p in zip(levels, probs, strict=True)
                },
            }
    return {
        "model": req.model,
        "answers": answers,
        "usage": {"input_tokens": res.input_tokens, "output_tokens": 0},
    }


# ── serving ──────────────────────────────────────────────────────────────────────────────


async def _resolve_decision_engine(model_id: str):
    from ..engine import get_engine, get_model_manager

    manager = get_model_manager()
    if manager is not None:
        entry = manager.get_entry(model_id)
        if entry is not None and entry.is_loaded and entry.engine is not None:
            return entry.engine
        try:
            return await manager.get_engine(model_id)
        except KeyError:
            pass
        except Exception:
            logger.warning(
                "Failed to load engine for decision model %r", model_id, exc_info=True
            )
            return None
    engine = get_engine()
    if engine and engine.is_loaded:
        return engine
    return None


def _validation_message(e: ValidationError) -> str:
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err.get("loc", []))
        msg = str(err.get("msg", "invalid")).removeprefix("Value error, ")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts)


async def _serve(request: Request, parse, translate, render, bad_status: int):
    _check_permission(request, "can_infer")
    try:
        raw = await request.json()
    except ValueError:
        raise HTTPException(bad_status, "request body is not valid JSON") from None
    if not isinstance(raw, dict):
        raise HTTPException(bad_status, "request body must be a JSON object")
    try:
        body = parse(raw)
        req = translate(body)
    except ValidationError as e:
        raise HTTPException(bad_status, _validation_message(e)) from None
    except ValueError as e:
        raise HTTPException(bad_status, str(e)) from None
    _check_model_access(request, req.model)

    from yunshu_engine.decision_engine import DecisionEngine

    engine = await _resolve_decision_engine(req.model)
    if engine is None:
        raise HTTPException(404, f"Model '{req.model}' not found")
    if not isinstance(engine, DecisionEngine):
        raise HTTPException(
            400,
            f"Model '{req.model}' is not a decision model; load a decision checkpoint "
            "(for example Cloudflare Clef in MLX format).",
        )
    if req.images and not engine.supports_multimodal:
        raise HTTPException(400, f"Model '{req.model}' does not accept images")
    try:
        res = await engine.decide(req)
    except DecisionError as e:
        raise HTTPException(400, str(e)) from None
    except MemoryError:
        raise HTTPException(507, "Out of GPU memory") from None
    except Exception:
        logger.error("Decision failed", exc_info=True)
        raise HTTPException(500, "Decision failed") from None
    return JSONResponse(render(req, res))


@router.post("/decisions", response_model=None)
async def create_decision(request: Request):
    return await _serve(
        request, DecisionCreate.model_validate, to_internal, decision_response, 400
    )


@router.post("/systemone", response_model=None)
async def create_systemone(request: Request):
    return await _serve(
        request,
        SystemOneRequest.model_validate,
        systemone_to_internal,
        systemone_response,
        422,
    )
