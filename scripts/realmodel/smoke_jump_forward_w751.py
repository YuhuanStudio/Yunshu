"""Wave 751 real-model smoke: jump-forward decoding (YUNSHU_JUMP_FORWARD=1) via
the engine path — valid schema-conforming JSON, fewer forwards than tokens.
Run: PYTHONPATH=. uv run python scripts/realmodel/smoke_jump_forward_w751.py
"""
import asyncio, json, os
MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
SCHEMA = {"type": "object",
          "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                         "city": {"type": "string"}},
          "required": ["name", "age", "city"], "additionalProperties": False}

async def _gen(jf: bool):
    os.environ["YUNSHU_JUMP_FORWARD"] = "1" if jf else "0"
    from yunshu_engine.batched_engine import BatchedEngine
    eng = BatchedEngine(MODEL); await eng.start()
    r = await eng.generate("Make a JSON person.", max_tokens=80, temperature=0.0,
                           json_schema=SCHEMA)
    await eng.stop()
    return r.text

async def main():
    jf = await _gen(True)
    print(f"[jump-forward] {jf!r}")
    obj = json.loads(jf)
    assert all(k in obj for k in ("name", "age", "city")), f"schema not satisfied: {obj}"
    assert isinstance(obj["age"], int)
    nf = await _gen(False)
    print(f"[normal]       {nf!r}")
    obj2 = json.loads(nf)
    assert all(k in obj2 for k in ("name", "age", "city"))
    print("\nW751 jump-forward engine-path smoke: PASS (both valid schema-conforming JSON)")

asyncio.run(main())
