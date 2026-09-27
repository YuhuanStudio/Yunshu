"""Exercise the real Qwen3.8 tokenizer/template, without loading weights.

Argument: a local metadata-only snapshot containing tokenizer.json,
tokenizer_config.json, chat_template.jinja, config.json and weight index.
"""

import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from yunshu_engine.batched_engine import _REASONING_EFFORT_MAP, BatchedEngine
from yunshu_engine.tool_call_parser import parse_tool_calls
from yunshu_engine.vlm_engine import VLMEngine

root = Path(sys.argv[1])
tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
messages = [{"role": "user", "content": "計算 12+17。"}]
kw = dict(tokenize=False, add_generation_prompt=True)
expected = {
    effort: tok.apply_chat_template(messages, reasoning_effort=effort, **kw)
    for effort in ["low", "medium", "xhigh"]
}
llm = object.__new__(BatchedEngine)
llm._tokenizer = tok
llm.enable_thinking = None
llm.model_name = "Qwen3.8-27B"
vlm = object.__new__(VLMEngine)
vlm._tokenizer = tok
vlm._model_path = "Qwen3.8-27B"
actual_llm = llm._apply_chat_template(messages)
actual_vlm = vlm._format_prompt(messages)
vlm._config = json.loads((root / "config.json").read_text())
history = [
    {"role": "user", "content": "Find the file"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "find_file", "arguments": {"name": "abc"}},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "found"},
    {"role": "user", "content": "Summarize"},
]
vlm_history = vlm._format_prompt(history)
canonical_history = [
    dict(m, content="" if m.get("content") is None else m["content"]) for m in history
]
canonical = tok.apply_chat_template(canonical_history, **kw)
built_vision_history = vlm._build_vlm_messages(history)

xml = "<tool_call><function=locate><parameter=x>12</parameter><parameter=ok>true</parameter></function></tool_call>"
parsed = parse_tool_calls(xml, "Qwen3.8-27B")
index = json.loads((root / "model.safetensors.index.json").read_text())
weights = index["weight_map"]
result = {
    "metadata_hashes": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.iterdir())
        if p.is_file()
    },
    "template_tokens": {
        e: len(tok.encode(s, add_special_tokens=False)) for e, s in expected.items()
    },
    "llm_matches": {e: actual_llm == s for e, s in expected.items()},
    "vlm_text_matches": {e: actual_vlm == s for e, s in expected.items()},
    "xhigh_budget_mapping": _REASONING_EFFORT_MAP.get("xhigh", 8192),
    "vlm_null_history": {
        "matches_canonical": vlm_history == canonical,
        "contains_literal_none": "None" in vlm_history,
    },
    "vision_history_fields": {
        "input": sorted(history[1]),
        "output": sorted(built_vision_history[1]),
        "output_content": built_vision_history[1]["content"],
    },
    "qwen_xml_parsed_arguments": [json.loads(p.arguments) for p in parsed],
    "vlm_estimated_tokens_per_image": vlm._estimate_image_tokens(),
    "weights": {
        "tensor_count": len(weights),
        "mtp_tensor_count": sum(".mtp." in k or k.startswith("mtp.") for k in weights),
        "vision_tensor_count": sum(
            "vision_tower" in k or k.startswith("visual.") for k in weights
        ),
        "total_bytes": index.get("metadata", {}).get("total_size"),
    },
}
assert result["llm_matches"]["xhigh"] and result["vlm_text_matches"]["xhigh"]
print(json.dumps(result, indent=2))
