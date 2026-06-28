"""(MED): a realtime response.create with conversation="none" (out-of-band) still
appended its output to the persistent conversation.

OpenAI Realtime out-of-band responses (conversation="none") deliver the generated output to
the client but must NOT enter the default conversation — otherwise the NEXT normal turn's
prompt (_build_messages iterates self.conversation.items) wrongly includes the side-query
(e.g. a guardrail / classification call). _generate_response added both the assistant message
and any function_call items unconditionally.

Fix: snapshot _oob = config.get("conversation") == "none" and gate the conversation.add_item
calls (and the conversation.item.created signal) on `not _oob`. The response output events
(output_item.added/done, response.done) still fire so the client receives the response.
"""
from __future__ import annotations

import inspect

from yunshu_gateway.routers import realtime


def test_generate_response_snapshots_out_of_band_flag():
    src = inspect.getsource(realtime.RealtimeSession._generate_response)
    assert '_oob = config.get("conversation") == "none"' in src


def test_both_add_item_calls_are_gated_on_not_oob():
    src = inspect.getsource(realtime.RealtimeSession._generate_response)
    # strip comments so the assertions match real code lines only
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # both history writes (function_call item + assistant message) must be present...
    assert code.count("self.conversation.add_item(") == 2, "expected exactly the 2 history writes"
    # ...and each must be guarded by a `not _oob` check (source-guard, count >= 2)
    assert code.count("if not _oob:") >= 2, "both add_item calls must be gated on `not _oob`"
    # the conversation.item.created signal must sit inside the not-_oob block (no separate emit)
    gate = src.index("if not _oob:\n                self.conversation.add_item(assistant_item)")
    assert "conversation.item.created" in src[gate:gate + 400]


def test_out_of_band_still_emits_response_output_events():
    # the fix must NOT suppress the client-facing output events — only the history writes.
    src = inspect.getsource(realtime.RealtimeSession._generate_response)
    assert "response.output_item.done" in src  # still emitted for the assistant item
    # the function_call output_item.added streaming loop is unconditional (uses _fc_items)
    assert "_fc_items.append(_fc_item)" in src
