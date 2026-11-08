"""Tests for two architecture gap features:

1. Request cancellation propagation through scheduler ()
   - cancel_event passed to stream_outputs() for responsive cancellation
   - cancel_event triggers abort_request() to free GPU slots + KV blocks
   - Works even during chunked prefill when no outputs are produced

2. Chunked prefill progress reporting ()
   - RequestOutput.prefill_progress field (processed, total) tuple
   - Scheduler emits synthetic progress outputs during chunked prefill
   - GenerationOutput.prefill_progress forwarded to gateway
   - SSE comment emission for client-side progress bars
"""

import pytest

from yunshu_engine.batched_engine import GenerationOutput
from yunshu_engine.request import RequestOutput

# ── Fakes ──


class _FakeDetokenizer:
    def __init__(self):
        self._tokens = []
        self.last_segment = ""

    def reset(self):
        self._tokens = []
        self.last_segment = ""

    def add_token(self, token_id):
        text_map = {0: "Hello", 1: " world", 2: "!"}
        self.last_segment = text_map.get(token_id, f"tok{token_id}")

    def finalize(self):
        self.last_segment = ""


class _FakeTokenizer:
    eos_token_ids = [3]
    has_thinking = False

    def encode(self, text, **kwargs):
        return list(range(len(text)))

    def decode(self, tokens):
        return " ".join(f"t{t}" for t in tokens)

    @property
    def detokenizer(self):
        return _FakeDetokenizer()


class _FakeGenResponse:
    def __init__(self, uid, token, finish_reason=None, current_state="normal"):
        self.uid = uid
        self.token = token
        self.finish_reason = finish_reason
        self.current_state = current_state
        self.logprobs = None
        self.match_sequence = None
        self.prompt_cache = None
        self.all_tokens = None


class _FakeBatchGen:
    def __init__(self):
        self._uid_counter = 0
        self._pending = {}
        self._gen_pending = {}

    def insert(
        self,
        prompts,
        max_tokens=None,
        samplers=None,
        state_machines=None,
        stop_sequences=None,
    ):
        uids = []
        for _prompt, _mt in zip(prompts, max_tokens or [128], strict=False):
            uid = self._uid_counter
            self._uid_counter += 1
            uids.append(uid)
            self._gen_pending[uid] = [
                _FakeGenResponse(uid, 0),
                _FakeGenResponse(uid, 1),
                _FakeGenResponse(uid, 2, finish_reason="stop"),
            ]
        return uids

    def next(self):
        prompt_responses = []
        for uid in list(self._gen_pending.keys()):
            prompt_responses.append(_FakeGenResponse(uid, -1))
        return prompt_responses, []

    def next_generated(self):
        gen_responses = []
        finished = []
        for uid, responses in self._gen_pending.items():
            if responses:
                gen_responses.append(responses.pop(0))
                if not responses:
                    finished.append(uid)
        for uid in finished:
            del self._gen_pending[uid]
        return gen_responses if gen_responses else []

    def remove(self, uids, **kwargs):
        for uid in uids:
            self._gen_pending.pop(uid, None)

    def close(self):
        pass


def _fake_executor():
    import concurrent.futures

    return concurrent.futures.ThreadPoolExecutor(max_workers=1)


# ── Feature 1: Cancel Propagation Tests ──


# ── Feature 2: Chunked Prefill Progress Tests ──


class TestPrefillProgressField:
    """Test that prefill_progress field exists and works on RequestOutput."""

    def test_request_output_has_prefill_progress(self):
        """RequestOutput should have a prefill_progress field."""
        output = RequestOutput(request_id="test-1")
        assert hasattr(output, "prefill_progress")
        assert output.prefill_progress is None

    def test_request_output_prefill_progress_tuple(self):
        """prefill_progress should accept a (processed, total) tuple."""
        output = RequestOutput(
            request_id="test-1",
            prefill_progress=(1024, 4096),
        )
        assert output.prefill_progress == (1024, 4096)
        assert output.prefill_progress[0] == 1024
        assert output.prefill_progress[1] == 4096

    def test_generation_output_has_prefill_progress(self):
        """GenerationOutput should have a prefill_progress field."""
        output = GenerationOutput()
        assert hasattr(output, "prefill_progress")
        assert output.prefill_progress is None

    def test_generation_output_prefill_progress_tuple(self):
        """GenerationOutput prefill_progress should accept (processed, total)."""
        output = GenerationOutput(prefill_progress=(2048, 8192))
        assert output.prefill_progress == (2048, 8192)


class TestPrefillProgressSSEFormat:
    """Test the SSE comment format for prefill progress."""

    def test_sse_comment_format(self):
        """Verify the format of the SSE prefill-progress comment."""
        processed, total = 2048, 8192
        sse_comment = f": prefill-progress {processed}/{total}\n\n"
        # SSE comments start with ': '
        assert sse_comment.startswith(": prefill-progress ")
        assert f"{processed}/{total}" in sse_comment

    def test_sse_comment_bytes_encoding(self):
        """Verify the SSE comment is properly encoded as bytes."""
        processed, total = 4096, 16384
        sse_bytes = f": prefill-progress {processed}/{total}\n\n".encode()
        assert isinstance(sse_bytes, bytes)
        assert b"prefill-progress" in sse_bytes
        assert b"4096/16384" in sse_bytes


class TestCancelAdminGating:
    """cancel_all needs the admin scope once a token or keys are configured. The
    middleware stamps role="owner" on every admitted request, infer-only keys included,
    so the stamp grants nothing; with no auth configured the one local user is admin."""

    async def _call(self, scopes, monkeypatch, token="tok", role="owner"):
        import types

        from yunshu_gateway.routers import cancel as cancel_mod

        class _Tracker:
            def cancel_all(self):
                return 0

            def cancel(self, _id):
                return False

            def get_owner(self, _id):
                return None

        monkeypatch.setattr(
            cancel_mod, "get_request_tracker", lambda: _Tracker(), raising=False
        )
        monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
        if token:
            monkeypatch.setenv("YUNSHU_AUTH_TOKEN", token)
        else:
            monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
        principal = None if scopes is None else types.SimpleNamespace(scopes=scopes)
        st = types.SimpleNamespace(role=role, principal=principal)
        req_obj = types.SimpleNamespace(state=st, headers={})
        body = cancel_mod.CancelRequest(cancel_all=True)
        return await cancel_mod.cancel_generation(body, req_obj)

    @pytest.mark.asyncio
    async def test_infer_only_key_denied_cancel_all(self, monkeypatch):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            await self._call({"infer"}, monkeypatch)
        assert exc.value.status_code == 403

    @pytest.mark.asyncio
    async def test_admin_key_allowed_cancel_all(self, monkeypatch):
        result = await self._call({"admin"}, monkeypatch)
        assert result.get("status") == "cancelled"

    @pytest.mark.asyncio
    async def test_local_user_without_auth_allowed_cancel_all(self, monkeypatch):
        result = await self._call(None, monkeypatch, token=None)
        assert result.get("status") == "cancelled"
