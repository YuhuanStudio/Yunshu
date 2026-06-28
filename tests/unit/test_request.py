"""Tests for Request, RequestStatus, SamplingParams, and RequestOutput."""

from yunshu_engine.request import (
    Request,
    RequestOutput,
    RequestStatus,
    SamplingParams,
)


class TestRequestStatus:
    def test_is_finished(self):
        assert not RequestStatus.is_finished(RequestStatus.WAITING)
        assert not RequestStatus.is_finished(RequestStatus.RUNNING)
        assert not RequestStatus.is_finished(RequestStatus.PREEMPTED)
        assert RequestStatus.is_finished(RequestStatus.FINISHED_STOPPED)
        assert RequestStatus.is_finished(RequestStatus.FINISHED_LENGTH)
        assert RequestStatus.is_finished(RequestStatus.FINISHED_ABORTED)
        assert RequestStatus.is_finished(RequestStatus.FINISHED_ERROR)

    def test_finish_reason(self):
        assert RequestStatus.finish_reason(RequestStatus.FINISHED_STOPPED) == "stop"
        assert RequestStatus.finish_reason(RequestStatus.FINISHED_LENGTH) == "length"
        assert RequestStatus.finish_reason(RequestStatus.FINISHED_ABORTED) == "abort"
        assert RequestStatus.finish_reason(RequestStatus.FINISHED_ERROR) == "error"
        assert RequestStatus.finish_reason(RequestStatus.FINISHED_TIMEOUT) == "timeout"
        assert RequestStatus.finish_reason(RequestStatus.WAITING) is None

    def test_ordering(self):
        assert RequestStatus.WAITING < RequestStatus.RUNNING
        assert RequestStatus.RUNNING < RequestStatus.FINISHED_STOPPED
        assert RequestStatus.PREEMPTED < RequestStatus.FINISHED_STOPPED


class TestSamplingParams:
    def test_defaults(self):
        sp = SamplingParams()
        assert sp.max_tokens == 256
        assert sp.temperature == 0.7
        assert sp.top_p == 1.0
        assert sp.repetition_penalty == 1.0
        assert sp.priority == 0
        assert sp.logprobs is False

    def test_custom(self):
        sp = SamplingParams(max_tokens=1024, temperature=0.0, top_p=0.9)
        assert sp.max_tokens == 1024
        assert sp.temperature == 0.0
        assert sp.top_p == 0.9


class TestRequestOutput:
    def test_defaults(self):
        out = RequestOutput(request_id="test")
        assert out.new_text == ""
        assert out.finished is False
        assert out.finish_reason is None
        assert out.prompt_tokens == 0
        assert out.completion_tokens == 0

    def test_usage(self):
        out = RequestOutput(request_id="test", prompt_tokens=10, completion_tokens=20)
        assert out.usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    def test_backward_compat_aliases(self):
        out = RequestOutput(
            request_id="test",
            new_text="hello",
            new_token_ids=[1, 2, 3],
        )
        assert out.token_text == "hello"
        assert out.token_id == 3


class TestRequest:
    def test_create(self):
        req = Request(request_id="r1", prompt="Hello")
        assert req.request_id == "r1"
        assert req.status == RequestStatus.WAITING
        assert req.num_output_tokens == 0
        assert req.num_tokens == 0
        assert req.num_preemptions == 0

    def test_append_token(self):
        req = Request(request_id="r1", prompt="Hello")
        req.append_token(42)
        assert req.output_token_ids == [42]
        assert req.num_output_tokens == 1
        assert req.num_computed_tokens == 0  # tracks prefill only, not output
        req.append_token(43)
        assert req.num_output_tokens == 2

    def test_set_finished(self):
        req = Request(request_id="r1", prompt="Hello")
        req.set_finished(RequestStatus.FINISHED_STOPPED)
        assert req.status == RequestStatus.FINISHED_STOPPED
        assert req.finish_reason == "stop"
        assert req.generation_end > 0

    def test_is_finished(self):
        req = Request(request_id="r1", prompt="Hello")
        assert not req.is_finished()
        req.set_finished(RequestStatus.FINISHED_LENGTH)
        assert req.is_finished()

    def test_priority_ordering(self):
        req_low = Request(request_id="low", prompt="", priority=1)
        req_high = Request(request_id="high", prompt="", priority=10)
        assert req_low < req_high

    def test_hash_and_equality(self):
        r1 = Request(request_id="same", prompt="")
        r2 = Request(request_id="same", prompt="different")
        assert r1 == r2
        assert hash(r1) == hash(r2)

    def test_inequality(self):
        r1 = Request(request_id="a", prompt="")
        r2 = Request(request_id="b", prompt="")
        assert r1 != r2

    def test_max_tokens(self):
        req = Request(
            request_id="r1",
            prompt="",
            sampling_params=SamplingParams(max_tokens=512),
        )
        assert req.max_tokens == 512

    def test_num_tokens(self):
        req = Request(request_id="r1", prompt="")
        req.num_prompt_tokens = 10
        req.append_token(1)
        req.append_token(2)
        assert req.num_tokens == 12

    def test_prefill_duration(self):
        req = Request(request_id="r1", prompt="")
        req.prefill_start = 1.0
        req.prefill_end = 2.0
        assert req.prefill_duration == 1.0

    def test_generation_duration(self):
        req = Request(request_id="r1", prompt="")
        req.generation_start = 1.0
        req.generation_end = 3.0
        assert req.generation_duration == 2.0

    def test_duration_unset(self):
        req = Request(request_id="r1", prompt="")
        assert req.prefill_duration == 0.0
        assert req.generation_duration == 0.0
