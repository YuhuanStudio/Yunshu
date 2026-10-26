"""A complete stream alone cannot certify lossless APC restoration."""

import pytest
from scripts.research.apc_restore_identity import assert_identity


def results():
    return {
        mode: dict(cached=128, tokens_equal=True, logprobs_equal=True)
        for mode in ("partial", "full")
    }


@pytest.mark.parametrize("mode", ["partial", "full"])
@pytest.mark.parametrize("field", ["cached", "tokens_equal", "logprobs_equal"])
def test_identity_probe_rejects_miss_and_bit_drift(mode, field):
    out = results()
    out[mode][field] = False
    with pytest.raises(RuntimeError, match="not bit-equal"):
        assert_identity(out)


def test_identity_probe_accepts_both_restores():
    assert_identity(results())


def test_spec_off_reference_does_not_require_a_cache_hit():
    out = results()
    out["spec_off"] = dict(cached=0, tokens_equal=True, logprobs_equal=True)
    assert_identity(out)
    out["spec_off"]["logprobs_equal"] = False
    with pytest.raises(RuntimeError, match="spec_off restore is not bit-equal"):
        assert_identity(out)


def test_sampled_identity_request_preserves_explicit_sampling(monkeypatch):
    import json

    from scripts.research import apc_restore_identity as probe

    seen = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            row = {
                "choices": [
                    {
                        "logprobs": {"content": [{"token": "x", "logprob": -1.0}]},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 5},
            }
            return iter([b"data: " + json.dumps(row).encode(), b"data: [DONE]"])

    def open_request(req, **kw):
        seen.append(json.loads(req.data))
        return Response()

    monkeypatch.setattr(probe.urllib.request, "urlopen", open_request)
    sampling = dict(temperature=1.0, top_p=0.95, top_k=20, seed=31)
    result = probe.ask("http://localhost:18996", "model", "prompt", 1, sampling)
    assert all(seen[0][k] == v for k, v in sampling.items())
    assert result["toks"] == ["x"] and result["lps"] == [-1.0]
