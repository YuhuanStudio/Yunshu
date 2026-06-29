"""cancel-identity stamping + json-schema nullable-number termination
+ completions sampling-param parity.

(HIGH, rescopeded to single-consumer): the auth middleware now stamps the
  single-owner identity on current_actor for every request (the multi-tenant
  TenantManager/RBAC stack has been removed). This keeps request_tracker ownership
  and engine_core dedup working in the single-consumer model.
(HIGH): a top-level nullable/union scalar number (Pydantic Optional[int] →
  oneOf:[{number},{null}] → _top_level_type is a LIST) never reached a terminable state, so
  generation ran away to max_tokens. can_terminate now accepts an all-scalar list root.
completions dropped min_tokens/ignore_eos/suppress_tokens on the non-batched +
  streaming paths (chat parity gap).
"""

from __future__ import annotations

import inspect


def test_simplified_middleware_stamps_owner_identity():
    """Single-consumer model: the simplified auth middleware stamps the
    same owner identity on current_actor for every request (no per-tenant
    branch). This keeps request_tracker ownership / engine_core dedup working."""
    from yunshu_gateway.middleware import tenant_auth

    src = inspect.getsource(tenant_auth)
    # The middleware stamps current_actor with the single-owner identity.
    assert "current_actor.set(owner)" in src
    assert 'request.state.role = "owner"' in src


def test_resolve_actor_returns_single_owner():
    # Single-consumer model: resolve_actor always returns the constant owner
    # identity (per-tenant/RBAC actor resolution removed).
    from yunshu_control.audit_log import resolve_actor

    class _T:
        name = "acme"
        tenant_id = "tn-123"

    class _State:
        rbac_key = None
        tenant = _T()

    class _Req:
        state = _State()

    assert resolve_actor(_Req()) == "owner"


def test_nullable_number_terminates_but_nested_does_not():
    from yunshu_engine.json_schema import JsonSchemaConstraint

    def can_term(schema, s):
        c = JsonSchemaConstraint(schema)
        c._process_text(s)
        return c.can_terminate()

    # top-level scalar + nullable union → terminable after a complete number
    assert can_term({"type": "number"}, "42") is True
    assert can_term({"anyOf": [{"type": "number"}, {"type": "null"}]}, "7") is True
    assert can_term({"type": "integer"}, "13") is True
    # nested number (object/array) must NOT stop before the container closes
    assert (
        can_term(
            {
                "type": "object",
                "properties": {"a": {"type": "number"}},
                "required": ["a"],
            },
            '{"a": 5',
        )
        is False
    )
    # a mixed root union (object|number) must stay conservative in the object branch
    assert (
        can_term(
            {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {"a": {"type": "number"}},
                        "required": ["a"],
                    },
                    {"type": "number"},
                ]
            },
            '{"a": 5',
        )
        is False
    )
    # string root unaffected
    assert can_term({"type": "string"}, '"hi') is False


def test_completions_passes_sampling_params_all_paths():
    from yunshu_gateway.routers import completions

    src = inspect.getsource(completions)
    # min_tokens/ignore_eos/suppress_tokens forwarded on all 4 engine calls now
    assert src.count("min_tokens=req.min_tokens") >= 4
    assert src.count("ignore_eos=req.ignore_eos") >= 4
    assert src.count("suppress_tokens=req.suppress_tokens") >= 4
