# Roadmap and proposals

Yunshu prioritizes fast local LLM/VLM inference on Apple Silicon: decode speed,
cold and cached first-token latency, prefix reuse, and complete client behavior.
Qwen3.8-27B is the first tuned model. This page records direction, not dates or
claims that work has shipped. Released behavior belongs in CHANGELOG; measured
results belong in BENCHMARKS and PERF_TREND.

## Priorities

| Priority | Work | Completion evidence |
|---|---|---|
| Now | Release notes and upgrade clarity | Every release has sourced highlights, migration notes and install commands |
| Next | Apple Silicon portability evidence | Recorded M1–M5 / macOS checks distinguish passed, failed and untested combinations |
| Next | Model/client capability evidence | Checkpoint- and client-version-specific results; unsupported features give actionable errors |
| Next | Reproducible performance publication | Same-checkpoint cold/warm/turn-2 runs with correctness checks and raw receipt links |
| Later | Broader model tuning | Measured end-to-end gains without reducing output correctness |

Audio, speech, image/video generation and embeddings remain supported
capabilities. Shared-path regressions and explicitly prioritized work take
precedence. Distributed serving and multi-tenant infrastructure are outside scope.

## Lightweight RFC process

Routine fixes use the contribution/review process. For substantial changes to
public APIs, settings, cache formats, supported hardware or architecture, record
a proposal before implementation with:

- User problem and concrete proposed behavior.
- Scope, alternatives and reasons for the choice.
- Compatibility, migration, privacy and failure behavior.
- Correctness checks, measurement that decides success and rollback plan.
- Owner, status (proposed / accepted / implemented / superseded) and decision rationale.

Use an existing discussion or a reviewable Markdown proposal; no new issue is
required for a small fix. Maintainers record the decision where the proposal is
reviewed. Accepted is not shipped: close it with implementation and validation
links. Private vulnerability details follow SECURITY.md instead.

## Decisions and maintenance

Contributors own the quality and explanation of their changes. Maintainers review
scope, compatibility and evidence; repository administrators control merges,
release publication and access. Reviewers should identify unresolved objections
and record the chosen tradeoff. Governance changes need an explicit maintainer
review; this document does not grant repository permissions or create new roles.
