# Phase B — design, 2026-10-25

Base: e3208d55 (sampledspec-1024); monitor and merge later sampledspec commits.
No GPU jobs submitted yet. No speed or identity verdict yet.

Reuse round_driver lifecycle / scheduler / APC, keyed_sampling accept, DFlash fast tree shapes / lattice / budgets, tree_verify transaction records and commit replay. Do not add a serving path or new setting. Keep existing paths until the gates justify removing them.

Round structure: reserve one root for every decoding request; distribute the remaining shared 32-node verify budget evenly among eligible drafting rows (rotate remainder to avoid permanent preference). A non-drafting row still verifies its root. Run DFlash proposals over rows with independent context rings, pack target projection inputs across all windows, run attention and GDN on each sequence's existing independent state, then pack output projections / MLP. Greedy or position-keyed Gumbel targets select each accepted path independently. Commit each transaction with tree_commit; replay only accepted GDN edges and compact only that row's KV. Stop/length/budget cuts trim the accepted path before commit.

First implementation boundary: generalize tree verify to packed sequence windows while delegating the one-window case to today's exact fast tree. This keeps the solo arithmetic and fusion unchanged and provides a direct 2/4-sequence x one-node test before drafter integration. Extract GDN projection input/output from its mixer with optional arguments; default call unchanged. Clear projection sum tables after every packed transaction. Aborts roll every touched row back.

Next: packed verifier parity tests; gpuq cheap pilot; batched ring drafter and driver integration; 1-row parity; N-row parity and timing (4K/32K, greedy/seeded sampled, >=3 interleaved quiet reps). Full units, lint/mypy, lead precheck, path matrix, yv long and serve-27b required before READY.
