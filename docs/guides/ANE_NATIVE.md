# Native ANE phase 1

`yunshu_engine.ane.native` is an optional research runtime. It has no serving
importer, user setting, automatic split, or claim of qualified lossless arithmetic.
The existing GPU paths remain the serving paths.

`NativeSession.try_create` builds a small Objective-C++ dylib using `clang++` on
first use, in an explicitly supplied cache directory. No Xcode project, Core ML
Python runtime, MLX import, or Metal work is needed to import the module. A caller
must pass MIL text, a weight blob, surface shapes and that cache directory.
Apple's private ANE framework is resolved dynamically. Each selector's argument
and return encoding is checked before any private message is sent. Missing or
changed interfaces, compiler/build errors and model failures return unavailable.
`run` returns `None` on evaluation failure; the caller must discard the output
and use the GPU. A failed session is permanently retired. The C ABI retains
in-flight state after a timeout so the service cannot access freed surfaces.

The adapted Splash runtime provides IOSurface-backed, channel-major fp16 memory
and Metal shared-event completion. Phase 1 copies inputs and outputs through
locked methods; native timings exclude those separately measured copies. No MLX
buffer is currently backed by these surfaces. End-to-end zero-copy integration
is a later phase, conditional on an actual measured gain.

The CPU MIL builder supports fp16 and symmetric per-channel int8 weights, with
`up * 2**h_exp` and `down * 2**-h_exp` to reduce intermediate underflow. This does
not establish bit identity with MLX bf16 arithmetic. Int8 introduces additional
weight quantization. Neither mode is approved for serving from a microbenchmark.

Reproduce the pilot with `scripts/research/anenative_pilot.sh` and the same-weight
comparison with `scripts/research/anenative_micro.sh` under the GPU queue. The
latter emits JSONL with a final `complete: true` only after all cells pass their
finite/reference checks. It records GPU FFN, native evaluate, host copies, Core ML
predict, first compile/load, and relative L2 error, with interleaved arms.
Core ML's `CPU_AND_NE` permits CPU fallback; native `_ANEClient` evaluation
establishes the device path. A paired timing receipt is required before drawing
split/concurrency conclusions. Unmeasured speedups are not product claims.

Source provenance: Splash 1.4.0, Apache-2.0, pinned in `vendor.json` and credited
in `THIRD_PARTY_NOTICES.md`.
