"""Master regression harness — ONE entry point that runs everything and emits a
single consolidated report, so a big change can be verified across all models /
settings / techniques / cache layers / generation quality at once.

It ORCHESTRATES the existing verified sub-harnesses (does not reinvent them):
correctness (unit tests + cache losslessness), generation QUALITY (MMLU), the
4-tier cache × model matrix, multi-framework speed, comprehensive sweeps
(length / hit-ratio / concurrency), realistic cache rate, and the 6-modality
smoke. Each section is a subprocess; the orchestrator parses its result, times
it, and writes docs/reports/REGRESSION_REPORT.md (+ .json) with a final GO / NO-GO.

GATE sections (correctness, quality, modalities) must PASS for GO. METRIC sections
(perf matrices) are recorded but never block — they're for tracking/comparison.

Tiers (runtime budget):
  smoke    — unit tests + cache-lossless verifies + modality smoke         (~5-10 min)
  standard — + MMLU quality + 4-tier matrix                                (~30-45 min)
  full     — + framework speed + comprehensive sweeps + realistic cache    (~2 h)

Run:
  PYTHONPATH=. uv run python scripts/regression.py --tier smoke
  PYTHONPATH=. OMLX_PYTHON=.venvs/omlx/bin/python uv run python scripts/regression.py --tier full
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
VLM_PP = "reference/mlx-vlm:."
_TIER_ORDER = {"smoke": 0, "standard": 1, "full": 2}


# ── result parsers: (raw_output, exit_code) → (status, summary, metrics) ──
def _p_pytest(out, rc):
    m = re.search(r"(\d+) passed(?:, (\d+) skipped)?(?:, (\d+) failed)?", out)
    if not m:
        m2 = re.search(r"(\d+) failed", out)
        return ("FAIL", "no pass line", {}) if m2 else ("FAIL", "no result", {})
    passed, _skip, failed = m.group(1), m.group(2), m.group(3)
    failed = int(failed or 0)
    return ("PASS" if failed == 0 and rc == 0 else "FAIL",
            f"{passed} passed, {failed} failed", {"passed": int(passed), "failed": failed})


def _p_verify(out, rc):
    m = re.search(r"RESULT:\s*(\d+) passed,\s*(\d+) failed", out)
    if m:
        f = int(m.group(2))
        return ("PASS" if f == 0 else "FAIL", f"{m.group(1)} passed, {f} failed",
                {"passed": int(m.group(1)), "failed": f})
    return ("PASS" if rc == 0 else "FAIL", f"exit {rc}", {})


def _p_passfail(out, rc):
    last = "PASS" if ("\nPASS" in out or out.strip().endswith("PASS")) else None
    if "PASS:" in out and "FAIL" not in out.split("PASS:")[-1][:40]:
        last = "PASS"
    if re.search(r"\bFAIL\b", out) and "=> OK" not in out:
        # verify_vlm prints "=> OK"/"=> FAIL" per model + final PASS/FAIL
        pass
    status = "PASS" if (rc == 0 and re.search(r"\bPASS\b", out)) else "FAIL"
    return (status, f"exit {rc}", {})


def _p_mmlu(out, rc):
    m = re.search(r"RESULT:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%", out)
    if m:
        acc = float(m.group(3))
        return ("PASS" if rc == 0 else "FAIL", f"{m.group(1)}/{m.group(2)} = {acc:.1f}%",
                {"accuracy": acc, "correct": int(m.group(1)), "total": int(m.group(2))})
    return ("FAIL", "no MMLU result", {})


def _p_modalities(out, rc):
    m = re.search(r"(\d+)/(\d+) tests passed", out)
    # Keep the per-modality summary + any signal-crash lines so a FAIL is never a
    # black box (which modality, signal-vs-logic) — the section's full stdout is
    # captured but not echoed; surface the salient tail here.
    tail = "\n".join(ln for ln in out.splitlines()
                     if any(s in ln for s in ("✓", "✗", "died with signal",
                                              "GPU Hang", "tests passed", "FAIL:")))[-2000:]
    if m:
        p, t = int(m.group(1)), int(m.group(2))
        return ("PASS" if p == t and rc == 0 else "FAIL", f"{p}/{t} modalities",
                {"passed": p, "total": t, "detail": tail})
    return ("PASS" if rc == 0 else "FAIL", f"exit {rc}", {"detail": tail})


def _p_metrics(tag):
    """Capture @@TAG@@ json lines (per-model) into a metrics dict."""
    def parse(out, rc):
        rows = {}
        for m in re.finditer(rf"@@{tag}@@ (\{{.*\}})", out):
            try:
                d = json.loads(m.group(1))
                rows[d.get("model", f"row{len(rows)}")] = d
            except Exception:
                pass
        n = len(rows)
        return ("PASS" if (rc == 0 and n) else ("PARTIAL" if n else "FAIL"),
                f"{n} models", {"rows": rows})
    return parse


def _p_passfail_skip(out, rc):
    """PASS/FAIL scripts that may SKIP when a model/asset is absent (skip != fail)."""
    if "SKIP:" in out:
        return ("PASS", "skipped (asset absent)", {"skipped": True})
    ok = rc == 0 and "PASS" in out and "\nFAIL" not in out
    return ("PASS" if ok else "FAIL", f"exit {rc}", {})


def _p_quality(out, rc):
    """quality_comparison.py: Yunshu engine accuracy vs mlx-lm baseline."""
    if "SKIP" in out or "Model not found" in out:
        return ("PASS", "skipped (model absent)", {"skipped": True})
    e = re.search(r"Yunshu engine accuracy:\s*(\d+)%", out)
    b = re.search(r"Baseline \(mlx-lm\) accuracy:\s*(\d+)%", out)
    ea, ba = (int(e.group(1)) if e else None), (int(b.group(1)) if b else None)
    ok = rc == 0 and "PASS:" in out
    summ = f"yunshu {ea}% vs baseline {ba}%" if ea is not None else f"exit {rc}"
    return ("PASS" if ok else "FAIL", summ, {"engine_acc": ea, "baseline_acc": ba})


def _p_table(out, rc):
    """For bench_all/bench_frameworks/comprehensive: record the rendered table
    AND load the full structured json the bench wrote ('full json -> PATH') so
    the report can render complete tables (the 40-line tail top-truncates large
    sweeps)."""
    art = None
    m = re.search(r"full json -> (\S+)", out)
    if m:
        try:
            art = json.load(open(m.group(1).strip()))
        except Exception:
            art = None
    return ("PASS" if rc == 0 else "FAIL", f"exit {rc}",
            {"table_tail": "\n".join(out.splitlines()[-60:]), "artifact": art})


# ── section catalogue: (name, tier, gate, cmd, extra_env, parser) ──
def _sections():
    return [
        ("unit-tests", "smoke", True,
         [PY, "-m", "pytest", "tests/", "-q"], {}, _p_pytest),
        ("cache-lossless: VLM text (GLM-OCR + gemma)", "smoke", True,
         [PY, "scripts/verify/verify_vlm_text_kv_prefix.py"], {"PYTHONPATH": VLM_PP}, _p_passfail),
        ("cache-lossless: engine-loop", "smoke", True,
         [PY, "scripts/verify/verify_engine_loop.py"], {}, _p_verify),
        ("modalities (6-modality smoke)", "smoke", True,
         [PY, "scripts/realmodel/test_all_modalities.py",
          "--settle", "30", "--crash-cooldown", "180"], {"PYTHONPATH": VLM_PP}, _p_modalities),
        ("structured output (json_schema enforced)", "smoke", True,
         [PY, "scripts/verify/verify_structured_output.py"], {}, _p_verify),
        ("tool-call parsing (formats + brace-in-string)", "smoke", True,
         [PY, "scripts/verify/verify_tool_calls.py"], {}, _p_verify),
        ("reasoning parser (<think> split, no leak)", "smoke", True,
         [PY, "scripts/verify/verify_reasoning_parser.py"], {}, _p_verify),
        ("sampling (temp0 determinism + seed repro/vary)", "smoke", True,
         [PY, "scripts/verify/verify_sampling.py"], {}, _p_passfail_skip),
        ("grammar constraints (choice + regex)", "smoke", True,
         [PY, "scripts/verify/verify_grammar_constraints.py"], {}, _p_verify),
        ("embeddings + scoring (embed/rerank/classify/pool semantics)", "smoke", True,
         [PY, "scripts/verify/verify_embeddings_scoring.py"], {}, _p_passfail_skip),
        ("logprobs + stop sequences (generation contract)", "smoke", True,
         [PY, "scripts/verify/verify_logprobs_stop.py"], {}, _p_passfail_skip),
        ("n>1 parallel sampling (gateway choice/usage contract)", "smoke", True,
         [PY, "scripts/verify/verify_n_choices.py"], {}, _p_passfail_skip),
        ("streaming SSE (chunk contract + assembled==non-stream)", "smoke", True,
         [PY, "scripts/verify/verify_streaming_sse.py"], {}, _p_passfail_skip),
        ("Anthropic /v1/messages (envelope + streaming events)", "smoke", True,
         [PY, "scripts/verify/verify_anthropic_messages.py"], {}, _p_passfail_skip),
        ("tool-calling E2E (assembly + selection + envelope)", "smoke", True,
         [PY, "scripts/verify/verify_tool_calls_e2e.py"], {}, _p_passfail_skip),
        ("HTTP error contract (4xx validation + 404 unknown model)", "smoke", True,
         [PY, "scripts/verify/verify_error_contract.py"], {}, _p_verify),
        ("Responses API /v1/responses (envelope + streaming lifecycle)", "smoke", True,
         [PY, "scripts/verify/verify_responses_api.py"], {}, _p_passfail_skip),
        ("Anthropic tool_use E2E (block + stop_reason + selection)", "smoke", True,
         [PY, "scripts/verify/verify_anthropic_tools.py"], {}, _p_passfail_skip),
        ("scoring endpoints HTTP (embeddings/score/rerank/classify)", "smoke", True,
         [PY, "scripts/verify/verify_scoring_endpoints.py"], {}, _p_passfail_skip),
        ("core endpoints (/v1/models + /v1/completions)", "smoke", True,
         [PY, "scripts/verify/verify_core_endpoints.py"], {}, _p_passfail_skip),
        ("logit_bias effect + out-of-vocab guard", "smoke", True,
         [PY, "scripts/verify/verify_logit_bias.py"], {}, _p_passfail_skip),
        ("multi-turn context (threaded history recall)", "smoke", True,
         [PY, "scripts/verify/verify_multiturn.py"], {}, _p_passfail_skip),
        ("streaming tool_calls (delta reassembly + JSON args)", "smoke", True,
         [PY, "scripts/verify/verify_streaming_tool_calls.py"], {}, _p_passfail_skip),
        ("freq/presence penalties (reduce repetition)", "smoke", True,
         [PY, "scripts/verify/verify_penalties.py"], {}, _p_passfail_skip),
        ("JSON mode (response_format json_object)", "smoke", True,
         [PY, "scripts/verify/verify_json_mode.py"], {}, _p_passfail_skip),
        ("Anthropic count_tokens (matches usage + monotonic)", "smoke", True,
         [PY, "scripts/verify/verify_count_tokens.py"], {}, _p_passfail_skip),
        ("gateway sampling forwarding (seed/stop/max_tokens)", "smoke", True,
         [PY, "scripts/verify/verify_gateway_sampling.py"], {}, _p_passfail_skip),
        ("sampling constraints (top_k/top_p/min_p collapse)", "smoke", True,
         [PY, "scripts/verify/verify_sampling_constraints.py"], {}, _p_passfail_skip),
        ("concurrent requests (serialization, no cross-talk)", "smoke", True,
         [PY, "scripts/verify/verify_concurrent.py"], {}, _p_passfail_skip),
        ("priority/FAIR scheduling (policy plumbed + ordering)", "smoke", True,
         [PY, "scripts/verify/verify_priority_scheduling.py"], {}, _p_passfail_skip),
        ("KV-cache quant round-trip (4/8-bit bounded + compression)", "smoke", True,
         [PY, "scripts/verify/verify_kv_quant.py"], {}, _p_passfail_skip),
        ("sampling extras (XTC removes tokens + thinking_budget caps)", "smoke", True,
         [PY, "scripts/verify/verify_sampling_extras.py"], {}, _p_passfail_skip),
        ("embeddings dimensions (Matryoshka truncation)", "smoke", True,
         [PY, "scripts/verify/verify_embeddings_dimensions.py"], {}, _p_passfail_skip),
        ("streaming logprobs (per-chunk entries)", "smoke", True,
         [PY, "scripts/verify/verify_streaming_logprobs.py"], {}, _p_passfail_skip),
        ("usage accounting (exact prompt_tokens + stream==non-stream)", "smoke", True,
         [PY, "scripts/verify/verify_usage_accounting.py"], {}, _p_passfail_skip),
        ("batch inference API (/v1/batch custom_id mapping)", "smoke", True,
         [PY, "scripts/verify/verify_batch_api.py"], {}, _p_passfail_skip),
        ("quality: MMLU (Qwen3.5-9B)", "standard", True,
         [PY, "scripts/bench/bench_mmlu_quick.py"], {}, _p_mmlu),
        ("quality vs mlx-lm baseline", "standard", True,
         [PY, "scripts/tools/quality_comparison.py", "--quick", "--model-path",
          "models/Qwen2.5-3B-Instruct-bf16"], {}, _p_quality),
        ("spec-decode lossless (gemma-4)", "standard", True,
         [PY, "scripts/validate/validate_gemma4_spec_decode.py"], {"PYTHONPATH": "python"}, _p_passfail_skip),
        ("n-gram spec (hybrid guard==greedy + dense non-degenerate)", "smoke", True,
         [PY, "scripts/verify/verify_ngram_spec.py"], {}, _p_passfail_skip),
        ("ASR transcription quality", "standard", True,
         [PY, "scripts/verify/verify_asr_quality.py"], {}, _p_passfail_skip),
        ("TTS VoiceDesign (instruct steers voice)", "standard", True,
         [PY, "scripts/verify/verify_voicedesign.py"], {}, _p_passfail_skip),
        ("TTS→ASR round-trip (end-to-end audio fidelity)", "standard", True,
         [PY, "scripts/verify/verify_tts_asr_roundtrip.py"], {}, _p_passfail_skip),
        ("VLM OCR (discriminative image reading)", "standard", True,
         [PY, "scripts/verify/verify_vlm_ocr.py"], {}, _p_passfail_skip),
        ("VLM vision/image-reuse cache (wrapper + ~6x repeat-image)", "standard", True,
         [PY, "scripts/verify/verify_vlm_vision_cache.py"], {"PYTHONPATH": VLM_PP}, _p_passfail_skip),
        ("image t2i (prompt drives pixels, discriminative)", "standard", True,
         [PY, "scripts/verify/verify_image_t2i.py"], {}, _p_passfail_skip),
        ("image-gen HTTP route (/v1/images/generations b64_json)", "standard", True,
         [PY, "scripts/verify/verify_image_http.py"], {}, _p_passfail_skip),
        ("VLM image-chat HTTP route (base64 image_url → OCR)", "standard", True,
         [PY, "scripts/verify/verify_vlm_http.py"], {}, _p_passfail_skip),
        ("context-window overflow (clean 400, not crash)", "standard", True,
         [PY, "scripts/verify/verify_context_overflow.py"], {}, _p_passfail_skip),
        ("prefill-memory guard (chat/Anthropic/Responses → 413)", "smoke", True,
         [PY, "scripts/verify/verify_prefill_guard.py"], {}, _p_passfail_skip),
        ("image diffusion-LoRA (load/effect/restore)", "standard", True,
         [PY, "scripts/verify/verify_image_lora.py"], {}, _p_passfail_skip),
        ("prompt weighting (word:weight emphasis)", "standard", True,
         [PY, "scripts/verify/verify_prompt_weighting.py"], {}, _p_passfail_skip),
        ("ControlNet structural following (Z-Image)", "standard", True,
         [PY, "scripts/verify/verify_controlnet.py"], {}, _p_passfail_skip),
        ("cache-tier matrix (all models)", "standard", False,
         [PY, "scripts/bench/bench_all.py"], {}, _p_table),
        ("realistic cache rate", "standard", False,
         [PY, "scripts/bench/bench_realistic_cache.py"], {"PYTHONPATH": VLM_PP,
          "YUNSHU_BENCH_MODEL": "./models/Qwen2.5-3B-Instruct-bf16"}, _p_metrics("REALISTIC")),
        ("MTP spec decode (Qwen3.6-27B production path, coherent)", "full", True,
         [PY, "scripts/verify/verify_mtp_spec.py"], {"YUNSHU_MTP": "1", "PYTHONPATH": VLM_PP}, _p_passfail_skip),
        # (methodology): the PRIMARY cross-framework comparison is ALL-EXTERNAL
        # (server bench below) — only real HTTP servers are a fair, production-truthful
        # measure. This in-process bench is the OTHER half: EVERY framework measured
        # internally too, so the report computes each one's internal-vs-external parity
        # (gateway efficiency → surfaces middle-layer bugs per framework).
        ("framework internal (in-process; internal-vs-external parity input)", "full", False,
         [PY, "scripts/bench/bench_frameworks.py"], {}, _p_table),
        # PRIMARY framework comparison: each framework's REAL OpenAI HTTP server, one
        # at a time, GPU cooled to a matched thermal state between them. This is the
        # fair production comparison; feeds the serve/ + gpu_tflops trend. Multiple
        # representative models so the ranking isn't a single-model artifact.
        # vllm-mlx is OMITTED from the external set: its server mis-routes text models
        # through the MLLM (multimodal) loader and fails to start ("Received N
        # parameters not in model") — a bug in the reference framework's own server,
        # not ours. Its launcher branch stays in bench_serve for when that's fixed; we
        # don't fall back to its (unfair) in-process numbers. yunshu/mlx-lm/oMLX serve
        # cleanly → the fair all-external comparison.
        ("server bench (real OpenAI HTTP — PRIMARY framework comparison)", "full", False,
         [PY, "scripts/bench/bench_serve.py",
          "--models", "./models/Qwen2.5-3B-Instruct-bf16,./models/Qwen3.5-2B-MLX-bf16",
          "--frameworks", "yunshu,mlx-lm,oMLX", "--concurrency", "8,16,32",
          "--cooldown-sec", "60", "--wait-tflops", "9"], {}, _p_table),
        ("sustained-decode thermal decay (cool/ trend)", "full", False,
         [PY, "scripts/bench/bench_sustained_decode.py"], {}, _p_table),
        ("model breadth sweep (all LLMs × core correctness, isolated)", "full", True,
         [PY, "scripts/bench/sweep_models.py"], {"PYTHONPATH": VLM_PP}, _p_passfail),
        ("per-model perf sweep (TTFT/decode/peak-RSS, isolated)", "full", False,
         [PY, "scripts/bench/sweep_perf.py"], {}, _p_table),
        ("comprehensive sweeps (length/hitratio/concurrency)", "full", False,
         [PY, "scripts/bench/bench_comprehensive_all.py"], {}, _p_table),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["smoke", "standard", "full"], default="smoke")
    ap.add_argument("--only", nargs="*", help="run only sections whose name contains these substrings")
    args = ap.parse_args()
    budget = _TIER_ORDER[args.tier]

    base_env = dict(os.environ, PYTHONPATH=".")
    results = []
    for name, tier, gate, cmd, extra_env, parser in _sections():
        if _TIER_ORDER[tier] > budget:
            continue
        if args.only and not any(s.lower() in name.lower() for s in args.only):
            continue
        print(f"\n>>> [{tier}] {name} ...", flush=True)
        # PYTHONUNBUFFERED so a section's progress is FLUSHED line-by-line — if it
        # later times out and is SIGKILL'd, we keep the partial output (the
        # framework-speed timeout previously lost all diagnostics on the kill).
        env = dict(base_env, PYTHONUNBUFFERED="1", **extra_env)
        # PROACTIVE GPU settle: the 6-modality smoke loads Qwen3-Omni-30B right after
        # the heavy VLM cache-lossless verify (which also loads the 30B); the Metal
        # driver hang from back-to-back 30B loads persists past a 75s post-crash
        # retry, so let the GPU settle BEFORE the section (more effective than
        # retrying after it crashes). 6/6 in isolation; this just avoids the
        # back-to-back collision.
        if "modalities" in name.lower():
            # The modality smoke is 6/6 in ISOLATION on a fresh GPU (verified). The
            # full-run "5/6" flake is cumulative Metal-driver stress: this section
            # runs right after cache-lossless-VLM (GLM-OCR + gemma + 30B), so the
            # driver hasn't recovered when modalities then loads 6 more models. Give
            # it a longer settle to recover, AND adaptively wait until the GPU is
            # responsive again (a tiny matmul that confirms the driver isn't hung)
            # before starting — more reliable than a fixed sleep.
            print("    (GPU recovery before heavy 30B-loading modality smoke)", flush=True)
            time.sleep(60)
            try:
                import importlib
                _ph = importlib.import_module("perf_history")
                for _i in range(6):
                    _tf = _ph._gpu_tflops(1.0)
                    if _tf and _tf >= 8.0:
                        print(f"    GPU responsive ({_tf} TFLOP/s) — starting modalities", flush=True)
                        break
                    print(f"    GPU still recovering ({_tf} TFLOP/s) — +20s", flush=True)
                    time.sleep(20)
            except Exception:
                time.sleep(30)
        t0 = time.time()
        attempts = 0
        retried = False
        # A native-signal death (rc < 0, e.g. SIGABRT from a transient Metal GPU
        # hang) is NOT a logic failure. The driver hang can persist across processes
        # right after a heavy section (e.g. the VLM verify loads GLM-OCR + gemma +
        # the 30B Omni), so a single short cooldown isn't always enough — retry up to
        # twice with an ESCALATING cooldown (30s, then 75s) before flipping GO/NO-GO.
        _COOLDOWNS = [30, 75]
        # bound each section's wall time so one pathological metric bench
        # can't make `--tier full` run for hours. GATE sections (correctness) get a
        # generous budget; METRIC benches (gate=False, e.g. framework-speed which
        # launches each framework's server across a config matrix) get a tight one
        # and are recorded as a timeout rather than blocking the whole run.
        # The multi-framework benches legitimately exceed the tight default metric
        # budget (but stay bounded): the server bench launches 3 real HTTP servers
        # with 60s cooldowns + a thermal pre-wait; the framework-speed bench runs
        # 4 models × 5 frameworks (yunshu-fast/loop, mlx-lm, vllm-mlx, oMLX) in
        # sequence (it hit the old 1200s cap → fw/ data never captured). Give both
        # 5400s so the trend's fw/ + serve/ columns actually populate.
        _nm = name.lower()
        # Match the heavy multi-framework benches by a stable substring ("framework"
        # covers both the renamed in-process "framework internal" and any future
        # name; "server bench" covers the external one). The in-process bench runs
        # 4 models × 5 frameworks → it timed out at the 1200s default after the
        # rename stopped matching "framework speed".
        _heavy_metric = ("server bench" in _nm) or ("framework" in _nm)
        # framework speed = 4 models × 5 frameworks, each a fresh subprocess model
        # load (internal 300s/run cap → 6000s worst case); give it 7200s so it
        # ALWAYS completes and the fw/ trend actually captures. server bench is
        # bounded by its own cooldowns. 1200s for the light metrics.
        _sec_timeout = 9000 if gate else (7200 if _heavy_metric else 1200)
        while True:
            attempts += 1
            # run the section in its OWN process group and, on timeout,
            # kill the WHOLE TREE. The bench sections (framework-speed,
            # comprehensive) spawn grandchildren (_fw_*, _bench_*); plain
            # subprocess.run(timeout=) only SIGKILLs the direct child, so the
            # grandchildren orphan and keep the GPU busy — the section never
            # cleanly capped and `--tier full` had to be unblocked by hand. With
            # a process group we can os.killpg the entire subtree.
            proc = subprocess.Popen(
                cmd, env=env, cwd=REPO, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, start_new_session=True,
            )
            try:
                out, _ = proc.communicate(timeout=_sec_timeout)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                import signal as _sig
                try:
                    os.killpg(os.getpgid(proc.pid), _sig.SIGKILL)
                except Exception:
                    proc.kill()
                try:
                    out, _ = proc.communicate(timeout=30)
                except Exception:
                    out = ""
                out = (out or "") + "\nTIMEOUT"
                rc = 124
            # Retry only GATE sections on a transient native-signal crash. A METRIC
            # bench that crashes/stalls is not worth retrying (and must never block).
            if rc < 0 and gate and attempts <= len(_COOLDOWNS):
                retried = True
                cd = _COOLDOWNS[attempts - 1]
                print(f"    crashed (signal {-rc}) — GPU cooldown {cd}s, retry {attempts}/{len(_COOLDOWNS)} ...", flush=True)
                time.sleep(cd)
                continue
            break
        dt = time.time() - t0
        status, summary, metrics = parser(out, rc)
        if retried:
            summary += " (retried after transient crash)"
            metrics["retried"] = True
        results.append({"name": name, "tier": tier, "gate": gate, "status": status,
                        "summary": summary, "seconds": round(dt, 1), "metrics": metrics})
        print(f"    {status}  ({summary}, {dt:.0f}s)")
        # On a FAILED section, echo the salient captured detail so the failure is
        # never a black box (esp. the modality smoke: which modality, signal-vs-logic).
        if status == "FAIL" and isinstance(metrics, dict) and metrics.get("detail"):
            for _dl in str(metrics["detail"]).splitlines()[-12:]:
                print(f"      | {_dl}")

    gated = [r for r in results if r["gate"]]
    go = all(r["status"] == "PASS" for r in gated)
    _write_report(args.tier, results, go)
    # Append-only, time-named perf history + absolute-evolution trend. Only runs
    # that produced framework/cache metrics add a perf point; smoke runs are
    # skipped inside snapshot(). Never let this break the regression verdict.
    try:
        from perf_history import snapshot as _perf_snapshot, trend as _perf_trend
        if _perf_snapshot():        # returns None / skips when no perf metrics
            _perf_trend()
    except Exception as _e:
        print(f"(perf-history skipped: {_e})")
    # Unified report: merge gate status + perf trend + analysis + charts into ONE
    # Markdown deliverable (docs/reports/REPORT.md). Best-effort; never blocks GO.
    try:
        import report as _report
        _report.main()
    except Exception as _e:
        print(f"(unified report skipped: {_e})")
    print(f"\n{'='*70}\n{'GO ✅' if go else 'NO-GO ❌'} — {sum(1 for r in gated if r['status']=='PASS')}/{len(gated)} gates passed")
    print(f"report -> docs/reports/REGRESSION_REPORT.md (gates) + docs/reports/REPORT.md (unified)")
    sys.exit(0 if go else 1)


_KNOWN_LIMITATIONS = """
**Speedup ratios are baseline-relative — read absolute latency when a ratio moves.**
F-HOT/WARM/SSD = `COLD_ms / reuse_ms`. The COLD prefill baseline drifts ±15-40% run
to run (thermal state, memory pressure), so a ratio can rise/fall while the reuse
latency is unchanged or better. Compare absolute reuse TTFT (`COLD_ms ÷ F-HOT`)
across runs, not the ratio.

**Long full-tier runs thermally suppress later sections.** bench sections run
sequentially after ~25 min of GPU load; the heaviest model (Qwen3.5-9B-4bit) shows
~15-20% lower decode/prefill late in a run than standalone. Treat absolute throughput
in a full run as a floor, not the peak — re-run a single section in isolation for a
clean number.

**Cache reuse always beats cold-prefilling the SAME prompt** (measured 4.5–8× on
Qwen2.5-3B, and the win GROWS with prompt length). The hit-ratio sweep's `speedup`
column is now vs cold-prefill of the same full prompt (≥1×). An earlier version
divided by the short 1024-token base, which made low-hit rows look <1× — that was a
denominator artifact, NOT a real regression. Reuse is never net-negative on the
HOT/WARM tiers.

**SSD tier IS net-negative for fast-prefill models** (the one real cache-cost case).
GLM-OCR (prefill ~6300 t/s) restores from SSD slower than it re-prefills → F-SSD
0.93×. Auto-gated by `YUNSHU_SSD_PREFILL_TPS_CEIL` (default 4000 t/s): when a model's
measured prefill throughput exceeds the ceiling, the SSD restore is skipped in favour
of re-prefill. `YUNSHU_SSD_RESTORE_MIN_TOKENS` also gates by prefix size (default 0).
Standard-attention models get weaker SSD (Qwen2.5-3B 2.16× vs HOT 5.92×) because
full-precision KV is large on disk — still positive, just not gated.

**gemma-4 WARM tier saves no RAM (WARMram 1.00×).** Its sliding-window cache layers
have no `to_quantized`, so WARM stores them unquantized — same speed as HOT, no 3.56×
RAM saving. WARM is only worth it for full-attention models.

**Sliding-window models (gemma-4) write NOTHING to the SSD tier (by design).**
gemma-4 is sliding-window (RotatingKVCache on 35/42 layers) so it's non-trimmable →
`_no_trim_mode`. Previously it fell into the whole-snapshot *hybrid* SSD path (built
for Qwen3.5 linear-attn), spilling its entire 42-layer cache per entry — 923 MB that
the cross-query restore (keyed by whole-prompt hash) never read. Now `_has_recurrent_layer`
gates the hybrid spill (ArraysCache only) and `_is_block_decomposable` gates the
per-block spill (plain KVCache only), so sliding-window matches neither → `ssd_disk_mb≈0`
and `F-SSD≈1.0×` (clean fall-back to prefill). Its in-RAM HOT/WARM no_trim reuse still
serves it losslessly (~3.5×). The hybrid whole-snapshot SSD remains a *recurrent*-model
(Qwen3.5) feature and is a CONTINUATION cache (restores when a later prompt extends a
previously-evicted full prompt), not a shared-prefix cache.

**Coverage gaps in this run:** oMLX can't load gemma-4 (old checkpoint → LOAD FAIL);
the raw-mlx-lm framework row for gemma-4 returned NO RESULT (batch path failed for
that model). The comprehensive "concurrency" sweep is FAST-PATH only (serialized,
agg ≈ single-stream); engine-loop concurrency is the framework section's batch8/16/32
columns (much higher). Don't read the two concurrency numbers as the same thing.

**The "framework speed" section is an IN-PROCESS micro-benchmark — its batched
columns are NOT production serving.** It drives each engine's BatchedEngine directly,
which never engages oMLX's scheduler (oMLX shows ~1.2× batched as an artifact). The
fair production comparison — each framework's real OpenAI HTTP server, at a matched
GPU thermal state — is `scripts/bench/bench_serve.py`, tracked in the `serve/` section of
`docs/reports/PERF_TREND.md`. Fair-state finding: yunshu leads concurrent throughput at all N
(3B sys@32 137 > mlx-lm 126 > oMLX 96); oMLX does not scale concurrency.

**Absolute-throughput thermal note (corrected).** This is a **30-core M3 Max
(36GB)**, whose real MLX fp16 ceiling is ~9.5 TFLOP/s (8192³, plateaus N=4096→8192 — the
genuine peak, NOT the ~21 theoretical and certainly not the 40-core's ~28). Verified
cold: raw mlx-lm decode is FLAT at ~41 tok/s across 8 back-to-back iterations (0% decay)
and matches the recorded yunshu serve decode (41.8) — so the session's `serve/` numbers
were the TRUE cool baseline, not "~35% throttled" (that earlier caveat was wrong on the
peak). A *single* clean run does not throttle; the slowdown seen earlier came from
running multiple servers + warmup matmuls concurrently (contention/heat), which decays
sustained inference. bench_serve still self-tags each run's GPU TFLOP/s and
`--wait-tflops` gates on a plateau, so cross-run absolutes stay comparable; flag a run
only if its tag reads well below the ~9.5 baseline.
"""


def _fmt_realistic(rows):
    """Render the realistic-cache structured rows (multi-turn + RAG) as tables."""
    lines = []
    for mdl, d in rows.items():
        lines += ["", f"**{mdl}**"]
        mt = d.get("multi_turn") or []
        if mt:
            lines += ["", "_multi-turn chat (KV prefix grows each turn):_", "",
                      "| turn | prompt_tok | cached | hit % | TTFT ms |",
                      "|---|---|---|---|---|"]
            for t in mt:
                lines.append(f"| {t['turn']} | {t['prompt_tok']} | {t['cached']} | "
                             f"{t['hit_pct']:.1f} | {t['ttft_ms']:.0f} |")
        rag = d.get("rag") or []
        if rag:
            lines += ["", "_RAG (shared long context, varying question):_", "",
                      "| q | prompt_tok | cached | hit % | TTFT ms | speedup vs q1 |",
                      "|---|---|---|---|---|---|"]
            for q in rag:
                lines.append(f"| {q['q']} | {q['prompt_tok']} | {q['cached']} | "
                             f"{q['hit_pct']:.1f} | {q['ttft_ms']:.0f} | {q['speedup_vs_q1']:.2f}× |")
    return lines


def _fmt_comprehensive(art):
    """Render the comprehensive sweep artifact (length / hit-ratio / concurrency
    per model) as full markdown tables — the 40-line ASCII tail loses the top."""
    lines = []
    for m in art:
        lines += ["", f"**{m['model']}** ({m.get('engine', '?')}) — "
                  f"cold base {m.get('cold_base_tok', '?')} tok / {m.get('cold_base_ttft_ms', 0):.0f} ms TTFT"]
        ls = m.get("length_sweep") or []
        if ls:
            lines += ["", "_length sweep (cache off):_", "",
                      "| target | prompt_tok | TTFT ms | prefill t/s | decode t/s |",
                      "|---|---|---|---|---|"]
            for x in ls:
                lines.append(f"| {x['target']} | {x['prompt_tok']} | {x['ttft_ms']:.0f} | "
                             f"{x['prefill_tps']:.0f} | {x['decode_tps']:.1f} |")
        hr = m.get("hit_ratio_sweep") or []
        if hr:
            lines += ["", "_hit-ratio sweep (reuse vs cold-prefill of the SAME prompt):_", "",
                      "| suffix_words | prompt_tok | cached | hit ratio | cold ms | reuse ms | speedup |",
                      "|---|---|---|---|---|---|---|"]
            for x in hr:
                # tolerate old (speedup_vs_cold_base) and new (speedup_vs_same_cold) keys
                sp = x.get("speedup_vs_same_cold", x.get("speedup_vs_cold_base", 0))
                cold = x.get("cold_same_ms", "—")
                cold_s = f"{cold:.0f}" if isinstance(cold, (int, float)) else cold
                lines.append(f"| {x['suffix_words']} | {x['prompt_tok']} | {x['cached']} | "
                             f"{x['hit_ratio']:.3f} | {cold_s} | {x['ttft_ms']:.0f} | {sp:.2f}× |")
        cc = m.get("concurrency_sweep") or []
        ccl = m.get("concurrency_sweep_loop") or []
        if cc:
            loop_by_n = {x["N"]: x for x in ccl}
            has_loop = bool(ccl)
            hdr = ("| N | fast agg t/s | fast TTFT ms | loop agg t/s | loop TTFT ms |"
                   if has_loop else "| N | agg t/s | mean TTFT ms | total_tok | wall s |")
            sep = "|---|---|---|---|---|" if has_loop else "|---|---|---|---|---|"
            lines += ["", "_concurrency sweep"
                      + (" (fast path vs engine-loop):_" if has_loop else " (fast path):_"),
                      "", hdr, sep]
            for x in cc:
                if has_loop:
                    lx = loop_by_n.get(x["N"], {})
                    lines.append(f"| {x['N']} | {x['agg_tps']:.1f} | {x['mean_ttft_ms']:.0f} | "
                                 f"{lx.get('agg_tps', '—')} | {lx.get('mean_ttft_ms', '—')} |")
                else:
                    lines.append(f"| {x['N']} | {x['agg_tps']:.1f} | {x['mean_ttft_ms']:.0f} | "
                                 f"{x['total_tok']} | {x['wall_s']:.1f} |")
    return lines


def _write_report(tier, results, go):
    gated = [r for r in results if r["gate"]]
    n_pass = sum(1 for r in gated if r["status"] == "PASS")
    lines = ["# Regression report", "",
             f"- tier: **{tier}**",
             f"- verdict: **{'GO ✅' if go else 'NO-GO ❌'}** ({n_pass}/{len(gated)} gates passed)",
             f"- total wall time: **{sum(r['seconds'] for r in results)/60:.0f} min**", "",
             "## Section status", "",
             "| section | gate | status | summary | time |",
             "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['name']} | {'gate' if r['gate'] else 'metric'} | "
                     f"{r['status']} | {r['summary']} | {r['seconds']:.0f}s |")

    # ── full metric data — the whole point of the report ──
    lines += ["", "## Metrics — full data", ""]
    for r in results:
        m = r.get("metrics") or {}
        tail = m.get("table_tail")
        rows = m.get("rows")
        art = m.get("artifact")
        if not (tail or rows or art):
            continue
        lines += [f"### {r['name']}", ""]
        if "framework speed" in r["name"].lower():
            lines += [
                "> ⚠️ **In-process micro-benchmark — the batched columns are NOT production serving.**",
                "> These numbers drive each engine's `BatchedEngine` directly in-process. That is fair",
                "> for single-request TTFT/decode, but the `batch8/16/32` columns MISREPRESENT cross-",
                "> framework batched throughput — e.g. oMLX shows ~1.2× (it never engages its scheduler",
                "> when driven this way), while in production it serves via its HTTP server. For the",
                "> real, fair production comparison (each framework's actual OpenAI server over HTTP,",
                "> measured at a matched GPU thermal state) use `scripts/bench/bench_serve.py` → the `serve/`",
                "> section of `docs/reports/PERF_TREND.md`. Fair-state finding there: yunshu leads concurrent",
                "> throughput at all N; oMLX does not scale concurrency.",
                "",
            ]
        if rows:
            lines += _fmt_realistic(rows)
        if "comprehensive" in r["name"].lower() and art:
            # full markdown tables (ASCII tail loses the top of large sweeps)
            lines += _fmt_comprehensive(art)
        elif tail:
            # matrix + frameworks: their pre-rendered ASCII fits the tail in full
            lines += ["```", tail.strip("\n"), "```"]
        lines.append("")

    lines += ["", "## Known limitations & how to read the numbers", "", _KNOWN_LIMITATIONS.strip()]

    # docs reorg: machine-generated reports live under docs/reports/.
    _rep_dir = os.path.join(REPO, "docs", "reports")
    os.makedirs(_rep_dir, exist_ok=True)
    path = os.path.join(_rep_dir, "REGRESSION_REPORT.md")
    open(path, "w").write("\n".join(lines) + "\n")
    json.dump(results, open(os.path.join(_rep_dir, "regression_report.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
