"""MMLU Accuracy Benchmark: 4-framework parallel comparison.

Frameworks: mlx-lm, Yunshu, oMLX, vllm-mlx
Modes: thinking (1024 tok) + direct/non-thinking (32 tok)

Usage:
    PYTHONPATH=. uv run python scripts/bench_mmlu.py
    PYTHONPATH=. uv run python scripts/bench_mmlu.py --quick
    PYTHONPATH=. uv run python scripts/bench_mmlu.py --framework mlx-lm
    PYTHONPATH=. uv run python scripts/bench_mmlu.py --framework yunshu
    PYTHONPATH=. uv run python scripts/bench_mmlu.py --framework omlx
    PYTHONPATH=. uv run python scripts/bench_mmlu.py --framework vllm-mlx
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx

REF_DIR = Path(__file__).resolve().parent.parent.parent / "reference"

# ── MMLU Questions: 7 subjects × (5-shot + 4 test) ──

MMLU_SUBJECTS = {
    "abstract_algebra": {
        "examples": [
            {"q": "Find the degree of the polynomial 3x^4 + 2x^2 - 5.", "choices": ["2", "3", "4", "5"], "answer": "C"},
            {"q": "Is the set of all 2x2 matrices with real entries a group under addition?", "choices": ["No, not closed", "No, no inverse", "No, not associative", "Yes"], "answer": "D"},
            {"q": "What is the order of the symmetric group S3?", "choices": ["3", "6", "9", "12"], "answer": "B"},
            {"q": "In a ring R, which axiom is NOT required?", "choices": ["Associativity of addition", "Commutativity of multiplication", "Distributivity", "Existence of additive inverse"], "answer": "B"},
            {"q": "Find the inverse of the matrix [[1,2],[3,4]].", "choices": ["[[-2,1],[1.5,-0.5]]", "[[4,-2],[-3,1]]", "[[0.5,-1],[-0.75,0.5]]", "[[-2,1.5],[1,-1]]"], "answer": "A"},
        ],
        "test": [
            {"q": "Let G be a group of order 15. How many Sylow 3-subgroups does G have?", "choices": ["1", "3", "5", "15"], "answer": "A"},
            {"q": "What is the characteristic of the field Z/pZ where p is prime?", "choices": ["0", "p", "p-1", "1"], "answer": "B"},
            {"q": "Which of the following is a field?", "choices": ["Z (integers)", "Z/6Z", "Z/7Z", "2x2 matrices over R"], "answer": "C"},
            {"q": "The polynomial x^2 + 1 is irreducible over:", "choices": ["C (complex numbers)", "R (real numbers)", "Q (rational numbers)", "Z/pZ for any prime p"], "answer": "B"},
        ],
    },
    "college_physics": {
        "examples": [
            {"q": "What is the SI unit of electric current?", "choices": ["Volt", "Watt", "Ampere", "Ohm"], "answer": "C"},
            {"q": "Newton's second law states F =", "choices": ["mv", "ma", "mv^2", "mgh"], "answer": "B"},
            {"q": "The speed of light in vacuum is approximately:", "choices": ["3 × 10^6 m/s", "3 × 10^8 m/s", "3 × 10^10 m/s", "3 × 10^12 m/s"], "answer": "B"},
            {"q": "Which of these is a vector quantity?", "choices": ["Speed", "Temperature", "Mass", "Velocity"], "answer": "D"},
            {"q": "The unit of resistance is:", "choices": ["Henry", "Farad", "Ohm", "Tesla"], "answer": "C"},
        ],
        "test": [
            {"q": "A 5 kg object accelerates at 2 m/s^2. What force is applied?", "choices": ["2.5 N", "7 N", "10 N", "25 N"], "answer": "C"},
            {"q": "The period of a pendulum depends on:", "choices": ["Mass and length", "Length and gravity", "Mass and gravity", "Amplitude only"], "answer": "B"},
            {"q": "In an elastic collision, which quantity is conserved?", "choices": ["Only momentum", "Only kinetic energy", "Both momentum and kinetic energy", "Neither"], "answer": "C"},
            {"q": "What is the work done by a 10 N force moving an object 5 m in the force's direction?", "choices": ["2 J", "15 J", "50 J", "5 J"], "answer": "C"},
        ],
    },
    "world_history": {
        "examples": [
            {"q": "In which year did World War II end?", "choices": ["1943", "1944", "1945", "1946"], "answer": "C"},
            {"q": "The French Revolution began in which year?", "choices": ["1776", "1789", "1799", "1812"], "answer": "B"},
            {"q": "Who was the first Emperor of Rome?", "choices": ["Julius Caesar", "Augustus", "Nero", "Caligula"], "answer": "B"},
            {"q": "The Berlin Wall fell in which year?", "choices": ["1987", "1988", "1989", "1991"], "answer": "C"},
            {"q": "Which civilization built the pyramids at Giza?", "choices": ["Roman", "Greek", "Egyptian", "Mesopotamian"], "answer": "C"},
        ],
        "test": [
            {"q": "The Magna Carta was signed in which year?", "choices": ["1066", "1215", "1453", "1492"], "answer": "B"},
            {"q": "Which country was NOT part of the Allied Powers in WWI?", "choices": ["France", "Britain", "Ottoman Empire", "Russia"], "answer": "C"},
            {"q": "The Renaissance began in which country?", "choices": ["France", "England", "Italy", "Spain"], "answer": "C"},
            {"q": "The Industrial Revolution began in which country?", "choices": ["France", "Germany", "United States", "Britain"], "answer": "D"},
        ],
    },
    "machine_learning": {
        "examples": [
            {"q": "Which activation function outputs values between 0 and 1?", "choices": ["ReLU", "Tanh", "Sigmoid", "Leaky ReLU"], "answer": "C"},
            {"q": "Overfitting can be reduced by:", "choices": ["Adding more parameters", "Removing regularization", "Using dropout", "Increasing learning rate"], "answer": "C"},
            {"q": "The loss function used for binary classification is typically:", "choices": ["MSE", "Cross-entropy", "MAE", "Hinge loss"], "answer": "B"},
            {"q": "Gradient descent updates weights by:", "choices": ["w = w - lr * gradient", "w = w + lr * gradient", "w = lr * gradient", "w = gradient / lr"], "answer": "A"},
            {"q": "A CNN is best suited for:", "choices": ["Text processing", "Image recognition", "Time series", "Tabular data"], "answer": "B"},
        ],
        "test": [
            {"q": "Batch normalization is applied to:", "choices": ["Inputs only", "Hidden layer activations", "Loss function", "Gradients"], "answer": "B"},
            {"q": "The vanishing gradient problem is most severe with:", "choices": ["ReLU", "Sigmoid", "Leaky ReLU", "ELU"], "answer": "B"},
            {"q": "Transfer learning involves:", "choices": ["Training from scratch", "Using a pre-trained model", "Random initialization", "Only unsupervised learning"], "answer": "B"},
            {"q": "The Adam optimizer combines ideas from:", "choices": ["SGD and AdaGrad", "Momentum and RMSProp", "Newton and SGD", "L-BFGS and AdaGrad"], "answer": "B"},
        ],
    },
    "philosophy": {
        "examples": [
            {"q": "Who wrote 'Critique of Pure Reason'?", "choices": ["Hegel", "Kant", "Descartes", "Hume"], "answer": "B"},
            {"q": "Utilitarianism is primarily associated with:", "choices": ["Kant and Hegel", "Bentham and Mill", "Plato and Aristotle", "Descartes and Spinoza"], "answer": "B"},
            {"q": "'I think, therefore I am' was stated by:", "choices": ["Plato", "Aristotle", "Descartes", "Socrates"], "answer": "C"},
            {"q": "The 'veil of ignorance' thought experiment was proposed by:", "choices": ["Nozick", "Rawls", "Singer", "Foucault"], "answer": "B"},
            {"q": "Existentialism is most associated with:", "choices": ["Sartre", "Kant", "Hume", "Leibniz"], "answer": "A"},
        ],
        "test": [
            {"q": "The trolley problem is most associated with which branch of philosophy?", "choices": ["Aesthetics", "Epistemology", "Ethics", "Logic"], "answer": "C"},
            {"q": "Plato's allegory of the cave appears in which work?", "choices": ["The Republic", "Nicomachean Ethics", "Meditations", "Leviathan"], "answer": "A"},
            {"q": "Which philosopher is associated with the concept of the 'Ubermensch'?", "choices": ["Kant", "Hegel", "Nietzsche", "Schopenhauer"], "answer": "C"},
            {"q": "Pragmatism as a philosophical tradition originated in:", "choices": ["Britain", "France", "Germany", "United States"], "answer": "D"},
        ],
    },
    "computer_security": {
        "examples": [
            {"q": "SQL injection is an attack against:", "choices": ["Availability", "Confidentiality", "Integrity", "All of the above"], "answer": "D"},
            {"q": "AES is a:", "choices": ["Stream cipher", "Block cipher", "Hash function", "Key exchange"], "answer": "B"},
            {"q": "TLS stands for:", "choices": ["Total Layer Security", "Transport Layer Security", "Transmission Layer Standard", "Transfer Level Security"], "answer": "B"},
            {"q": "Phishing is a form of:", "choices": ["Malware", "Social engineering", "Buffer overflow", "SQL injection"], "answer": "B"},
            {"q": "A firewall typically operates at which OSI layers?", "choices": ["Only Layer 3", "Only Layer 4", "Layer 3 and 4", "Only Layer 7"], "answer": "C"},
        ],
        "test": [
            {"q": "What does a nonce prevent in cryptographic protocols?", "choices": ["Encryption", "Replay attacks", "Brute force", "Side-channel attacks"], "answer": "B"},
            {"q": "Zero-trust security means:", "choices": ["No security", "Trust no one, verify everyone", "Only trust internal networks", "Disable all firewalls"], "answer": "B"},
            {"q": "A buffer overflow attack exploits:", "choices": ["Memory management errors", "Network protocols", "Encryption weaknesses", "Authentication flaws"], "answer": "A"},
            {"q": "OAuth is primarily used for:", "choices": ["Encryption", "Authorization", "Authentication only", "Key management"], "answer": "B"},
        ],
    },
    "anatomy": {
        "examples": [
            {"q": "What is the largest bone in the human body?", "choices": ["Tibia", "Femur", "Humerus", "Fibula"], "answer": "B"},
            {"q": "Which chamber of the heart pumps blood to the lungs?", "choices": ["Left atrium", "Left ventricle", "Right atrium", "Right ventricle"], "answer": "D"},
            {"q": "The central nervous system consists of:", "choices": ["Brain and spinal cord", "Brain and nerves", "Spinal cord and nerves", "Brain, spinal cord, and nerves"], "answer": "A"},
            {"q": "Which type of muscle is found in the heart?", "choices": ["Skeletal", "Smooth", "Cardiac", "Voluntary"], "answer": "C"},
            {"q": "The functional unit of the kidney is the:", "choices": ["Nephron", "Neuron", "Lobule", "Alveolus"], "answer": "A"},
        ],
        "test": [
            {"q": "Which cranial nerve is responsible for facial expressions?", "choices": ["Trigeminal (V)", "Facial (VII)", "Vagus (X)", "Glossopharyngeal (IX)"], "answer": "B"},
            {"q": "The aorta carries blood from the:", "choices": ["Right ventricle to lungs", "Left ventricle to body", "Right atrium to ventricle", "Left atrium to ventricle"], "answer": "B"},
            {"q": "Hemoglobin is found in:", "choices": ["White blood cells", "Platelets", "Red blood cells", "Plasma"], "answer": "C"},
            {"q": "The diaphragm separates the:", "choices": ["Left and right lungs", "Thoracic and abdominal cavities", "Heart and lungs", "Brain and spinal cord"], "answer": "B"},
        ],
    },
}

QUICK_SUBJECTS = ["college_physics", "world_history", "machine_learning"]


def build_mmlu_prompt(subject: str, question: dict, examples: list[dict]) -> str:
    parts = [f"The following are multiple choice questions about {subject.replace('_', ' ')}.\n"]
    parts.append("Answer each question with ONLY the letter (A, B, C, or D).\n\n")
    for ex in examples:
        parts.append(f"Question: {ex['q']}\n")
        for i, label in enumerate("ABCD"):
            parts.append(f"  {label}. {ex['choices'][i]}\n")
        parts.append(f"Answer: {ex['answer']}\n\n")
    parts.append(f"Question: {question['q']}\n")
    for i, label in enumerate("ABCD"):
        parts.append(f"  {label}. {question['choices'][i]}\n")
    parts.append("Answer:")
    return "".join(parts)


def extract_answer(output: str) -> str | None:
    if not output or not output.strip():
        return None
    clean = re.sub(r'</?think[^>]*>', '', output)
    patterns = [
        r'(?:the\s+)?answer\s+is\s*\*\*?([A-D])\*\*?',
        r'(?:therefore|so|thus|hence)[,\s]+(?:the answer is )?([A-D])',
        r'(?:correct answer|correct choice)\s*(?:is|:)\s*([A-D])',
        r'\*\*([A-D])\*\*',
    ]
    for pat in patterns:
        m = re.search(pat, clean, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    lines = clean.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^([A-D])(?:[.\s]|$)', line)
        if m:
            return m.group(1).upper()
    matches = list(re.finditer(r'\b([A-D])\b', clean))
    if matches:
        return matches[-1].group(1).upper()
    return None


def P(msg):
    print(msg, flush=True)


def cleanup():
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


def apply_template(tokenizer, messages, thinking):
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if not thinking:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def eval_subjects(subjects, predict_fn):
    all_c, all_t = 0, 0
    subj_res = {}
    for sn, sd in subjects.items():
        c, t = 0, 0
        for q in sd["test"]:
            prompt = build_mmlu_prompt(sn, q, sd["examples"])
            output, elapsed, ntok = predict_fn(prompt)
            pred = extract_answer(output)
            if pred == q["answer"]:
                c += 1
            t += 1
        subj_res[sn] = {"accuracy": c / t if t else 0, "correct": c, "total": t}
        all_c += c
        all_t += t
    return subj_res, {"accuracy": all_c / all_t if all_t else 0, "correct": all_c, "total": all_t}


# ── mlx-lm ──

def run_mlx_lm(model_path, subjects, max_tokens, thinking):
    from mlx_lm.generate import generate_step
    from mlx_lm.utils import load as mlx_load

    model, tokenizer = mlx_load(model_path)

    def predict(prompt):
        text = apply_template(tokenizer, [{"role": "user", "content": prompt}], thinking)
        ids = mx.array(tokenizer.encode(text))
        t0 = time.perf_counter()
        toks = [t for t, _ in generate_step(ids, model, max_tokens=max_tokens)]
        return tokenizer.decode(toks, skip_special_tokens=True), time.perf_counter() - t0, len(toks)

    subj_res, overall = eval_subjects(subjects, predict)
    cleanup()
    return subj_res, overall


# ── Yunshu ──

def run_yunshu(model_path, subjects, max_tokens, thinking):
    from yunshu_engine.batched_engine import BatchedEngine

    async def _run():
        engine = BatchedEngine(model_name=model_path)
        await engine.start()
        subj_res = {}
        all_c, all_t = 0, 0
        for sn, sd in subjects.items():
            c, t = 0, 0
            for q in sd["test"]:
                prompt = build_mmlu_prompt(sn, q, sd["examples"])
                time.perf_counter()
                r = await engine.generate(
                    prompt=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens, temperature=0.0,
                    enable_thinking=thinking if thinking else False,
                )
                output = r.text if hasattr(r, 'text') else str(r)
                r.completion_tokens if hasattr(r, 'completion_tokens') else 0
                pred = extract_answer(output)
                if pred == q["answer"]:
                    c += 1
                t += 1
            subj_res[sn] = {"accuracy": c / t if t else 0, "correct": c, "total": t}
            all_c += c; all_t += t
        await engine.stop()
        return subj_res, {"accuracy": all_c / all_t if all_t else 0, "correct": all_c, "total": all_t}

    subj_res, overall = asyncio.run(_run())
    cleanup()
    return subj_res, overall


# ── oMLX ──

def run_omlx(model_path, subjects, max_tokens, thinking):
    sys.path.insert(0, str(REF_DIR / "omlx"))
    from omlx.models.llm import MLXLanguageModel

    llm = MLXLanguageModel(model_path)
    llm.load()

    def predict(prompt):
        t0 = time.perf_counter()
        r = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.0,
            enable_thinking=thinking if thinking else False,
        )
        return r.text, time.perf_counter() - t0, len(r.tokens) if hasattr(r, 'tokens') else 0

    subj_res, overall = eval_subjects(subjects, predict)
    cleanup()
    return subj_res, overall


# ── vllm-mlx ──

def run_vllm_mlx(model_path, subjects, max_tokens, thinking):
    sys.path.insert(0, str(REF_DIR / "vllm-mlx"))
    from mlx_lm.utils import load as mlx_load
    from vllm_mlx.engine_core import EngineConfig, EngineCore
    from vllm_mlx.request import SamplingParams

    model, tokenizer = mlx_load(model_path)

    async def _run():
        core = EngineCore(model, tokenizer, EngineConfig())
        await core.start()
        subj_res = {}
        all_c, all_t = 0, 0
        for sn, sd in subjects.items():
            c, t = 0, 0
            for q in sd["test"]:
                prompt = build_mmlu_prompt(sn, q, sd["examples"])
                text = apply_template(tokenizer, [{"role": "user", "content": prompt}], thinking)
                sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
                time.perf_counter()
                try:
                    out = await core.generate(prompt=text, sampling_params=sp)
                    output = out.new_text if hasattr(out, 'new_text') else str(out)
                    out.completion_tokens if hasattr(out, 'completion_tokens') else 0
                except Exception as e:
                    output = f"ERROR: {e}"
                pred = extract_answer(output)
                if pred == q["answer"]:
                    c += 1
                t += 1
            subj_res[sn] = {"accuracy": c / t if t else 0, "correct": c, "total": t}
            all_c += c; all_t += t
        await core.stop()
        return subj_res, {"accuracy": all_c / all_t if all_t else 0, "correct": all_c, "total": all_t}

    subj_res, overall = asyncio.run(_run())
    cleanup()
    return subj_res, overall


RUNNERS = {
    "mlx-lm": run_mlx_lm,
    "yunshu": run_yunshu,
    "omlx": run_omlx,
    "vllm-mlx": run_vllm_mlx,
}


# ── Reporting ──

def print_comparison(all_results):
    frameworks = []
    for fw in ["mlx-lm", "yunshu", "omlx", "vllm-mlx"]:
        for mode in ["think", "direct"]:
            if f"{fw}_{mode}" in all_results:
                if fw not in frameworks:
                    frameworks.append(fw)
                break
    if not frameworks:
        return

    P("\n" + "=" * 90)
    P("  MMLU ACCURACY COMPARISON: " + " vs ".join(fw.upper() for fw in frameworks))
    P("=" * 90)

    cw = 10
    header = f"  {'Subject':<22} │"
    for fw in frameworks:
        for mode in ["think", "direct"]:
            header += f" {mode:>{cw}}"
    P(header)
    P("  " + "─" * 22 + " ┼" + ("─" * (cw + 1) * 2) * len(frameworks))

    all_subjects = set()
    for r in all_results.values():
        all_subjects.update(r["subjects"].keys())
    for subj in sorted(all_subjects):
        row = f"  {subj:<22} │"
        for fw in frameworks:
            for mode in ["think", "direct"]:
                acc = all_results.get(f"{fw}_{mode}", {}).get("subjects", {}).get(subj, {}).get("accuracy", 0)
                row += f" {acc:>{cw}.0%}"
        P(row)

    P("  " + "─" * 22 + " ┼" + ("─" * (cw + 1) * 2) * len(frameworks))
    row = f"  {'OVERALL':<22} │"
    for fw in frameworks:
        for mode in ["think", "direct"]:
            acc = all_results.get(f"{fw}_{mode}", {}).get("overall", {}).get("accuracy", 0)
            row += f" {acc:>{cw}.0%}"
    P(row)

    P("\n  SUMMARY:")
    for fw in frameworks:
        parts = []
        for mode in ["direct", "think"]:
            acc = all_results.get(f"{fw}_{mode}", {}).get("overall", {}).get("accuracy")
            if acc is not None:
                parts.append(f"{mode}={acc:.0%}")
        P(f"    {fw:<10} {'  '.join(parts)}")

    # Parity check
    direct_accs = {fw: all_results.get(f"{fw}_direct", {}).get("overall", {}).get("accuracy")
                   for fw in frameworks}
    direct_accs = {k: v for k, v in direct_accs.items() if v is not None}
    if len(direct_accs) >= 2:
        vals = list(direct_accs.values())
        if max(vals) - min(vals) < 0.05:
            P(f"\n  ✅ All frameworks match in direct mode (Δ < 5%): {', '.join(f'{k} {v:.0%}' for k, v in direct_accs.items())}")
        else:
            P("\n  ⚠ Direct mode spread:")
            for fw, acc in sorted(direct_accs.items(), key=lambda x: -x[1]):
                P(f"    {fw:<10} {acc:.0%}")


def main():
    parser = argparse.ArgumentParser(description="MMLU accuracy: 4-framework comparison")
    parser.add_argument("--model", default="models/Qwen3.5-9B-MLX-4bit")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--framework", choices=["all", "mlx-lm", "yunshu", "omlx", "vllm-mlx"], default="all")
    args = parser.parse_args()

    if not Path(args.model).exists():
        P(f"Model not found: {args.model}")
        sys.exit(1)

    subjects = {k: MMLU_SUBJECTS[k] for k in (QUICK_SUBJECTS if args.quick else MMLU_SUBJECTS.keys())}
    n_q = sum(len(v["test"]) for v in subjects.values())
    fws = list(RUNNERS.keys()) if args.framework == "all" else [args.framework]

    P(f"Model: {args.model}")
    P(f"Subjects: {len(subjects)}, Questions: {n_q}")
    P(f"Frameworks: {', '.join(fws)}")
    P("Format: MMLU 5-shot (A/B/C/D)")

    all_results = {}
    phase = 1

    for fw in fws:
        for thinking, mode_key in [(True, "think"), (False, "direct")]:
            max_tokens = 1024 if thinking else 32
            mode = "thinking" if thinking else "direct"
            P(f"\nPhase {phase}: {fw} {mode} (max_tokens={max_tokens})")
            try:
                subj_res, overall = RUNNERS[fw](args.model, subjects, max_tokens, thinking)
                all_results[f"{fw}_{mode_key}"] = {
                    "framework": fw, "thinking": thinking, "max_tokens": max_tokens,
                    "subjects": subj_res, "overall": overall,
                }
                for sn, data in subj_res.items():
                    P(f"  {fw} {mode} {sn}: {data['correct']}/{data['total']} ({data['accuracy']:.0%})")
                P(f"  → {fw} {mode}: {overall['correct']}/{overall['total']} ({overall['accuracy']:.0%})")
            except Exception as e:
                import traceback
                P(f"  → {fw} {mode}: FAILED ({e})")
                traceback.print_exc()
            phase += 1

    print_comparison(all_results)

    if args.json:
        P(json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
