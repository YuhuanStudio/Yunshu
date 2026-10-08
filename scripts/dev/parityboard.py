#!/usr/bin/env python3
"""CPU-only per-item snapshot gate. Unknown evidence never establishes parity.

Product ranking includes different quantizations, explicitly recorded in each sample.
Noise = median absolute deviation across >=3 independent repetitions; parity requires
our directional gap <= combined MAD (not overlapping min/max ranges).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

ENGINES = ('yunshu-new', 'llamacpp', 'mlxlm', 'omlx', 'splash', 'tf-new', 'mtplx', 'strata')
CONTEXTS = (1024, 32768, 131072)
MODEL = 'Qwen3.8-27B'
# Memory cells compare only when measured as host-level used-memory delta. Per-process phys_footprint (single
# PID or whole tree) misses file-backed/mmap'd weights, so engines that mmap (e.g. Splash: 3.4 GiB on a 27B) look tiny.
COMPARABLE_MEMORY_METHOD = 'system-delta'
METRICS = {
    'ttft_cold_s': False, 'ttft_warm_s': False, 'followup_ttft_s': False,
    'decode_cold_tps': True, 'decode_warm_tps': True, 'decode_turn2_tps': True,
    'prefill_cold_tps': True, 'energy_j_token': False, 'agentic_session_s': False,
    'memory_peak_gib': False, 'memory_idle_gib': False, 'accuracy_needle': True,
}


def read_jsonl(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError('empty or non-object JSONL')
    return rows


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_cells(runs, jobs):
    """Choose latest submitted attempt per logical cell, then verify its actual job/output.

    Never fall back to an earlier attempt after latest failed. Explicit contamination
    markers block named cells; only a job with rerun-contam in its label can replace it.
    A marker with no parseable cell list blocks the entire run.
    """
    chosen = {}
    rejected = []
    sources = []
    for run in sorted(runs):
        events = run / 'snapshot.jsonl'
        if not events.exists():
            continue
        try:
            submissions = [r for r in read_jsonl(events) if r.get('ev') == 'cell_submitted']
        except (OSError, ValueError) as exc:
            rejected.append({'run': str(run), 'reason': str(exc)})
            continue
        marker = run / 'CONTAMINATED.md'
        contamination = marker.read_text() if marker.exists() else ''
        blocked = set(re.findall(r'snapshot\.([\w-]+)', contamination))
        sources.append({'run': str(run), 'events_sha256': hashlib.sha256(events.read_bytes()).hexdigest(),
                        'contamination': contamination})
        for event in submissions:
            cell = event.get('cell', '')
            if 'pilot' in cell or cell.startswith('yunshu-base'):
                continue
            key = cell
            stamp = event.get('t', 0)
            if key not in chosen or stamp >= chosen[key][0]:
                chosen[key] = (stamp, run, event, cell in blocked or bool(contamination and not blocked))
    accepted = []
    for _, run, event, contaminated in chosen.values():
        job_id = event.get('job', '')
        try:
            job = json.loads((jobs / (job_id + '.json')).read_text())
            if job.get('state') != 'done' or job.get('rc') != 0:
                raise ValueError('job incomplete/failed or missing exit status')
            if job.get('device') != 'm5' or job.get('contended') is not False or not job.get('quiet'):
                raise ValueError('not quiet clean M5 evidence')
            if contaminated and 'rerun-contam' not in job.get('label', ''):
                raise ValueError('CONTAMINATED.md excludes this cell; verified rerun required')
            argv = event['argv']
            path = Path(argv[argv.index('--out') + 1])
            if path.parent.resolve() != (run / 'cells').resolve():
                raise ValueError('output is outside this run cells directory')
            rows = read_jsonl(path)
            if rows[-1].get('part') != 'part_done' or rows[-1].get('complete') is not True:
                raise ValueError('missing terminal complete')
            sessions = [r for r in rows if r.get('part') == 'session']
            if len(sessions) != 1:
                raise ValueError('expected one session')
            session = sessions[0]
            if not session.get('spec_mode_expected') or session.get('engaged_spec_mode') != session['spec_mode_expected']:
                raise ValueError('engaged mode not proven')
            meta = ('engine', 'git_sha', 'checkpoint', 'version', 'snapshot_rep')
            if any(k not in session for k in meta):
                raise ValueError('missing model/version/rep provenance')
            if any(any(row.get(k) != session[k] for k in meta) for row in rows):
                raise ValueError('mixed provenance')
            if any(row.get('device') != 'm5' for row in rows):
                raise ValueError('mixed or missing device')
            decode = [r for r in rows if r.get('part') == 'decode']
            if decode:
                ctxs = {r['ctx'] for r in decode}
                kinds = {r['kind'] for r in decode}
                expected = {(c, k, p) for c in ctxs for k in kinds for p in ('cold', 'warm', 'turn2')}
                actual = [(r['ctx'], r['kind'], r['phase']) for r in decode]
                if set(actual) != expected or len(actual) != len(expected):
                    raise ValueError('incomplete/duplicate decode phases')
                for row in decode:
                    target = 256 if row['phase'] == 'turn2' else 2048
                    if row.get('ct') != target or row.get('finish') != 'length':
                        raise ValueError('incomplete decode reply')
                    if not all(number(row.get(k)) and row[k] > 0 for k in ('ttft_s', 'dec_tps', 'pt')):
                        raise ValueError('invalid decode metrics')
                if len([r for r in rows if r.get('part') == 'memory']) != 1:
                    raise ValueError('missing memory')
            needles = [r for r in rows if r.get('part') == 'needle']
            if needles and (len(needles) != 10 or len({r.get('item') for r in needles}) != 10
                            or any(not isinstance(r.get('correct'), bool) for r in needles)):
                raise ValueError('incomplete accuracy items')
            accepted.append({'cell': event['cell'], 'job': job_id, 'file': str(path),
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'rows': rows})
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            rejected.append({'cell': event.get('cell'), 'job': job_id, 'reason': str(exc)})
    return accepted, rejected, sources


def samples_from(cells, flagged=None):
    flagged = [] if flagged is None else flagged
    samples = defaultdict(dict)
    for cell in cells:
        rows = cell['rows']
        session = next(r for r in rows if r['part'] == 'session')
        # Snapshot currently has one model; future adapters must provide model explicitly.
        model = session.get('model', MODEL if 'Qwen3.8-27B' in session['checkpoint'] else session['checkpoint'])
        rep = session['snapshot_rep']
        engine = session['engine']
        def add(ctx, kind, metric, value):
            if number(value) and value >= 0:
                key = (model, ctx, kind, metric, engine)
                evidence = {'value': value, 'rep': rep, 'job': cell['job'], 'file': cell['file'],
                            'checkpoint': session['checkpoint'], 'git_sha': session['git_sha'],
                            'version': session['version'], 'mode': session['engaged_spec_mode'],
                            'weights_vs_oQ4e': session.get('weights_vs_oQ4e', 'unknown')}
                # Duplicate rep from separate files must not inflate sample size.
                if rep in samples[key]:
                    samples[key][rep] = None
                else:
                    samples[key][rep] = evidence
        decode = [r for r in rows if r['part'] == 'decode']
        for row in decode:
            phase = row['phase']
            ttft = {'cold': 'ttft_cold_s', 'warm': 'ttft_warm_s', 'turn2': 'followup_ttft_s'}[phase]
            add(row['ctx'], row['kind'], ttft, row['ttft_s'])
            add(row['ctx'], row['kind'], f'decode_{phase}_tps', row['dec_tps'])
            # Prefer engine prefill accounting; do not mislabel HTTP pt/TTFT as prefill.
            if phase == 'cold':
                add(row['ctx'], row['kind'], 'prefill_cold_tps', (row.get('xy') or {}).get('prefill_tps'))
            add(row['ctx'], row['kind'], 'energy_j_token', row.get('energy_j_token'))
        for row in rows:
            if row['part'] == 'memory':
                method = row.get('memory_method', 'single-pid')
                if method != COMPARABLE_MEMORY_METHOD:
                    flagged.append({'cell': cell['cell'], 'engine': engine, 'method': method})
                    continue
                for ctx in {r['ctx'] for r in decode}:
                    for metric, field in (('memory_peak_gib', 'peak_gib'), ('memory_idle_gib', 'idle_gib')):
                        add(ctx, 'session', metric, row.get(field))
            elif row['part'] == 'agentic':
                if row.get('success') is True:
                    add(row['ctx'], row['task'], 'agentic_session_s', row.get('total_s'))
        needle = [r for r in rows if r['part'] == 'needle']
        if needle:
            add(needle[0]['ctx'], 'needle', 'accuracy_needle', sum(r['correct'] for r in needle) / len(needle))
    return samples


def summarize(reps, minimum=3, strict=True):
    evidence = [r for r in reps.values() if r is not None]
    if len(evidence) < minimum or (strict and len({(r['git_sha'], r['checkpoint'], r['version'], r['mode']) for r in evidence}) != 1):
        return {'status': 'unknown', 'reason': 'need >=3 independent reps with identical provenance', 'samples': evidence}
    values = [r['value'] for r in evidence]
    median = statistics.median(values)
    mad = statistics.median(abs(v - median) for v in values)
    return {'status': 'measured', 'median': median, 'mad': mad, 'samples': evidence}


def build(runs, jobs, models=(MODEL,)):
    cells, rejected, sources = load_cells(runs, jobs)
    flagged = []
    samples = samples_from(cells, flagged)
    keys = {(m, c, k, metric) for m in models for c in CONTEXTS for metric in METRICS
            for k in (('session',) if metric.startswith('memory') else ('needle',) if metric.startswith('accuracy')
                      else ('agentic',) if metric.startswith('agentic') else ('prose', 'code'))}
    keys.update(key[:4] for key in samples)
    items = []
    for key in sorted(keys):
        model, ctx, kind, metric = key
        engines = {engine: summarize(samples.get((*key, engine), {})) for engine in ENGINES}
        measured = [(engine, stat) for engine, stat in engines.items() if stat['status'] == 'measured']
        high = METRICS[metric]
        best = (max if high else min)(measured, key=lambda p: p[1]['median']) if measured else None
        ours = engines['yunshu-new']
        item = {'model': model, 'ctx': ctx, 'kind': kind, 'metric': metric, 'higher_is_better': high,
                'engines': engines, 'best_engine': best[0] if best else None,
                'best': best[1]['median'] if best else None, 'ours': ours.get('median'),
                'ratio': None, 'gap': None, 'noise_band': None, 'status': 'unknown'}
        # At least one external runtime is required; self-only evidence never opens gate.
        if best and ours['status'] == 'measured' and any(e != 'yunshu-new' for e, _ in measured):
            b, o = best[1]['median'], ours['median']
            gap = b-o if high else o-b
            noise = best[1]['mad'] + ours['mad']
            item.update(ratio=o/b if b else None, gap=gap, noise_band=noise,
                        status='parity' if gap <= noise else 'gap')
        # Informational only (never gating): best-so-far with <3 reps.
        prov = {e: summarize(samples.get((*key, e), {}), minimum=1, strict=False) for e in ENGINES}
        pm = [(e, v) for e, v in prov.items() if v['status'] == 'measured']
        pb = (max if high else min)(pm, key=lambda p: p[1]['median']) if pm else None
        item['provisional'] = None
        if pb and prov['yunshu-new']['status'] == 'measured' and len(pm) > 1:
            b, o = pb[1]['median'], prov['yunshu-new']['median']
            item['provisional'] = {'best_engine': pb[0], 'best': b, 'ours': o, 'ratio': o / b if b else None,
                                   'reps': {e: len(v['samples']) for e, v in pm}}
        items.append(item)
    n = sum(i['status'] == 'parity' for i in items)
    unknown = [f"{i['model']}/{i['ctx']}/{i['kind']}/{i['metric']}" for i in items if i['status'] == 'unknown']
    groups = defaultdict(list)
    for i in items:
        if i['status'] == 'unknown':
            groups[i['metric']].append(f"{i['ctx']}/{i['kind']}")
    verdict = f"parity: {n}/{len(items)} items, missing: " + (
        '; '.join(f"{m} x{len(v)}" for m, v in sorted(groups.items())) or 'none')
    return {'schema_version': 1, 'verdict': verdict, 'parity': n, 'total': len(items),
            'missing': unknown, 'gate_open': n == len(items), 'items': items,
            'rejected': rejected, 'sources': sources, 'memory_method_flagged': flagged,
            'policy': 'Product ranking, not same-checkpoint proof; >=3 reps; combined MAD; missing engines listed individually; no historical markdown numbers used as gate evidence.'}


def markdown(board):
    lines = ['# Per-item parity board', '', board['verdict'], '', board['policy'], '',
             'Ratio = ours / best; positive directional gap means Yunshu is worse. Values are M5 only.',
             'Memory is whole cell-group process-tree peak/30s idle, not per-request RAM. Prefill uses engine accounting only.',
             '', '| Model | Context | Kind | Item | Best engine | Best | Ours | Ratio | Gap | MAD band | Status |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for i in board['items']:
        vals = [i[k] for k in ('model','ctx','kind','metric','best_engine','best','ours','ratio','gap','noise_band','status')]
        lines.append('| ' + ' | '.join('unknown' if v is None else f'{v:.6g}' if isinstance(v,float) else str(v) for v in vals) + ' |')
    lines += ['', '## Provisional (informational, <3 reps, never gating)', '',
              '| Ctx | Kind | Item | Best engine | Best | Ours | Ratio | Reps |', '|---|---|---|---|---|---|---|---|']
    for i in board['items']:
        pv = i.get('provisional')
        if pv and i['status'] == 'unknown':
            lines.append(f"| {i['ctx']} | {i['kind']} | {i['metric']} | {pv['best_engine']} | {pv['best']:.5g} | "
                         f"{pv['ours']:.5g} | {pv['ratio']:.3f} | {pv['reps']} |")
    flagged = board.get('memory_method_flagged', [])
    lines += ['', '## Memory cells needing rerun (method != system-delta; never counted as gap or parity)', '',
              'phys_footprint (single PID or process tree) excludes mmap/file-backed weights, so it is not comparable across engines.',
              'Rerun with memory_method=system-delta (host used-memory delta vs a pre-launch baseline).', '']
    lines += [f"- {f['cell']} ({f['engine']}): method={f['method']}" for f in flagged]
    lines += ['', '## Excluded evidence', '']
    lines += [f"- {r.get('cell', r.get('run'))}: {r['reason']} (job {r.get('job', 'unknown')})" for r in board['rejected']]
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', type=Path, default=Path('/Volumes/P5Plus/yunshu-build/verify/runs'))
    ap.add_argument('--jobs', type=Path, default=Path('/Volumes/P5Plus/yunshu-gpuq/jobs'))
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[2] / 'docs/research/parityboard')
    ap.add_argument('--model', action='append')
    args = ap.parse_args()
    board = build(args.runs.glob('snapshot014-*'), args.jobs, args.model or (MODEL,))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'board.json').write_text(json.dumps(board, indent=2, ensure_ascii=False) + '\n')
    (args.out / 'BOARD.md').write_text(markdown(board))
    print(board['verdict'])
    return 0 if board['gate_open'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
