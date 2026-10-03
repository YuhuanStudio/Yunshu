"""Dispatch canonical atoms without changing caches, samplers or token spans."""

import importlib
import threading
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from tests.unit.test_auxiliary_scheduling import job
from tests.unit.test_vlm_runner_batching import FakeGen, runner  # noqa: F401
from yunshu_engine.serving.work_scheduler import AGING_S, DECODE_QUANTUM_S, Work
from yunshu_gateway import admission


class DecodeRows(dict):
    @property
    def uids(self):
        return list(self)


class AtomGen(FakeGen):
    """Model the upstream decode-first next()/one-atom prefill interface."""

    clock = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prompt_batch = None
        self._unprocessed_sequences = []
        self._generation_batch = DecodeRows()
        self.completion_batch_size = 32
        self.atoms = []

    def insert(self, prompts, **kw):
        uids = super().insert(prompts, **kw)
        self._unprocessed_sequences.append((uids[0], prompts[0]))
        return uids

    def remove(self, uid):
        super().remove(uid)
        self._generation_batch.pop(uid, None)
        self._unprocessed_sequences = [
            s for s in self._unprocessed_sequences if s[0] != uid
        ]

    def next(self):
        out = []
        for uid in list(self._generation_batch):
            state = self.rows[uid]
            state[0] += 1
            out.append(
                SimpleNamespace(
                    uid=uid,
                    token=100 * uid + state[0],
                    token_logprob=0.0,
                    top_logprobs=None,
                    finish_reason="length" if state[0] >= state[1] else None,
                )
            )
            if state[0] >= state[1]:
                self.rows.pop(uid)
                self._generation_batch.pop(uid)
        if out:
            self.clock[0] += 0.05
        if self.completion_batch_size == 0 or (self.kwargs.get("draft_model") and out):
            return [], out
        if self._prompt_batch is None and self._unprocessed_sequences:
            uid, ids = self._unprocessed_sequences.pop(0)
            self._prompt_batch = SimpleNamespace(
                uids=[uid],
                _prompt_uids=[uid],
                _processed_prompt_columns=0,
                _input_ids=SimpleNamespace(shape=(1, len(ids))),
            )
        pb = self._prompt_batch
        if pb is not None:
            rest = pb._input_ids.shape[1]
            n = min(2048, rest)
            self.atoms.append((pb.uids[0], pb._processed_prompt_columns, n))
            pb._processed_prompt_columns += n
            pb._input_ids.shape = (1, rest - n)
            self.clock[0] += max(0.01, n * 0.00121)
            if rest == n:
                self._generation_batch[pb.uids[0]] = True
                self._prompt_batch = None
        return [], out


@pytest.fixture
def atoms(runner, monkeypatch):  # noqa: F811
    clock = [100.0]
    AtomGen.clock = clock
    monkeypatch.setattr(
        "yunshu_engine.vlm_batch_runner.time.perf_counter", lambda: clock[0]
    )
    monkeypatch.setenv("YUNSHU_UNCACHED_SCHEDULING", "1")
    monkeypatch.setattr(
        importlib.import_module("mlx_vlm.generate.ar"), "BatchGenerator", AtomGen
    )
    return runner, clock


def test_work_uses_suffix_and_wait_not_total_context():
    cold = Work(0, 0, 32768)
    cached = Work(0, 0, 128)
    assert cached.key(1) < cold.key(1)
    assert cold.key(AGING_S) < Work(AGING_S, AGING_S, 1).key(AGING_S)
    assert Work(0, 0, 650, -1).key(1) > cached.key(1)
    assert Work(0, 0, 650, -1).key(AGING_S) < cached.key(AGING_S - 1)


def test_metadata_peek_has_no_lookup_or_lru_effect(atoms):
    r, _ = atoms
    entries = OrderedDict(
        a=SimpleNamespace(token_ids=tuple(range(100)), extra_hash=7),
        b=SimpleNamespace(token_ids=tuple(range(200)), extra_hash=8),
    )
    r.apc_manager = SimpleNamespace(lock=threading.Lock(), _exact_cache=entries)
    j = job(0)
    j.ids, j.salt = list(range(110)), 7
    assert r._work(j).uncached_tokens == 10
    assert list(entries) == ["a", "b"]
    j.salt = 9
    assert r._work(j).uncached_tokens == 110
    j.priority = -1
    assert r._work(j).uncached_tokens == 110


def test_short_arrival_bypasses_suspended_cold_atom_and_resumes_state(atoms):
    r, _ = atoms
    cold = job(0)
    cold.ids = list(range(8192))
    r._submit(cold)
    r._drive_slice(False)
    g = r._groups()[0]
    saved = g.prefills[cold.uid]
    short = job(0)
    r._submit(short)
    r._drive_slice(False)
    assert g.gen.atoms[:2] == [(cold.uid, 0, 2048), (short.uid, 0, 2)]
    assert g.prefills[cold.uid] is saved
    for _ in range(80):
        r._drive_slice(False)
        if not r.busy():
            break
    assert not r.busy()
    assert [a[1:] for a in g.gen.atoms if a[0] == cold.uid] == [
        (0, 2048),
        (2048, 2048),
        (4096, 2048),
        (6144, 2048),
    ]
    assert cold.stats.generated == short.stats.generated == 2


def test_decode_quantum_is_paid_in_measured_time(atoms):
    r, _ = atoms
    dec = job(0)
    dec.max_tokens = 30
    r._submit(dec)
    r._drive_slice(False)
    cold = job(0)
    cold.ids = list(range(8192))
    r._submit(cold)
    g = r._groups()[0]
    for _ in range(2):
        r._drive_slice(False)
    assert len(g.gen.atoms) == 1
    for _ in range(3):
        r._drive_slice(False)
    assert any(a[0] == cold.uid for a in g.gen.atoms)
    assert dec.stats.generated >= 2


def test_cancel_suspended_prefill_releases_state(atoms):
    r, _ = atoms
    cold = job(0)
    cold.ids = list(range(8192))
    cold.cancel_event = threading.Event()
    r._submit(cold)
    r._drive_slice(False)
    g = r._groups()[0]
    assert cold.uid in g.prefills
    cold.cancel_event.set()
    r._drive_slice(False)
    assert not g.prefills and not r.busy()
    assert cold.stats.finish_reason == "cancel"


def test_auxiliary_ages_under_continuous_primary_traffic(atoms):
    r, clock = atoms
    aux, main = job(-1), job(0)
    main.max_tokens = 1000
    r._submit(aux)
    r._submit(main)
    r._drive_slice(False)
    assert aux in r._pending
    clock[0] += AGING_S
    for _ in range(20):
        r._drive_slice(False)
        clock[0] += 1
        if aux.stats.generated:
            break
    assert aux.stats.generated > 0
    assert main.stats.generated > 0


@pytest.mark.asyncio
async def test_gateway_auxiliary_age_escapes_endless_interactive_wait(monkeypatch):
    from yunshu_engine.request_tracker import current_request_info
    from yunshu_gateway.x_yunshu import RequestInfo, registry

    aux = RequestInfo(
        "aux", "POST", "/v1/chat/completions", scheduling_priority=-1, arrived=1
    )
    main = RequestInfo("main", "POST", "/v1/chat/completions", arrived=2)
    monkeypatch.setattr(registry, "active", lambda: [aux, main])
    monkeypatch.setattr(admission.time, "perf_counter", lambda: 1 + AGING_S)
    sleeps = []

    async def sleep(t):
        sleeps.append(t)

    monkeypatch.setattr("asyncio.sleep", sleep)
    token = current_request_info.set(aux)
    try:
        await admission.defer_auxiliary()
    finally:
        current_request_info.reset(token)
    assert sleeps == [0.5]


def test_paused_spec_decode_does_not_block_primary_prefill(atoms):
    r, clock = atoms
    aux = job(-1)
    aux.use_draft = True
    aux.max_tokens = 1000
    r._submit(aux)
    r._drive_slice(False)
    lane = r._aux_spec
    assert lane is not None and lane.gen._generation_batch
    main = job(0)
    main.use_draft = True
    main.ids = list(range(8192))
    r._submit(main)
    for _ in range(4):
        r._drive_slice(False)
    assert main.stats.prefill_done == main.stats.prefill_total
    assert len(r._spec.gen.atoms) == 4
    assert not lane.gen.rows[aux.uid][0]
    assert clock[0] - main.queued < 10


def test_cold_prefill_keeps_protection_after_short_overtake(atoms):
    r, _ = atoms
    cold = job(0)
    cold.ids = list(range(8192))
    r._submit(cold)
    r._drive_slice(False)
    g = r._groups()[0]
    short = job(0)
    r._submit(short)
    r._drive_slice(False)
    assert cold.prefill_skips == 1
    for _ in range(3):
        fresh = job(0)
        r._submit(fresh)
        # Drain decode debt, then exactly one protected cold atom.
        before = len(g.gen.atoms)
        for _ in range(10):
            r._drive_slice(False)
            if len(g.gen.atoms) > before:
                break
        assert g.gen.atoms[-1][0] == cold.uid
        assert cold.prefill_skips >= 1


def test_decode_slice_does_not_complete_suspended_prefill(atoms):
    r, _ = atoms
    cold = job(0)
    cold.ids = list(range(8192))
    r._submit(cold)
    r._drive_slice(False)
    short = job(0)
    short.max_tokens = 100
    r._submit(short)
    r._drive_slice(False)
    g = r._groups()[0]
    assert cold.uid in g.prefills
    assert cold.stats.prefill_done == 2048
    r._drive_slice(False)
    assert cold.stats.prefill_done == 2048
    assert r._work(cold).uncached_tokens == 6144


def test_last_atom_and_first_token_precede_decode_repayment(atoms):
    r, _ = atoms
    dec = job(0)
    dec.max_tokens = 100
    r._submit(dec)
    r._drive_slice(False)
    r._drive_slice(False)
    suffix = job(0)
    suffix.ids = list(range(128))
    suffix.max_tokens = 100
    r._submit(suffix)
    r._decode_debt = 0.1
    r._drive_slice(False)
    g = r._groups()[0]
    assert g.gen.atoms[-1] == (suffix.uid, 0, 128)
    assert not suffix.stats.t_first
    long = job(0)
    long.ids = list(range(8192))
    r._submit(long)
    r._drive_slice(False)
    assert suffix.stats.t_first
    assert not any(uid == long.uid for uid, _, _ in g.gen.atoms)
    # First-token delivery does not silently erase the fairness obligation.
    assert r._decode_debt == DECODE_QUANTUM_S
    for _ in range(4):
        r._drive_slice(False)
    assert any(uid == long.uid for uid, _, _ in g.gen.atoms)


def test_last_cold_atom_keeps_numeric_spans_and_bypasses_debt(atoms):
    r, _ = atoms
    cold = job(0)
    cold.ids = list(range(4096))
    r._submit(cold)
    r._drive_slice(False)
    dec = job(0)
    dec.max_tokens = 100
    r._submit(dec)
    r._drive_slice(False)
    r._drive_slice(False)  # Deliver the overtaking request's first token.
    r._decode_debt = 0.1
    r._drive_slice(False)
    g = r._groups()[0]
    assert [a[1:] for a in g.gen.atoms if a[0] == cold.uid] == [(0, 2048), (2048, 2048)]


def test_cancel_final_window_before_first_token_clears_priority(atoms):
    r, _ = atoms
    suffix = job(0)
    suffix.ids = list(range(128))
    suffix.cancel_event = threading.Event()
    r._submit(suffix)
    r._drive_slice(False)
    assert suffix.finishing_prefill and not suffix.stats.t_first
    suffix.cancel_event.set()
    next_job = job(0)
    r._submit(next_job)
    for _ in range(8):
        r._drive_slice(False)
        if not r.busy():
            break
    assert suffix.stats.finish_reason == "cancel"
    assert not suffix.stats.generated
    assert next_job.stats.generated == 2
    assert not r.busy()


def test_executed_atom_latches_its_remaining_final_window(atoms):
    r, _ = atoms
    cold = job(0)
    cold.ids = list(range(8192))
    r._submit(cold)
    r._drive_slice(False)
    suffix = job(0)
    suffix.ids = list(range(3000))
    suffix.max_tokens = 100
    r._submit(suffix)
    r._drive_slice(False)
    assert cold.prefill_skips == 1
    assert suffix.stats.prefill_done == 2048
    r._drive_slice(False)
    assert suffix.stats.prefill_done == 3000
    assert cold.stats.prefill_done == 2048
    r._drive_slice(False)
    assert suffix.stats.t_first
    assert cold.stats.prefill_done == 2048
    g = r._groups()[0]
    assert [a[1:] for a in g.gen.atoms if a[0] == suffix.uid] == [
        (0, 2048),
        (2048, 952),
    ]


def test_metadata_peek_uses_admitted_apc_namespace(atoms):
    r, _ = atoms
    r.apc_manager = SimpleNamespace(
        lock=threading.Lock(),
        _exact_cache={
            "base": SimpleNamespace(token_ids=tuple(range(100)), extra_hash=7),
            "planned": SimpleNamespace(token_ids=tuple(range(200)), extra_hash=8),
        },
    )
    j = job(0)
    j.ids, j.salt, j.apc_salt = list(range(210)), 7, 8
    assert r._work(j).uncached_tokens == 10


def test_aged_aux_yields_once_to_best_primary_final_window(atoms):
    r, clock = atoms
    background = job(0)
    background.max_tokens = 1000
    r._submit(background)
    r._drive_slice(False)
    r._drive_slice(False)
    aux = job(-1)
    aux.ids = list(range(738))
    r._submit(aux)
    clock[0] += AGING_S
    suffix = job(0)
    suffix.ids = list(range(128))
    r._submit(suffix)
    r._decode_debt = 0
    r._drive_slice(False)
    assert suffix.finishing_prefill and not aux.stats.prefill_done
    assert aux.prefill_skips == 1
    r._drive_slice(False)
    assert suffix.stats.t_first
    fresh = job(0)
    r._submit(fresh)
    r._decode_debt = 0
    r._drive_slice(False)
    assert aux.stats.prefill_done == 738
    assert not fresh.stats.prefill_done


def test_handoff_is_applied_after_finish_in_same_slice_and_expires(atoms):
    r, clock = atoms
    r.prefix_invariant = True
    main = job(0)
    main.max_tokens = 2
    r._submit(main)
    r._drive_slice(False)
    r._drive_slice(False)
    aux = job(-1)
    r._submit(aux)
    clock[0] += AGING_S
    r._decode_debt = 0
    r._drive_slice(False)
    assert main.stats.generated == 2
    assert not aux.stats.prefill_done
    started = r._primary_handoff_at
    assert started is not None
    # A second finish cannot extend this grace while auxiliary work waits.
    other = job(0)
    group = SimpleNamespace(jobs={42: other}, sampler=None, gen=SimpleNamespace())
    other.uid = 42
    clock[0] += 0.05
    r._finish(group, 42, "stop")
    assert r._primary_handoff_at == started
    clock[0] += 0.06
    r._drive_slice(False)
    assert aux.stats.prefill_done
    assert r._primary_handoff_at is None


def test_warm_estimate_cannot_latch_a_long_cold_miss(atoms, monkeypatch):
    r, _ = atoms
    dec = job(0)
    dec.max_tokens = 1000
    r._submit(dec)
    r._drive_slice(False)
    r._drive_slice(False)
    cold = job(0)
    cold.ids = list(range(8192))
    work = r._work

    def optimistic(j):
        if j is cold and not j.stats.prefill_done:
            return Work(j.queued, j.last_service, 128)
        return work(j)

    monkeypatch.setattr(r, "_work", optimistic)
    r._submit(cold)
    r._decode_debt = 0.1
    r._drive_slice(False)
    assert cold.stats.prefill_done == 2048
    assert not cold.finishing_prefill
    r._drive_slice(False)
    assert cold.stats.prefill_done == 2048


def test_handoff_yields_worker_but_not_a_ready_primary(atoms):
    r, clock = atoms
    r._primary_handoff_at = clock[0]
    assert r._handoff_delay() == 0.002
    primary = job(0)
    r._submit(primary)
    assert r._handoff_delay() == 0
    r._pending.clear()
    clock[0] += 0.11
    assert r._handoff_delay() == 0


def test_young_handoff_refreshes_but_aged_handoff_is_used_once(atoms):
    r, clock = atoms
    r.prefix_invariant = True
    aux = job(-1)
    r._submit(aux)

    def finish():
        primary = job(0)
        group = SimpleNamespace(jobs={42: primary}, sampler=None, gen=SimpleNamespace())
        r._finish(group, 42, "stop")

    finish()
    first = r._primary_handoff_at
    clock[0] += 0.05
    finish()
    assert r._primary_handoff_at > first
    clock[0] += AGING_S
    finish()
    aged = r._primary_handoff_at
    assert aux.handoff_graced
    clock[0] += 0.05
    finish()
    assert r._primary_handoff_at == aged
    clock[0] += 0.06
    r._drive_slice(False)
    assert aux.stats.prefill_done
    assert not aux.handoff_graced


def test_ready_primary_cancels_handoff_timer_without_duplicate_submit(atoms):
    r, _ = atoms
    calls, cancelled = [], []
    r._executor = SimpleNamespace(submit=lambda fn: calls.append(fn))
    r._driving = True
    timer = SimpleNamespace(cancel=lambda: cancelled.append(True))
    r._handoff_timer = timer
    r._submit(job(-1))
    assert not calls and not cancelled
    r._submit(job(0))
    assert len(calls) == 1 and cancelled == [True]
    assert r._handoff_timer is None
    r._wake_handoff(timer)
    assert len(calls) == 1


def test_sole_foreground_can_progress_before_reading_cache_estimates(
    atoms, monkeypatch
):
    r, _ = atoms
    foreground = job(0)
    foreground.ids = list(range(8192))
    r._admit(foreground, True)
    work = r._work

    def estimate(j):
        assert j.stats.prefill_done, "cache estimates must not gate a sole foreground"
        return work(j)

    monkeypatch.setattr(r, "_work", estimate)
    r._step_work_groups(True)
    assert foreground.stats.prefill_done == 2048


def test_two_40ms_decode_steps_release_a_canonical_atom(atoms, monkeypatch):
    r, clock = atoms
    dec = job(0)
    dec.max_tokens = 1000
    r._submit(dec)
    r._drive_slice(False)
    r._drive_slice(False)
    gen = r._groups()[0].gen
    next_step = gen.next

    def ar40ms():
        result = next_step()
        if result[1]:
            clock[0] -= 0.01  # AtomGen normally charges 50 ms per decode.
        return result

    monkeypatch.setattr(gen, "next", ar40ms)
    cold = job(0)
    cold.ids = list(range(8192))
    r._submit(cold)
    r._drive_slice(False)
    r._drive_slice(False)
    assert not cold.stats.prefill_done
    r._drive_slice(False)
    assert cold.stats.prefill_done == 2048
    assert dec.stats.generated >= 3
