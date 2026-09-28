"""Tests for Diffusion pipeline infrastructure — scheduler and LoRA offloader."""

import pytest

from yunshu_engine.diffusion_infra import (
    DiffusionLoRAOffloader,
    DiffusionScheduler,
    DiffusionStep,
    MemoryBudget,
    NoiseScheduleType,
    SchedulerType,
)

# ── DiffusionScheduler tests ──


class TestDiffusionScheduler:
    def test_default_construction(self):
        sched = DiffusionScheduler()
        assert sched.num_train_timesteps == 1000
        assert sched.num_inference_steps == 28
        assert sched.cfg_scale == 7.5
        assert len(sched.timesteps) == 28

    def test_custom_steps(self):
        sched = DiffusionScheduler(num_inference_steps=4, num_train_timesteps=100)
        assert len(sched.timesteps) == 4
        assert sched.total_steps == 4

    def test_timesteps_descending(self):
        """Timesteps should be in descending order (high noise → low noise)."""
        sched = DiffusionScheduler(num_inference_steps=10, num_train_timesteps=100)
        ts = sched.timesteps
        assert ts == sorted(ts, reverse=True)

    def test_get_step(self):
        sched = DiffusionScheduler(num_inference_steps=4, num_train_timesteps=100)
        step = sched.get_step(0)
        assert isinstance(step, DiffusionStep)
        assert step.step_index == 0
        assert step.is_last is False

    def test_get_step_last(self):
        sched = DiffusionScheduler(num_inference_steps=4, num_train_timesteps=100)
        step = sched.get_step(3)
        assert step.is_last is True

    def test_get_step_out_of_range(self):
        sched = DiffusionScheduler(num_inference_steps=4, num_train_timesteps=100)
        with pytest.raises(IndexError):
            sched.get_step(4)

    def test_steps_method(self):
        sched = DiffusionScheduler(num_inference_steps=4, num_train_timesteps=100)
        steps = sched.steps()
        assert len(steps) == 4
        assert all(isinstance(s, DiffusionStep) for s in steps)

    def test_iter_steps(self):
        sched = DiffusionScheduler(num_inference_steps=4, num_train_timesteps=100)
        steps = list(sched.iter_steps())
        assert len(steps) == 4
        assert sched.current_step == 4  # Past the end

    def test_iter_steps_with_range(self):
        sched = DiffusionScheduler(
            num_inference_steps=10, num_train_timesteps=100, start_step=2, end_step=5
        )
        steps = list(sched.iter_steps())
        assert len(steps) == 3
        assert steps[0].step_index == 2
        assert steps[2].step_index == 4

    def test_reset(self):
        sched = DiffusionScheduler(
            num_inference_steps=4, num_train_timesteps=100, start_step=0
        )
        list(sched.iter_steps())
        sched.reset()
        assert sched.current_step == 0

    def test_total_steps(self):
        sched = DiffusionScheduler(num_inference_steps=10, start_step=2, end_step=7)
        assert sched.total_steps == 5

    def test_sigmas_computed(self):
        sched = DiffusionScheduler(num_inference_steps=4, num_train_timesteps=100)
        assert len(sched.sigmas) == 4
        assert all(s > 0 for s in sched.sigmas)

    def test_add_noise_parameters(self):
        sched = DiffusionScheduler(num_inference_steps=4, num_train_timesteps=100)
        result = sched.add_noise(None, None, timestep=0)
        sqrt_alpha, sqrt_one_minus = result
        assert sqrt_alpha > 0
        assert sqrt_one_minus >= 0


class TestDiffusionSchedulerNoiseSchedules:
    def test_linear_schedule(self):
        sched = DiffusionScheduler(noise_schedule=NoiseScheduleType.LINEAR)
        assert len(sched._betas) == 1000
        assert sched._betas[0] == pytest.approx(0.00085, abs=1e-5)
        assert sched._betas[-1] == pytest.approx(0.012, abs=1e-4)

    def test_scaled_linear_schedule(self):
        sched = DiffusionScheduler(noise_schedule=NoiseScheduleType.SCALED_LINEAR)
        assert len(sched._betas) == 1000

    def test_cosine_schedule(self):
        sched = DiffusionScheduler(noise_schedule=NoiseScheduleType.COSINE)
        assert len(sched._betas) == 1000
        # Cosine schedule should have smooth betas
        assert all(0 < b < 1 for b in sched._betas)

    def test_sqrt_linear_schedule(self):
        sched = DiffusionScheduler(noise_schedule=NoiseScheduleType.SQRT_LINEAR)
        assert len(sched._betas) == 1000

    def test_different_step_counts(self):
        sched = DiffusionScheduler(num_inference_steps=1, num_train_timesteps=100)
        assert len(sched.timesteps) == 1

    def test_invalid_inference_steps(self):
        with pytest.raises(ValueError, match="num_inference_steps"):
            DiffusionScheduler(num_inference_steps=0)

    def test_inference_exceeds_train(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            DiffusionScheduler(num_inference_steps=2000, num_train_timesteps=1000)

    def test_invalid_start_step(self):
        with pytest.raises(ValueError, match="start_step"):
            DiffusionScheduler(num_inference_steps=10, start_step=10, end_step=10)


class TestDiffusionSchedulerTypes:
    def test_ddim_scheduler(self):
        sched = DiffusionScheduler(scheduler_type=SchedulerType.DDIM)
        assert sched.scheduler_type == SchedulerType.DDIM
        result = sched.scale_model_input(None, 0)
        assert result is None  # DDIM: no scaling

    def test_euler_scheduler(self):
        sched = DiffusionScheduler(scheduler_type=SchedulerType.EULER)
        result = sched.scale_model_input(None, 0)
        # Euler returns scaling factor
        assert isinstance(result, float)

    def test_dpm_scheduler(self):
        sched = DiffusionScheduler(scheduler_type=SchedulerType.DPM_PLUS_PLUS)
        assert len(sched.timesteps) == 28


# ── MemoryBudget tests ──


class TestMemoryBudget:
    def test_available(self):
        budget = MemoryBudget(total_bytes=1000, used_bytes=300)
        assert budget.available_bytes == 700

    def test_utilization(self):
        budget = MemoryBudget(total_bytes=1000, used_bytes=500)
        assert budget.utilization == pytest.approx(0.5)

    def test_allocate(self):
        budget = MemoryBudget(total_bytes=1000)
        assert budget.allocate(500) is True
        assert budget.used_bytes == 500
        assert budget.available_bytes == 500

    def test_allocate_over_budget(self):
        budget = MemoryBudget(total_bytes=1000)
        assert budget.allocate(1500) is False
        assert budget.used_bytes == 0

    def test_free(self):
        budget = MemoryBudget(total_bytes=1000, used_bytes=500)
        budget.free(200)
        assert budget.used_bytes == 300

    def test_free_below_zero(self):
        budget = MemoryBudget(total_bytes=1000, used_bytes=100)
        budget.free(200)
        assert budget.used_bytes == 0

    def test_peak_tracking(self):
        budget = MemoryBudget(total_bytes=1000)
        budget.allocate(500)
        budget.allocate(300)
        assert budget.peak_bytes == 800
        budget.free(300)
        assert budget.peak_bytes == 800  # Peak doesn't decrease

    def test_zero_total(self):
        budget = MemoryBudget(total_bytes=0)
        assert budget.available_bytes == 0
        assert budget.utilization == 0.0


# ── DiffusionLoRAOffloader tests ──


class TestDiffusionLoRAOffloader:
    def test_register_adapter(self):
        offloader = DiffusionLoRAOffloader(memory_budget_bytes=10000)
        offloader.register_adapter("style_lora", memory_bytes=1000, priority=5)
        assert "style_lora" not in offloader.loaded_adapters

    def test_load_for_step(self):
        offloader = DiffusionLoRAOffloader(memory_budget_bytes=10000)
        offloader.register_adapter(
            "style_lora", memory_bytes=1000, priority=5, assigned_steps=range(0, 14)
        )
        loaded = offloader.load_for_step(0)
        assert "style_lora" in loaded
        assert "style_lora" in offloader.loaded_adapters

    def test_load_explicit_ids(self):
        offloader = DiffusionLoRAOffloader(memory_budget_bytes=10000)
        offloader.register_adapter("lora_a", memory_bytes=1000)
        loaded = offloader.load_for_step(0, lora_ids=["lora_a"])
        assert "lora_a" in loaded

    def test_unload_after_step(self):
        offloader = DiffusionLoRAOffloader(memory_budget_bytes=10000)
        offloader.register_adapter(
            "lora_a", memory_bytes=1000, assigned_steps=range(0, 5)
        )
        offloader.load_for_step(0)
        unloaded = offloader.unload_after_step(4)  # Last step of range
        assert "lora_a" in unloaded

    def test_unload_not_needed(self):
        offloader = DiffusionLoRAOffloader(memory_budget_bytes=10000)
        offloader.register_adapter(
            "lora_a", memory_bytes=1000, assigned_steps=range(0, 10)
        )
        offloader.load_for_step(0)
        unloaded = offloader.unload_after_step(3)  # Still in range
        assert "lora_a" not in unloaded

    def test_unload_all(self):
        offloader = DiffusionLoRAOffloader(memory_budget_bytes=10000)
        offloader.register_adapter("lora_a", memory_bytes=1000)
        offloader.register_adapter("lora_b", memory_bytes=1000)
        offloader.load_for_step(0, lora_ids=["lora_a", "lora_b"])
        assert len(offloader.loaded_adapters) == 2
        unloaded = offloader.unload_all()
        assert len(unloaded) == 2
        assert len(offloader.loaded_adapters) == 0

    def test_memory_budget_tracking(self):
        offloader = DiffusionLoRAOffloader(memory_budget_bytes=10000)
        offloader.register_adapter("lora_a", memory_bytes=3000)
        offloader.register_adapter("lora_b", memory_bytes=3000)
        offloader.load_for_step(0, lora_ids=["lora_a"])
        assert offloader.memory.used_bytes == 3000
        offloader.load_for_step(0, lora_ids=["lora_b"])
        assert offloader.memory.used_bytes == 6000

    def test_eviction_on_memory_pressure(self):
        offloader = DiffusionLoRAOffloader(memory_budget_bytes=5000)
        offloader.register_adapter("low_prio", memory_bytes=4000, priority=1)
        offloader.register_adapter("high_prio", memory_bytes=4000, priority=10)

        offloader.load_for_step(0, lora_ids=["low_prio"])
        assert "low_prio" in offloader.loaded_adapters

        # Loading high_prio should evict low_prio (memory budget only 5000)
        loaded = offloader.load_for_step(1, lora_ids=["high_prio"])
        assert "high_prio" in loaded
        assert "low_prio" not in offloader.loaded_adapters

    def test_load_unload_counts(self):
        offloader = DiffusionLoRAOffloader(memory_budget_bytes=10000)
        offloader.register_adapter("lora_a", memory_bytes=1000)
        offloader.load_for_step(0, lora_ids=["lora_a"])
        assert offloader.load_count == 1
        offloader.unload_all()
        assert offloader.unload_count == 1

    def test_unregister_adapter(self):
        offloader = DiffusionLoRAOffloader(memory_budget_bytes=10000)
        offloader.register_adapter("lora_a", memory_bytes=1000)
        offloader.load_for_step(0, lora_ids=["lora_a"])
        offloader.unregister_adapter("lora_a")
        assert offloader.load_for_step(0, lora_ids=["lora_a"]) == []

    def test_unknown_adapter_load(self):
        offloader = DiffusionLoRAOffloader(memory_budget_bytes=10000)
        loaded = offloader.load_for_step(0, lora_ids=["nonexistent"])
        assert loaded == []

    def test_priority_ordering(self):
        offloader = DiffusionLoRAOffloader(memory_budget_bytes=50000)
        offloader.register_adapter("low", memory_bytes=1000, priority=1)
        offloader.register_adapter("high", memory_bytes=1000, priority=10)
        # Auto-load for step 0 — high priority should be first
        loaded = offloader.load_for_step(0)
        assert loaded[0] == "high"

    def test_idempotent_load(self):
        offloader = DiffusionLoRAOffloader(memory_budget_bytes=10000)
        offloader.register_adapter("lora_a", memory_bytes=1000)
        offloader.load_for_step(0, lora_ids=["lora_a"])
        offloader.load_for_step(0, lora_ids=["lora_a"])
        assert offloader.load_count == 1  # Not loaded twice
