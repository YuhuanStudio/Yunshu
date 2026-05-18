from __future__ import annotations
"""Diffusion pipeline infrastructure — distributed scheduling, LoRA offloading, coordination.

Provides infrastructure for production-grade diffusion inference:

1. DiffusionScheduler — step scheduling with DDIM/DPM++/Euler support,
   CFG, noise schedules, step-range control for img2img/inpainting.
2. DiffusionLoRAOffloader — swaps LoRA adapters in/out of GPU memory
   during diffusion steps with priority-based scheduling and memory budgeting.
3. DistributedDiffusionCoordinator — splits diffusion work across nodes
   with synchronization points, fault tolerance, and latent exchange.

Integration:
  - Used by image_engine / image_pipeline for image generation
  - LoRA offloader integrates with lora_manager
  - Distributed coordinator integrates with yunshu_mesh for multi-node

Reference:
  - diffusers: DDIMScheduler, DPMSolverMultistepScheduler
  - vLLM: LoRA request scheduling
"""

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


# ── Noise schedule ──


class NoiseScheduleType(str, Enum):
    """Supported noise schedule types."""
    LINEAR = "linear"
    SCALED_LINEAR = "scaled_linear"
    COSINE = "cosine"
    SQRT_LINEAR = "sqrt_linear"


# ── Scheduler type ──


class SchedulerType(str, Enum):
    """Supported diffusion scheduler algorithms."""
    DDIM = "ddim"
    DPM_PLUS_PLUS = "dpm_plus_plus"
    EULER = "euler"
    EULER_ANCESTRAL = "euler_ancestral"
    LMS = "lms"


# ── Diffusion step result ──


@dataclass
class DiffusionStep:
    """A single diffusion step."""
    step_index: int
    timestep: int
    sigma: float
    cfg_scale: float
    latent_state: Any = None  # Will hold mx.array in production
    is_last: bool = False


# ── DiffusionScheduler ──


class DiffusionScheduler:
    """Manages diffusion step scheduling with multiple algorithm support.

    Supports:
    - DDIM, DPM++, Euler, Euler Ancestral, LMS algorithms
    - CFG (classifier-free guidance) via cfg_scale
    - Step-range control for partial denoising (inpainting, img2img)
    - Configurable noise schedules (linear, scaled_linear, cosine)
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 28,
        scheduler_type: SchedulerType = SchedulerType.DDIM,
        noise_schedule: NoiseScheduleType = NoiseScheduleType.SCALED_LINEAR,
        cfg_scale: float = 7.5,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        start_step: int = 0,
        end_step: Optional[int] = None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.scheduler_type = scheduler_type
        self.noise_schedule = noise_schedule
        self.cfg_scale = cfg_scale
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.start_step = start_step
        self.end_step = end_step if end_step is not None else num_inference_steps

        # Validate
        if num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {num_inference_steps}")
        if num_inference_steps > num_train_timesteps:
            raise ValueError(
                f"num_inference_steps ({num_inference_steps}) cannot exceed "
                f"num_train_timesteps ({num_train_timesteps})"
            )
        if not (0 <= start_step < self.end_step):
            raise ValueError(
                f"start_step ({start_step}) must be in [0, {self.end_step})"
            )

        self._betas = self._compute_betas()
        self._alphas = [1.0 - b for b in self._betas]
        self._alphas_cumprod = self._cumprod(self._alphas)
        self._timesteps = self._compute_timesteps()
        self._sigmas = self._compute_sigmas()
        self._current_step = start_step

    @property
    def timesteps(self) -> list[int]:
        return list(self._timesteps)

    @property
    def sigmas(self) -> list[float]:
        return list(self._sigmas)

    @property
    def total_steps(self) -> int:
        """Total number of steps to execute (end - start)."""
        return self.end_step - self.start_step

    @property
    def current_step(self) -> int:
        return self._current_step

    def _compute_betas(self) -> list[float]:
        """Compute beta schedule based on noise_schedule type."""
        if self.noise_schedule == NoiseScheduleType.LINEAR:
            return [self.beta_start + i * (self.beta_end - self.beta_start) /
                    (self.num_train_timesteps - 1)
                    for i in range(self.num_train_timesteps)]
        elif self.noise_schedule == NoiseScheduleType.SCALED_LINEAR:
            start_sqrt = math.sqrt(self.beta_start)
            end_sqrt = math.sqrt(self.beta_end)
            return [(start_sqrt + i * (end_sqrt - start_sqrt) /
                     (self.num_train_timesteps - 1)) ** 2
                    for i in range(self.num_train_timesteps)]
        elif self.noise_schedule == NoiseScheduleType.COSINE:
            return self._cosine_betas()
        elif self.noise_schedule == NoiseScheduleType.SQRT_LINEAR:
            return [self.beta_start + math.sqrt(
                i / (self.num_train_timesteps - 1)) *
                    (self.beta_end - self.beta_start)
                    for i in range(self.num_train_timesteps)]
        return [self.beta_start] * self.num_train_timesteps

    def _cosine_betas(self, s: float = 0.008) -> list[float]:
        """Cosine schedule (Improved DDPM)."""
        steps = self.num_train_timesteps + 1
        t = [i / steps for i in range(steps)]
        alphas_cumprod = [math.cos((t_val + s) / (1 + s) * math.pi / 2) ** 2
                          for t_val in t]
        alphas_cumprod = [a / alphas_cumprod[0] for a in alphas_cumprod]
        betas = []
        for i in range(1, len(alphas_cumprod)):
            beta = min(1 - alphas_cumprod[i] / alphas_cumprod[i - 1], 0.999)
            betas.append(beta)
        return betas[:self.num_train_timesteps]

    @staticmethod
    def _cumprod(values: list[float]) -> list[float]:
        result = []
        cumul = 1.0
        for v in values:
            cumul *= v
            result.append(cumul)
        return result

    def _compute_timesteps(self) -> list[int]:
        """Compute inference timesteps by evenly spacing within train timesteps.

        Uses rounding to ensure exactly num_inference_steps timesteps that
        span the full training range, avoiding the truncation error that
        floor-division-based spacing introduces.
        """
        if self.num_inference_steps <= 1:
            # Single step: use the middle of the training range
            return [self.num_train_timesteps // 2]

        timesteps = [
            int(round(i * (self.num_train_timesteps - 1) / (self.num_inference_steps - 1)))
            for i in range(self.num_inference_steps)
        ]
        # Deduplicate (can happen with very few inference steps) and sort
        timesteps = sorted(set(timesteps))
        # If deduplication reduced the count, backfill with nearby values.
        # Guard: break if no new unique value was added in the last iteration
        # to prevent an infinite loop when all gaps are too small to split.
        max_attempts = self.num_inference_steps  # bound iterations
        attempts = 0
        while len(timesteps) < self.num_inference_steps and attempts < max_attempts:
            attempts += 1
            prev_len = len(timesteps)
            # Insert midpoints between consecutive timesteps
            gaps = [(timesteps[i + 1] - timesteps[i], i) for i in range(len(timesteps) - 1)]
            if not gaps:
                break
            gaps.sort(reverse=True)
            gap_size, gap_idx = gaps[0]
            if gap_size <= 1:
                # All gaps are 0 or 1 — cannot add more unique timesteps
                break
            mid = timesteps[gap_idx] + gap_size // 2
            timesteps.append(mid)
            timesteps = sorted(set(timesteps))
            if len(timesteps) == prev_len:
                # No new value added — stop to avoid infinite loop
                break
        return list(reversed(timesteps))

    def _compute_sigmas(self) -> list[float]:
        """Compute sigma values from alphas_cumprod."""
        sigmas = []
        for ts in self._timesteps:
            if ts < len(self._alphas_cumprod) and self._alphas_cumprod[ts] > 0:
                sigmas.append(math.sqrt((1 - self._alphas_cumprod[ts]) /
                                        self._alphas_cumprod[ts]))
            else:
                sigmas.append(1.0)
        return sigmas

    def get_step(self, step_index: int) -> DiffusionStep:
        """Get the diffusion step descriptor for a given index."""
        if step_index < 0 or step_index >= len(self._timesteps):
            raise IndexError(
                f"step_index {step_index} out of range [0, {len(self._timesteps)})")
        return DiffusionStep(
            step_index=step_index,
            timestep=self._timesteps[step_index],
            sigma=self._sigmas[step_index],
            cfg_scale=self.cfg_scale,
            is_last=(step_index == len(self._timesteps) - 1),
        )

    def steps(self) -> list[DiffusionStep]:
        """Return all diffusion steps in execution order."""
        return [self.get_step(i) for i in range(len(self._timesteps))]

    def iter_steps(self):
        """Iterate through steps, updating internal state."""
        for i in range(self.start_step, self.end_step):
            if i >= len(self._timesteps):
                break
            self._current_step = i
            yield self.get_step(i)
        self._current_step = self.end_step

    def reset(self) -> None:
        """Reset scheduler to beginning."""
        self._current_step = self.start_step

    def add_noise(self, sample: Any, noise: Any, timestep: int) -> Any:
        """Add noise to a sample at a given timestep (forward process).

        Returns (sqrt_alpha_prod, sqrt_one_minus_alpha_prod) scaling factors
        that the caller applies as: noisy_sample = sqrt_alpha * sample +
        sqrt_one_minus_alpha * noise.  For scheduling purposes, this returns
        the formula parameters rather than operating on arrays directly.
        """
        if timestep < 0 or timestep >= len(self._alphas_cumprod):
            # Out-of-range timestep: treat as pure noise
            return 0.0, 1.0
        alpha_prod = self._alphas_cumprod[timestep]
        sqrt_alpha = math.sqrt(alpha_prod)
        sqrt_one_minus_alpha = math.sqrt(1 - alpha_prod)
        return sqrt_alpha, sqrt_one_minus_alpha

    def scale_model_input(self, sample: Any, step_index: int) -> Any:
        """Scale the model input for the given step (scheduler-specific).

        Returns scaling parameters. In production, operates on mx.arrays.
        """
        if step_index >= len(self._sigmas):
            return sample
        sigma = self._sigmas[step_index]

        if self.scheduler_type == SchedulerType.DDIM:
            # DDIM: no scaling
            return sample
        elif self.scheduler_type == SchedulerType.EULER:
            # Euler: c_skip, c_out, c_in
            return sigma / (sigma + 1)
        return sample


# ── LoRA offloading ──


@dataclass
class LoRAAdapter:
    """Metadata for a LoRA adapter."""
    lora_id: str
    priority: int = 0  # Higher = more important
    memory_bytes: int = 0
    loaded: bool = False
    assigned_steps: Optional[range] = None


@dataclass
class MemoryBudget:
    """GPU memory budget tracking for LoRA offloading."""
    total_bytes: int = 0
    used_bytes: int = 0
    peak_bytes: int = 0

    @property
    def available_bytes(self) -> int:
        return max(0, self.total_bytes - self.used_bytes)

    @property
    def utilization(self) -> float:
        return self.used_bytes / self.total_bytes if self.total_bytes > 0 else 0.0

    def allocate(self, bytes_count: int) -> bool:
        """Try to allocate memory, returns True if successful."""
        if bytes_count > self.available_bytes:
            return False
        self.used_bytes += bytes_count
        self.peak_bytes = max(self.peak_bytes, self.used_bytes)
        return True

    def free(self, bytes_count: int) -> None:
        """Free allocated memory."""
        self.used_bytes = max(0, self.used_bytes - bytes_count)


class DiffusionLoRAOffloader:
    """Swaps LoRA adapters in/out of GPU memory during diffusion steps.

    Only loads the LoRA needed for the current step. Supports multiple LoRA
    with priority-based scheduling and memory budget tracking.

    Strategy:
    - Higher priority adapters stay loaded longer
    - Lower priority adapters are evicted first when memory is tight
    - Adapters assigned to specific step ranges are loaded/unloaded on demand
    """

    def __init__(self, memory_budget_bytes: int = 0):
        self._adapters: dict[str, LoRAAdapter] = {}
        self._memory = MemoryBudget(total_bytes=memory_budget_bytes)
        self._load_count = 0
        self._unload_count = 0
        self._load_history: list[tuple[int, str, str]] = []  # (step, lora_id, action)

    @property
    def memory(self) -> MemoryBudget:
        return self._memory

    @property
    def loaded_adapters(self) -> list[str]:
        return [aid for aid, a in self._adapters.items() if a.loaded]

    @property
    def load_count(self) -> int:
        return self._load_count

    @property
    def unload_count(self) -> int:
        return self._unload_count

    def register_adapter(self, lora_id: str, memory_bytes: int = 0,
                         priority: int = 0,
                         assigned_steps: Optional[range] = None) -> None:
        """Register a LoRA adapter with the offloader."""
        self._adapters[lora_id] = LoRAAdapter(
            lora_id=lora_id,
            priority=priority,
            memory_bytes=memory_bytes,
            assigned_steps=assigned_steps,
        )

    def unregister_adapter(self, lora_id: str) -> None:
        """Remove a LoRA adapter, unloading it if loaded."""
        adapter = self._adapters.get(lora_id)
        if adapter and adapter.loaded:
            self._unload(lora_id)
        self._adapters.pop(lora_id, None)

    def load_for_step(self, step: int, lora_ids: Optional[list[str]] = None) -> list[str]:
        """Load the appropriate LoRA adapter(s) for a given step.

        Args:
            step: The current diffusion step index.
            lora_ids: Explicit list of adapters to load. If None, loads
                      adapters assigned to this step.

        Returns:
            List of adapter IDs that were loaded.
        """
        loaded = []
        target_ids = lora_ids if lora_ids is not None else self._adapters_for_step(step)

        for lora_id in target_ids:
            adapter = self._adapters.get(lora_id)
            if adapter is None:
                logger.warning(f"Unknown LoRA adapter: {lora_id}")
                continue
            if adapter.loaded:
                loaded.append(lora_id)
                continue

            # Check if we need to evict to make room
            if self._memory.total_bytes > 0:
                if not self._evict_for(adapter.memory_bytes, lora_id, step):
                    logger.warning(f"Cannot load {lora_id}: insufficient memory after eviction")
                    continue

            self._load(lora_id)
            loaded.append(lora_id)

        return loaded

    def unload_after_step(self, step: int) -> list[str]:
        """Unload adapters that are no longer needed after a step.

        Returns list of unloaded adapter IDs.
        """
        unloaded = []
        for lora_id, adapter in list(self._adapters.items()):
            if not adapter.loaded:
                continue
            # Unload if not assigned to any future step
            if adapter.assigned_steps is not None:
                if step >= adapter.assigned_steps.stop - 1:
                    self._unload(lora_id)
                    unloaded.append(lora_id)
        return unloaded

    def unload_all(self) -> list[str]:
        """Unload all loaded adapters."""
        unloaded = []
        for lora_id in list(self._adapters.keys()):
            if self._adapters[lora_id].loaded:
                self._unload(lora_id)
                unloaded.append(lora_id)
        return unloaded

    def _adapters_for_step(self, step: int) -> list[str]:
        """Find adapters assigned to a given step, sorted by priority."""
        result = []
        for lora_id, adapter in self._adapters.items():
            if adapter.assigned_steps is not None:
                if step in adapter.assigned_steps:
                    result.append(adapter)
            else:
                # No step range means always applicable
                result.append(adapter)
        result.sort(key=lambda a: -a.priority)  # Higher priority first
        return [a.lora_id for a in result]

    def _load(self, lora_id: str) -> None:
        adapter = self._adapters[lora_id]
        adapter.loaded = True
        self._memory.allocate(adapter.memory_bytes)
        self._load_count += 1

    def _unload(self, lora_id: str) -> None:
        adapter = self._adapters[lora_id]
        adapter.loaded = False
        self._memory.free(adapter.memory_bytes)
        self._unload_count += 1

    def _evict_for(self, needed_bytes: int, exclude_id: str, step: int = 0) -> bool:
        """Evict loaded adapters to free memory, excluding a specific adapter."""
        if needed_bytes <= self._memory.available_bytes:
            return True

        # Sort loaded adapters by priority (lowest first) for eviction
        candidates = [(a.priority, lid, a) for lid, a in self._adapters.items()
                      if a.loaded and lid != exclude_id]
        candidates.sort()  # Lowest priority first

        freed = 0
        for priority, lid, adapter in candidates:
            if freed >= needed_bytes - self._memory.available_bytes:
                break
            self._unload(lid)
            self._load_history.append((step, lid, "evict"))
            freed += adapter.memory_bytes

        return self._memory.available_bytes >= needed_bytes


# ── Distributed coordination ──


@dataclass
class NodeAssignment:
    """Step assignment for a node.

    steps can be a range (for contiguous assignment) or a list[int]
    (for round-robin or non-contiguous assignment).
    """
    node_id: int
    steps: Union[range, list[int]]
    is_primary: bool = False

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def first_step(self) -> int:
        return self.steps[0] if self.steps else -1

    @property
    def last_step(self) -> int:
        return self.steps[-1] if self.steps else -1


@dataclass
class SyncCheckpoint:
    """Checkpoint for fault-tolerant latent exchange."""
    step: int
    node_id: int
    timestamp: float
    latent_hash: str = ""


class StepAssignmentStrategy(str, Enum):
    """Strategy for distributing steps across nodes."""
    CONTIGUOUS = "contiguous"  # Each node gets a contiguous block
    ROUND_ROBIN = "round_robin"  # Steps alternate between nodes
    DYNAMIC = "dynamic"  # Nodes pull steps from a queue


class DistributedDiffusionCoordinator:
    """Coordinates diffusion work across multiple nodes.

    Splits diffusion steps across nodes with:
    - Multiple assignment strategies (contiguous, round-robin, dynamic)
    - Synchronization points for latent exchange between nodes
    - Fault tolerance with checkpoint-based restart
    """

    def __init__(
        self,
        total_steps: int = 28,
        num_nodes: int = 1,
        strategy: StepAssignmentStrategy = StepAssignmentStrategy.CONTIGUOUS,
        sync_interval: int = 4,
    ):
        if num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}")

        self.total_steps = total_steps
        self.num_nodes = min(num_nodes, total_steps)
        self.strategy = strategy
        self.sync_interval = sync_interval
        self._assignments: dict[int, NodeAssignment] = {}
        self._checkpoints: list[SyncCheckpoint] = []
        self._sync_points: list[int] = []
        self._assign_steps()

    @property
    def assignments(self) -> dict[int, NodeAssignment]:
        return dict(self._assignments)

    @property
    def sync_points(self) -> list[int]:
        return list(self._sync_points)

    @property
    def checkpoints(self) -> list[SyncCheckpoint]:
        return list(self._checkpoints)

    def _assign_steps(self) -> None:
        """Assign steps to nodes based on the selected strategy."""
        if self.strategy == StepAssignmentStrategy.CONTIGUOUS:
            self._assign_contiguous()
        elif self.strategy == StepAssignmentStrategy.ROUND_ROBIN:
            self._assign_round_robin()
        elif self.strategy == StepAssignmentStrategy.DYNAMIC:
            self._assign_contiguous()  # Dynamic starts as contiguous, reassigned on-the-fly

        self._compute_sync_points()

    def _assign_contiguous(self) -> None:
        """Contiguous assignment: each node gets a contiguous block of steps."""
        steps_per_node = self.total_steps // self.num_nodes
        remainder = self.total_steps % self.num_nodes

        start = 0
        for node_id in range(self.num_nodes):
            count = steps_per_node + (1 if node_id < remainder else 0)
            self._assignments[node_id] = NodeAssignment(
                node_id=node_id,
                steps=range(start, start + count),
                is_primary=(node_id == 0),
            )
            start += count

    def _assign_round_robin(self) -> None:
        """Round-robin assignment: steps alternate between nodes."""
        node_steps: dict[int, list[int]] = {i: [] for i in range(self.num_nodes)}
        for step_idx in range(self.total_steps):
            node_id = step_idx % self.num_nodes
            node_steps[node_id].append(step_idx)

        for node_id, steps_list in node_steps.items():
            if steps_list:
                self._assignments[node_id] = NodeAssignment(
                    node_id=node_id,
                    steps=steps_list,  # Store actual step list, not a fake range
                    is_primary=(node_id == 0),
                )

    def _compute_sync_points(self) -> None:
        """Compute synchronization points based on sync_interval and node boundaries."""
        sync = set()

        # Node boundary sync points — use first/last steps of each assignment
        for node_id, assignment in self._assignments.items():
            if assignment.steps:
                sync.add(assignment.first_step)
                sync.add(assignment.last_step + 1)  # stop boundary

        # Interval-based sync points
        for i in range(0, self.total_steps + 1, self.sync_interval):
            sync.add(i)

        self._sync_points = sorted(sync)

    def get_node_for_step(self, step: int) -> int:
        """Get the node ID responsible for a given step."""
        for node_id, assignment in self._assignments.items():
            if step in assignment.steps:
                return node_id
        return 0

    def assign_steps(self, num_nodes: int, total_steps: int) -> dict[int, NodeAssignment]:
        """Re-assign steps with new parameters."""
        self.num_nodes = min(num_nodes, total_steps)
        self.total_steps = total_steps
        self._assignments.clear()
        self._sync_points.clear()
        self._assign_steps()
        return self.assignments

    def sync_latents(self, source_node: int, target_node: int,
                     step: int, latent_data: Any = None) -> SyncCheckpoint:
        """Record a latent synchronization between nodes.

        In production, this would transfer actual latent data via mesh.
        For scheduling, it records the sync checkpoint for fault tolerance.
        """
        checkpoint = SyncCheckpoint(
            step=step,
            node_id=source_node,
            timestamp=time.monotonic(),
            latent_hash=str(hash(latent_data)) if latent_data is not None else "",
        )
        self._checkpoints.append(checkpoint)
        return checkpoint

    def get_last_checkpoint(self) -> Optional[SyncCheckpoint]:
        """Get the most recent checkpoint for fault-tolerant restart."""
        if not self._checkpoints:
            return None
        return self._checkpoints[-1]

    def get_restart_assignment(self) -> Optional[NodeAssignment]:
        """Get the assignment for restarting from the last checkpoint.

        Returns the assignment for the node that was running at the last
        checkpoint, adjusted to skip already-completed steps.
        """
        last = self.get_last_checkpoint()
        if last is None:
            return None

        assignment = self._assignments.get(last.node_id)
        if assignment is None:
            return None

        # For list-based assignments, filter remaining steps
        steps = assignment.steps
        if isinstance(steps, list):
            remaining = [s for s in steps if s > last.step]
            if not remaining:
                return None
            return NodeAssignment(
                node_id=assignment.node_id,
                steps=remaining,
                is_primary=assignment.is_primary,
            )

        # For range-based assignments
        remaining_start = min(last.step + 1, steps.stop)
        if remaining_start >= steps.stop:
            return None

        return NodeAssignment(
            node_id=assignment.node_id,
            steps=range(remaining_start, steps.stop),
            is_primary=assignment.is_primary,
        )

    def get_progress(self) -> dict:
        """Get overall progress information."""
        completed_checkpoints = len(self._checkpoints)
        return {
            "total_steps": self.total_steps,
            "num_nodes": self.num_nodes,
            "strategy": self.strategy.value,
            "sync_points_count": len(self._sync_points),
            "checkpoints_count": completed_checkpoints,
            "progress_pct": (completed_checkpoints / self.total_steps * 100
                             if self.total_steps > 0 else 0.0),
        }
