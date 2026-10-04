"""Keep canonical APC prefill boundaries without retaining auxiliary caches."""

from types import SimpleNamespace

from ..apc_manager import _Coordinator


class AuxiliaryPrefillPolicy(SimpleNamespace):
    """Snapshot only numerical planning metadata; never touch primary storage."""

    def __init__(self, manager):
        super().__init__(
            _generation=0,
            **{
                name: getattr(manager, name)
                for name in (
                    "exact_cache_guard_tokens",
                    "checkpoint_interval_tokens",
                    "keep_interval_checkpoint",
                    "block_size",
                    "exact_cache_min_tokens",
                    "head_marker",
                )
            },
        )

    def head_boundary(self, token_ids):
        if self.head_marker is None:
            return 0
        a, b = self.head_marker
        return next(
            (
                i
                for i in range(1, len(token_ids) - 1)
                if token_ids[i] == a and token_ids[i + 1] == b
            ),
            0,
        )

    def note_head(self, token_ids):
        pass

    def begin_request(self):
        pass

    def release(self, blocks):
        pass

    def coordinator(self, model):
        return _AuxiliaryCoordinator(self, model)


class _AuxiliaryCoordinator(_Coordinator):
    def prepare_prefill(self, *args, **kwargs):
        pass

    def observe_cache(self, *args, **kwargs):
        pass

    def lookup(self, *args, **kwargs):
        return None

    def store_checkpoint(self, *args, **kwargs):
        # Mark this boundary consumed, without allocating a snapshot.
        return True

    def commit(self, *args, **kwargs):
        return True
