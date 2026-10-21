"""Opt-in chain diagnostics for shortdecode_probe; no additional GPU readbacks."""

import hashlib
import logging
import os
import struct
import time


def install(proxy):
    """Wrap CopyDraft's existing host-side acceptance counters, including final close."""
    observe = proxy.observe
    choose = proxy.choose
    logger = logging.getLogger("yunshu_engine.shortdecode_chain")

    def traced_observe(self):
        seen = self._seen_rounds
        previous = self._previous
        start = getattr(self, "_shortdecode_start", None)
        observe(self)
        if self._seen_rounds <= seen or previous is None or start is None:
            return
        copied, proposed = previous
        accepted = int(self._draft.accept_lens[-1]) + 1
        at = getattr(self, "_shortdecode_at", 1)
        logger.debug(
            "DFlash chain round: at=%d kind=%s rows=%d accepted=%d match=%d ms=%.2f",
            at,
            "copy" if copied else "model",
            proposed + 1,
            accepted,
            self._copy.last_match,
            (time.perf_counter() - start) * 1000,
        )
        self._shortdecode_at = at + accepted

    def traced_choose(self, block, remaining):
        width = choose(self, block, remaining)
        self._shortdecode_start = time.perf_counter()
        return width

    proxy.observe = traced_observe
    proxy.choose = traced_choose


def install_receipts(runner, *, plain=False):
    """Hash raw generated IDs at the existing host publication boundary."""
    original = runner.iter_tokens
    logger = logging.getLogger("yunshu_engine.shortdecode_tokens")

    def capture(self, input_ids, **kwargs):
        digest = hashlib.sha256()
        count = 0
        if plain:
            kwargs["allow_draft"] = False
        try:
            for token in original(self, input_ids, **kwargs):
                digest.update(struct.pack("<I", int(token)))
                count += 1
                yield token
        finally:
            logger.debug(
                "Raw token receipt: count=%d sha256=%s plain=%s",
                count,
                digest.hexdigest(),
                plain,
            )

    runner.iter_tokens = capture


def main():
    from yunshu_cli import app
    from yunshu_engine.dflash_copy import CopyDraft
    from yunshu_engine.vlm_batch_runner import VLMBatchRunner

    install(CopyDraft)
    install_receipts(VLMBatchRunner, plain=os.environ.get("SHORTDECODE_PLAIN") == "1")
    app()


if __name__ == "__main__":
    main()
