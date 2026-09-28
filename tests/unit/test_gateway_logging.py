"""create_app gives Yunshu loggers an output unless the host already configured one."""

import logging

import yunshu_gateway.main as gw_main


def test_configures_yunshu_loggers_when_root_is_bare(monkeypatch):
    root = logging.getLogger()
    saved = root.handlers[:], root.level
    engine = logging.getLogger("yunshu_engine")
    saved_engine = engine.level
    root.handlers = []
    monkeypatch.setenv("YUNSHU_LOG_LEVEL", "DEBUG")
    try:
        gw_main._configure_logging()
        assert root.handlers, "root handler installed"
        assert root.level == logging.WARNING
        assert engine.level == logging.DEBUG
    finally:
        root.handlers, root.level = saved
        engine.setLevel(saved_engine)


def test_respects_existing_root_configuration():
    root = logging.getLogger()
    handler = logging.NullHandler()
    root.addHandler(handler)
    level = root.level
    try:
        gw_main._configure_logging()
        assert root.level == level
        assert handler in root.handlers
    finally:
        root.removeHandler(handler)
