"""Compatibility alias: the shared network policy lives in :mod:`yunshu_engine.netguard`."""

from yunshu_engine.netguard import *  # noqa: F403
from yunshu_engine.netguard import (  # noqa: F401
    Target,
    UrlNotAllowedError,
    domain_matches,
    is_forbidden_ip,
    parse_url,
    resolve_target,
)
