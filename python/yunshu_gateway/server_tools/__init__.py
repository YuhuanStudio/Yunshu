"""Server-side tools: web search, web fetch and the MCP connector, run inside the generation loop."""

from __future__ import annotations


def status() -> dict:
    """What this server can run for a request that declares server tools (for /v1/models and agents)."""
    from yunshu_engine import settings

    from .search import SETUP_HINT, get_provider

    prov = get_provider()
    web_search: dict = {
        "available": prov is not None,
        "provider": prov.name if prov else None,
    }
    if prov is None:
        web_search["setup"] = SETUP_HINT
    return {
        "web_search": web_search,
        "web_fetch": {
            "available": bool(settings.get("YUNSHU_WEB_FETCH")),
            "private_addresses": bool(settings.get("YUNSHU_WEB_FETCH_ALLOW_PRIVATE")),
        },
        "mcp_connector": {"available": bool(settings.get("YUNSHU_MCP_CONNECTOR"))},
    }
