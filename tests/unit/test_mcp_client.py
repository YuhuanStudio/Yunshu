"""Tests for MCP Client Manager."""
import json

import pytest


class TestMCPServerConfig:
    def test_config_defaults(self):
        from yunshu_engine.mcp_client import MCPServerConfig
        config = MCPServerConfig(server_id="test")
        assert config.transport == "stdio"
        assert config.enabled is True
        assert config.command == ""


class TestMCPClientManager:
    def test_empty_manager(self):
        from yunshu_engine.mcp_client import MCPClientManager
        mgr = MCPClientManager()
        stats = mgr.get_stats()
        assert stats["connected_servers"] == 0
        assert stats["total_tools"] == 0

    def test_list_tools_empty(self):
        from yunshu_engine.mcp_client import MCPClientManager
        mgr = MCPClientManager()
        assert mgr.list_tools() == []

    def test_get_tools_as_openai_empty(self):
        from yunshu_engine.mcp_client import MCPClientManager
        mgr = MCPClientManager()
        assert mgr.get_tools_as_openai() == []

    def test_call_tool_not_found(self):
        from yunshu_engine.mcp_client import MCPClientManager
        mgr = MCPClientManager()
        with pytest.raises(KeyError):
            import asyncio
            asyncio.run(mgr.call_tool("nonexistent", {}))

    def test_parse_config_file(self, tmp_path):
        from yunshu_engine.mcp_client import MCPClientManager
        config = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    "transport": "stdio",
                },
                "search": {
                    "url": "http://localhost:3001",
                    "transport": "sse",
                },
            }
        }
        config_path = tmp_path / "mcp.json"
        config_path.write_text(json.dumps(config))

        mgr = MCPClientManager()
        configs = mgr._parse_config_file(config_path)
        assert len(configs) == 2
        assert configs[0].server_id == "filesystem"
        assert configs[0].transport == "stdio"
        assert configs[1].server_id == "search"
        assert configs[1].transport == "sse"

    def test_parse_nonexistent_config(self, tmp_path):
        from yunshu_engine.mcp_client import MCPClientManager
        mgr = MCPClientManager()
        configs = mgr._parse_config_file(tmp_path / "nonexistent.json")
        assert configs == []

    def test_load_config_env(self, monkeypatch):
        from yunshu_engine.mcp_client import MCPClientManager
        servers = json.dumps([
            {"id": "test", "command": "echo", "transport": "stdio", "enabled": False}
        ])
        monkeypatch.setenv("YUNSHU_MCP_SERVERS", servers)
        mgr = MCPClientManager()
        # This is async, just test the config parsing
        import asyncio
        result = asyncio.run(mgr.load_config())
        assert result == 0  # disabled, so 0 connected


class TestMCPServerConnection:
    def test_connection_initial_state(self):
        from yunshu_engine.mcp_client import MCPServerConfig, MCPServerConnection
        config = MCPServerConfig(server_id="test", command="echo")
        conn = MCPServerConnection(config)
        assert conn.is_connected is False
        assert conn.tools == []
