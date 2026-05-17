from __future__ import annotations
"""MCP Client Manager — connects to external MCP tool servers.

Complements the MCP Server (yunshu_gateway/routers/mcp.py) by adding
a Client role: lets the LLM call external MCP tool servers (filesystem,
search, databases, etc.) during generation.

Architecture:
  MCPClientManager — manages connections to multiple MCP servers
  MCPServerConnection — single server connection with tool discovery

Follows oMLX's MCP client pattern:
- Load config from mcp.json or environment
- Discover tools from connected servers
- Convert MCP tools ↔ OpenAI function format
- Execute tool calls and return results to the LLM
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict = field(default_factory=dict)
    server_id: str = ""


@dataclass
class MCPServerConfig:
    server_id: str
    transport: str = "stdio"  # stdio, sse, streamable_http
    command: str = ""
    args: list[str] = field(default_factory=list)
    url: str = ""
    headers: dict = field(default_factory=dict)
    enabled: bool = True


class MCPServerConnection:
    """Represents a connection to a single MCP tool server.

    Manages tool discovery and execution for one server.
    """

    _next_id = 1  # Class-level monotonic ID counter for JSON-RPC requests

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.tools: list[MCPTool] = []
        self._connected = False
        self._process = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        if self.config.transport == "stdio":
            return await self._connect_stdio()
        elif self.config.transport in ("sse", "streamable_http"):
            return await self._connect_http()
        return False

    async def _connect_stdio(self) -> bool:
        """Connect to a stdio-based MCP server."""
        try:
            self._process = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._connected = True
            await self._discover_tools()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.config.server_id}: {e}", exc_info=True)
            return False

    async def _connect_http(self) -> bool:
        """Connect to an HTTP-based MCP server."""
        self._connected = True
        await self._discover_tools()
        return True

    async def _discover_tools(self) -> None:
        """Discover available tools from the server."""
        result = await self._send_request("tools/list", {})
        if result and "tools" in result:
            for tool_data in result["tools"]:
                self.tools.append(MCPTool(
                    name=tool_data.get("name", ""),
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {}),
                    server_id=self.config.server_id,
                ))
            logger.info(f"Discovered {len(self.tools)} tools from {self.config.server_id}")

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Execute a tool call on this server."""
        return await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })

    async def _send_request(self, method: str, params: dict) -> Any:
        """Send a JSON-RPC request to the MCP server."""
        if self.config.transport == "stdio" and self._process:
            return await self._send_stdio(method, params)
        elif self.config.transport in ("sse", "streamable_http"):
            return await self._send_http(method, params)
        return None

    @classmethod
    def _next_request_id(cls) -> int:
        """Generate a unique monotonic ID for JSON-RPC requests."""
        req_id = cls._next_id
        cls._next_id += 1
        return req_id

    async def _send_stdio(self, method: str, params: dict) -> Any:
        """Send request via stdio transport."""
        if not self._process or not self._process.stdin:
            return None
        req_id = self._next_request_id()
        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        msg = json.dumps(request) + "\n"
        self._process.stdin.write(msg.encode())
        await self._process.stdin.drain()

        if self._process.stdout:
            # Read lines until we get a response with matching ID
            # (skip server notifications which have no "id" field)
            deadline = asyncio.get_event_loop().time() + 30.0
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    logger.warning(f"MCP stdio timeout waiting for response to {method}")
                    return None
                response_line = await asyncio.wait_for(
                    self._process.stdout.readline(), timeout=remaining
                )
                if not response_line:
                    return None
                try:
                    response = json.loads(response_line.decode())
                except json.JSONDecodeError:
                    logger.debug(f"MCP stdio: skipping non-JSON line")
                    continue
                # Skip notifications (no "id" field) — they're informational
                if "id" not in response:
                    logger.debug(f"MCP stdio: skipping notification: {response.get('method', '?')}")
                    continue
                # Check for error response
                if "error" in response:
                    err = response["error"]
                    logger.error(f"MCP server error: {err.get('code')} {err.get('message')}")
                    return None
                return response.get("result")
        return None

    async def _send_http(self, method: str, params: dict) -> Any:
        """Send request via HTTP transport."""
        import aiohttp
        url = self.config.url
        if not url:
            return None
        req_id = self._next_request_id()
        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=request,
                    headers=self.config.headers,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if "error" in data:
                            err = data["error"]
                            logger.error(f"MCP HTTP error: {err.get('code')} {err.get('message')}")
                            return None
                        return data.get("result")
        except Exception as e:
            logger.error(f"HTTP MCP request failed: {e}", exc_info=True)
        return None

    async def disconnect(self) -> None:
        if self._process:
            try:
                self._process.terminate()
                await self._process.wait()
            except Exception:
                logger.debug("MCP server process terminate failed", exc_info=True)
            self._process = None
        self._connected = False
        self.tools.clear()


class MCPClientManager:
    """Manages connections to multiple MCP tool servers.

    Provides:
    - Tool discovery from all connected servers
    - Tool call routing to the correct server
    - MCP tool → OpenAI function format conversion
    - Config loading from mcp.json or environment
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerConnection] = {}
        self._tool_index: dict[str, MCPServerConnection] = {}

    async def load_config(self, config_path: str | None = None) -> int:
        """Load MCP server configs from mcp.json or environment.

        Returns number of servers loaded.
        """
        configs = []

        # Try mcp.json file
        if config_path:
            path = Path(config_path)
            if path.exists():
                configs = self._parse_config_file(path)

        # Environment: YUNSHU_MCP_SERVERS=json_array
        env_servers = os.environ.get("YUNSHU_MCP_SERVERS")
        if env_servers and not configs:
            try:
                servers = json.loads(env_servers)
                for s in servers:
                    configs.append(MCPServerConfig(
                        server_id=s.get("id", s.get("name", "")),
                        transport=s.get("transport", "stdio"),
                        command=s.get("command", ""),
                        args=s.get("args", []),
                        url=s.get("url", ""),
                        headers=s.get("headers", {}),
                        enabled=s.get("enabled", True),
                    ))
            except json.JSONDecodeError:
                logger.error("Failed to parse YUNSHU_MCP_SERVERS env var")

        # Register and connect
        connected = 0
        for config in configs:
            if not config.enabled:
                continue
            conn = MCPServerConnection(config)
            if await conn.connect():
                self._servers[config.server_id] = conn
                for tool in conn.tools:
                    self._tool_index[tool.name] = conn
                connected += 1

        return connected

    def _parse_config_file(self, path: Path) -> list[MCPServerConfig]:
        try:
            with open(path) as f:
                data = json.load(f)
            configs = []
            servers = data.get("mcpServers", data.get("servers", []))
            for sid, sdata in servers.items() if isinstance(servers, dict) else enumerate(servers):
                if isinstance(sid, int):
                    sid = sdata.get("id", f"server-{sid}")
                configs.append(MCPServerConfig(
                    server_id=sid,
                    transport=sdata.get("transport", "stdio"),
                    command=sdata.get("command", ""),
                    args=sdata.get("args", []),
                    url=sdata.get("url", ""),
                    headers=sdata.get("headers", {}),
                    enabled=sdata.get("enabled", True),
                ))
            return configs
        except Exception as e:
            logger.error(f"Failed to parse MCP config from {path}: {e}", exc_info=True)
            return []

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Call a tool on the appropriate server."""
        conn = self._tool_index.get(tool_name)
        if conn is None:
            raise KeyError(f"Tool '{tool_name}' not found in any connected MCP server")
        return await conn.call_tool(tool_name, arguments)

    async def call_tools_parallel(self, calls: list[dict]) -> list[Any]:
        """Execute multiple tool calls in parallel.

        Args:
            calls: List of {"name": str, "arguments": dict} dicts.

        Returns:
            List of results in the same order as input calls.
        """
        tasks = [self.call_tool(c["name"], c.get("arguments", {})) for c in calls]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def get_tools_as_openai(self) -> list[dict]:
        """Get all tools in OpenAI function format."""
        result = []
        for tool_name, conn in self._tool_index.items():
            tool = next((t for t in conn.tools if t.name == tool_name), None)
            if tool:
                result.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                })
        return result

    def list_tools(self) -> list[dict]:
        """List all discovered tools."""
        tools = []
        for conn in self._servers.values():
            for tool in conn.tools:
                tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "server_id": tool.server_id,
                    "input_schema": tool.input_schema,
                })
        return tools

    def get_stats(self) -> dict:
        return {
            "connected_servers": len(self._servers),
            "total_tools": len(self._tool_index),
            "servers": {
                sid: {
                    "connected": conn.is_connected,
                    "tools": len(conn.tools),
                }
                for sid, conn in self._servers.items()
            },
        }

    async def disconnect_all(self) -> None:
        for conn in self._servers.values():
            await conn.disconnect()
        self._servers.clear()
        self._tool_index.clear()


# Singleton
_instance: MCPClientManager | None = None


def get_mcp_client_manager() -> MCPClientManager | None:
    return _instance


async def init_mcp_client(config_path: str | None = None) -> MCPClientManager:
    global _instance
    _instance = MCPClientManager()
    await _instance.load_config(config_path)
    return _instance
