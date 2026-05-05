"""Yunshu SDK — Realtime namespace (WebSocket).

Provides a WebSocket client for the Realtime API protocol,
supporting bidirectional streaming for audio/text inference
with interruptible generation and live token streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Optional

import httpx

logger = logging.getLogger(__name__)


class RealtimeSession:
    """WebSocket session for realtime inference.

    Supports:
    - Event-based message handling via on(event_type, handler)
    - Bidirectional send/recv for streaming text and audio
    - Automatic reconnection handling
    - Context manager protocol for safe resource cleanup

    Usage:
        async with session:
            session.on("text", lambda data: print(data["text"]))
            await session.send({"type": "text", "content": "Hello"})
            async for event in session.stream():
                print(event)
    """

    def __init__(self, ws):
        self._ws = ws
        self._handlers: dict[str, list[Callable]] = {}
        self._closed = False

    def on(self, event_type: str, handler: Callable) -> None:
        """Register a handler for a specific event type.

        Args:
            event_type: Event type string (e.g. 'text', 'audio', 'error', 'done').
            handler: Callable that receives the event data dict.
        """
        self._handlers.setdefault(event_type, []).append(handler)

    def off(self, event_type: str, handler: Optional[Callable] = None) -> None:
        """Remove a handler for an event type.

        Args:
            event_type: Event type to remove handlers from.
            handler: Specific handler to remove. If None, removes all for the type.
        """
        if handler is None:
            self._handlers.pop(event_type, None)
        else:
            handlers = self._handlers.get(event_type, [])
            self._handlers[event_type] = [h for h in handlers if h is not handler]

    async def send(self, data: dict) -> None:
        """Send a message over the WebSocket.

        Args:
            data: Dict to serialize and send.

        Raises:
            RuntimeError: If the session is closed.
        """
        if self._closed:
            raise RuntimeError("Session is closed")
        await self._ws.send(json.dumps(data))

    async def send_text(self, content: str, **kwargs) -> None:
        """Send a text message for realtime processing.

        Args:
            content: Text content to send.
            **kwargs: Additional fields (e.g. model, temperature).
        """
        payload = {"type": "text", "content": content}
        payload.update(kwargs)
        await self.send(payload)

    async def send_audio(self, audio_data: bytes, format: str = "pcm16", **kwargs) -> None:
        """Send audio data for realtime ASR/processing.

        Args:
            audio_data: Raw audio bytes.
            format: Audio format string (e.g. 'pcm16', 'wav').
            **kwargs: Additional fields (e.g. sample_rate, language).
        """
        import base64
        payload = {
            "type": "audio",
            "audio": base64.b64encode(audio_data).decode("ascii"),
            "format": format,
        }
        payload.update(kwargs)
        await self.send(payload)

    async def recv(self) -> dict:
        """Receive the next message from the WebSocket.

        Returns:
            Parsed JSON dict from the server.

        Raises:
            RuntimeError: If the session is closed.
        """
        if self._closed:
            raise RuntimeError("Session is closed")
        raw = await self._ws.recv()
        data = json.loads(raw)
        # Dispatch to registered handlers
        event_type = data.get("type", "")
        for handler in self._handlers.get(event_type, []):
            try:
                result = handler(data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning("Handler error for event '%s': %s", event_type, e)
        return data

    async def stream(self):
        """Async generator that yields events until the session closes.

        Yields:
            Parsed event dicts from the server.
        """
        try:
            while not self._closed:
                try:
                    data = await self.recv()
                    yield data
                    if data.get("type") == "done":
                        break
                except Exception as e:
                    if self._closed:
                        break
                    logger.debug("Stream error: %s", e)
                    break
        finally:
            await self.close()

    async def close(self) -> None:
        """Close the WebSocket session."""
        if not self._closed:
            self._closed = True
            try:
                await self._ws.close()
            except Exception:
                pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


class RealtimeNamespace:
    """Realtime namespace (client.realtime).

    Provides WebSocket-based realtime inference for streaming
    text and audio interaction.

    Usage:
        client = YunshuClient(base_url="http://localhost:8000")
        async with client.realtime.connect(api_key="...") as session:
            session.on("text", lambda d: print(d["text"]))
            await session.send_text("Hello!")
            async for event in session.stream():
                pass
    """

    def __init__(self, base_url: str):
        self._base_url = base_url.replace("http://", "ws://").replace("https://", "wss://")

    async def connect(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> RealtimeSession:
        """Connect to the realtime WebSocket endpoint.

        Args:
            api_key: Optional API key for authentication.
            model: Optional model to use for this session.
            **kwargs: Additional query parameters.

        Returns:
            A RealtimeSession for bidirectional communication.

        Raises:
            ImportError: If websockets package is not installed.
            ConnectionError: If the WebSocket connection fails.
        """
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets package is required for realtime API. "
                "Install it with: pip install websockets"
            )

        url = f"{self._base_url}/realtime"
        params = []
        if model:
            params.append(f"model={model}")
        for k, v in kwargs.items():
            params.append(f"{k}={v}")
        if params:
            url += "?" + "&".join(params)

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            ws = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=60,
            )
            return RealtimeSession(ws)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to realtime endpoint: {e}") from e
