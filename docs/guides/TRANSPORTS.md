# Transports beyond SSE

Yunshu serves the same app over several transports. Every one dispatches into the same handlers,
so auth, middleware, request ids, prefix cache and cancellation behave identically.

| Transport | Endpoint | Status |
|---|---|---|
| HTTP + SSE | `POST /v1/chat/completions`, `/v1/responses`, `/v1/messages` | stable |
| Text WebSocket, multiplexed | `WS /v1/stream` | stable, Yunshu protocol (below) |
| Responses WebSocket mode | `WS /v1/responses` | stable, OpenAI's protocol |
| Realtime WebSocket | `WS /v1/realtime` (GA schema; beta with `OpenAI-Beta: realtime=v1`), `/realtime` (beta) | stable |
| Realtime WebRTC | `POST /v1/realtime/calls` | optional `yunshu[webrtc]`, see below |
| Unix domain socket | `yunshu serve --uds PATH` | stable |
| HTTP/2 (h2c) | not offered, see below | -- |

## 1. Research: what the official specs say (checked 2026-09-29)

**OpenAI Responses WebSocket mode** (`developers.openai.com/api/docs/guides/websocket-mode`) exists.
Connect to `/v1/responses` as a WebSocket (SDK: `client.responses.connect()`); send
`{"type":"response.create", "model", "input", "store", "previous_response_id", "generate", "stream_id"}`;
the server emits the same `response.*` events as the SSE stream, raw, plus `error`. Connections last up
to 60 min, up to 16 in-flight responses and 32 named `stream_id`s per connection; requests with the same
`stream_id` run FIFO and never overlap, different ids run concurrently. Continuation is
`previous_response_id` with only the new input items. **We implement exactly this** at `WS /v1/responses`.

**Anthropic Messages**: SSE (`stream: true`) is the only streaming transport. There is no official
WebSocket mode, so nothing to match; Messages is reachable through our own `/v1/stream` (`api: "messages"`).

**OpenAI Realtime** (`/api/docs/guides/realtime-websocket`, `-conversations`, `-webrtc`):
- WebSocket `wss://.../v1/realtime?model=M`, `Authorization: Bearer`, browsers use the subprotocols
  `realtime` and `openai-insecure-api-key.<key>`. Events are JSON text frames.
- GA client events: `session.update`, `conversation.item.create|truncate|delete`, `response.create|cancel`,
  `input_audio_buffer.append|commit|clear`, `output_audio_buffer.clear`.
- GA server events: `session.created|updated`, `conversation.item.added|done`, `response.created|done`,
  `response.output_item.added|done`, `response.content_part.added|done`, `response.output_text.delta|done`,
  `response.output_audio.delta|done`, `response.output_audio_transcript.delta|done`,
  `response.function_call_arguments.delta|done`, `input_audio_buffer.speech_started|speech_stopped|committed`,
  `rate_limits.updated`, `error`.
- GA session: `type:"realtime"`, `output_modalities`, `audio.input.{format,turn_detection,transcription}`,
  `audio.output.{format,voice}`, `instructions`, `tools`, `tool_choice`. Audio formats `audio/pcm` (24 kHz,
  mono, 16-bit), `audio/pcmu`, `audio/pcma`. Turn detection `server_vad` / `semantic_vad` / null.
- WebRTC: `POST /v1/realtime/calls` with `Content-Type: application/sdp` (offer in, answer out), an
  `oai-events` data channel carrying the same JSON events, audio on media tracks, ephemeral keys for browsers.

**Other local servers.** vLLM: `/v1/realtime` WebSocket for streaming speech-to-text only (16 kHz PCM16,
`transcription.delta/done`); no text WebSocket. llama.cpp `llama-server`, Ollama (NDJSON over HTTP) and
LM Studio (its own SDK socket, not an API): text is SSE/NDJSON only, no multiplexing, no mid-stream
control besides dropping the connection. None serves Realtime with speech-to-speech, WebRTC, or a Unix socket
with the OpenAI/Anthropic surface.

## 2. `WS /v1/stream` (Yunshu protocol)

Used where no official WebSocket mode exists (chat.completions, completions, messages) and for the
multiplexing extras. One connection carries many concurrent requests. Server events are byte-for-byte the
objects the SSE stream carries (`data:` payloads), wrapped with the client's request id.

Handshake: same auth as HTTP (`Authorization: Bearer`, `x-api-key`, `?token=`), checked before the upgrade
(HTTP 403 on failure). The first server message is `session.created`.

Client to server (JSON text frames):

| Message | Meaning |
|---|---|
| `{"type":"request","id":"a","api":"chat.completions\|completions\|responses\|messages","body":{...},"stream_id":"s"?}` | start a request; `body` is the normal HTTP body (`stream` is forced on, `stream_options.include_usage` defaulted for chat) |
| `{"type":"cancel","id":"a"}` | abort; generation stops (same path as an SSE client disconnect); ends with `done(cancelled)` |
| `{"type":"stop","id":"a"}` | stop on demand; ends with `done(stopped)` |
| `{"type":"update","id":"a","max_tokens":N}` | cap total streamed deltas (about one per token) at N; ends with `done(max_tokens)` |
| `{"type":"ping","t":..}` / `{"type":"pong"}` | liveness both ways |

Server to client:

| Message | Meaning |
|---|---|
| `{"type":"session.created","protocol":"yunshu.stream","apis":[...],"limits":{...}}` | limits: `max_inflight`, `max_stream_ids`, `send_queue`, `ping_interval_s` |
| `{"type":"event","id":"a","event":"<sse event name or null>","data":{...}}` | one SSE event. For chat, `data` is the `chat.completion.chunk`; for responses/messages the named events |
| `{"type":"done","id":"a","reason":"completed\|cancelled\|stopped\|max_tokens\|error","stats":{"ttft_ms","duration_ms","events","deltas"}}` | terminal message of every request |
| `{"type":"error","id"?,"status":404,"error":{message,type,...}}` | request-scoped (with `id`) or connection-scoped error. Upstream HTTP errors keep their status/body |
| `{"type":"ping","t":...}` | heartbeat every `YUNSHU_WS_PING_INTERVAL` s (default 15) |
| `{"type":"updated","id","max_tokens"}` | ack of `update` |
| `{"type":"progress","id",...}` | queue / prefill progress: the fields of the SSE `: yunshu-progress` comment ([API_EXTENSIONS.md](API_EXTENSIONS.md)): `phase`, `percent`, `processed_tokens`, `tokens_per_second`, `eta_s`, `queue_position`, ... |
| `{"type":"stats","id",...}` | per-response stats: the fields of the `x_yunshu` object / `: yunshu-stats` comment (`ttft_ms`, `decode_tps`, cache hits, ...). The chat usage chunk still carries `x_yunshu` inside its `event` data |

Any progress / stats SSE events the HTTP handlers emit flow through as ordinary `event` messages, with no
change to the transport.

Rules:
- `id` is the client request id. When it is header-safe (1-128 of `A-Za-z0-9._:-`) it is also the HTTP `X-Request-Id`, so
  `GET/DELETE /v1/requests/{id}`, logs and error hints use the same id; otherwise the handler generates one, reported as
  `request_id` in `done`. `cancel` takes the same path as an SSE client disconnect (the handler's disconnect abort).
  Progress and stats events are not emitted in the Responses-WS dialect (OpenAI's protocol has no slot for them).
- At most `YUNSHU_WS_MAX_INFLIGHT` (16) concurrent requests; more get `429 too_many_requests`. Reusing an
  active id is `409`. Requests sharing a `stream_id` run FIFO; at most 32 stream ids.
- **Backpressure**: the per-connection outbound queue is bounded (`YUNSHU_WS_SEND_QUEUE`, 256 events) and the
  bridge between the app and the socket is a bounded queue too, so a slow reader blocks the app's `send()`
  and generation pauses instead of buffering. Heartbeats never queue behind a stalled reader.
- Closing the socket cancels every in-flight request.

### Responses WebSocket mode (`WS /v1/responses`)

OpenAI's protocol exactly: send `{"type":"response.create", ...body..., "stream_id"?}`, receive raw
`response.*` events. `stream_id` lanes as above. Extension: `{"type":"response.cancel","stream_id"|"id"}`.
Verified with the official `openai` SDK (`client.responses.connect()`).

### Client

```python
from yunshu_client import YunshuStream          # python/yunshu_client
async with YunshuStream("ws://127.0.0.1:8000/v1/stream", token=None) as conn:
    async for m in conn.chat({"model": "m", "messages": [...]}, id="a", with_done=True):
        ...                                     # events, then the done message
    await conn.cancel("a")                      # from any task
```

`examples/ws_stream.py` streams two chats on one socket and cancels one.

## 3. Realtime conformance

`/v1/realtime` implements the OpenAI Realtime protocol in two dialects, chosen at the handshake:
GA (default, what `client.realtime.connect()` speaks) and beta (`OpenAI-Beta: realtime=v1`, and the legacy
`/realtime` path). The session engine is shared; `yunshu_gateway/realtime_ga.py` translates event names
and the session shape at the socket boundary. Also implemented: `?model=`, the `realtime` and
`openai-insecure-api-key.<key>` subprotocols, auth and Origin checks before the upgrade (HTTP 403).

Verified with the official SDK by `scripts/dev/realtime_conformance.py` (unit test with a fake engine;
real-model smoke in `scripts/dev/transport_smoke.py`). Known differences from api.openai.com:

- `semantic_vad` is served as `server_vad` (`eagerness` is ignored).
- `turn_detection.idle_timeout_ms`: that long after a completed response (plus the playback time of its
  audio) without user speech, the server sends `input_audio_buffer.timeout_triggered`, commits the empty
  buffer as a silent user turn and answers.
- `audio.input.noise_reduction` (`near_field` / `far_field`; beta `input_audio_noise_reduction`): a causal
  CPU high-pass plus noise-floor expander (`realtime_dsp.py`) on the input audio before VAD and ASR.
  It is a light filter, not a neural denoiser.
- `rate_limits.updated` follows `response.created`. A local server has no quota: `requests` reports
  `YUNSHU_RATE_LIMIT_RPM` (unlimited = 2^31-1 when 0), `tokens` the model's context window.
- `output_audio_buffer.started` / `.stopped` / `.cleared` (GA dialect only; OpenAI sends them on WebRTC / SIP)
  bracket the audio of a response, `cleared` when it was cut by a cancel or barge-in; the client event
  `output_audio_buffer.clear` cancels the speaking response.
- `response.create.input` replaces the conversation as the context of that response (items or
  `item_reference`s; `[]` clears it); combine with `conversation: "none"` to keep it out of the history.
- Spoken replies carry `output_audio` content parts with the `transcript` (`audio` in the beta dialect).
- Pass-through of unsupported session fields is silent (no `invalid_request_error`).

### WebRTC (optional)

Install `uv pip install 'yunshu[webrtc]'`. `POST /v1/realtime/calls` accepts an SDP
body (`application/sdp`) or multipart `sdp` plus JSON `session`, matching
`client.realtime.calls.create`. It returns 201, an SDP answer and a `Location` call ID.
An `oai-events` data channel shares the GA Realtime session. Incoming audio is
resampled on CPU to mono PCM16 at 24 kHz; outgoing PCM travels on the RTP track,
with a bounded two-second buffer. Audio format changes on the data channel are
normalized to PCM24k; RTP negotiates the wire codec. Sessions close on peer failure,
client channel close, server shutdown, or after one hour; offers without an opened
channel expire after 60 seconds. At most 16 calls remain live.

aiortc 1.15 uses BSD-3-Clause; its PyAV arm64 wheel is about 18 MB. It is an opt-in
extra, so text-only installs do not acquire the codec stack. There are no public
STUN/TURN servers: loopback/LAN peers must have directly reachable ICE candidates.
Missing the extra returns 503 with `WS /v1/realtime` as the user-visible alternative.
WebSocket audio uses `input_audio_buffer.append` and `response.output_audio.delta`.
Ephemeral client secrets also authenticate this signalling endpoint. SIP call
management and a monitoring sideband WebSocket are not provided.

CPU evidence: SDK SDP create through ASGI/TestClient, two real aiortc peers exchanging
session events and audio frames, PCM framing, queue bounds and cleanup in
`tests/unit/test_respfeat.py`. M5 tiny real-server probe: `yv --suite respfeat`.

## 4. Unix domain socket

`yunshu serve --uds /path/y.sock` (setting `YUNSHU_UDS`) serves the identical app on a socket instead of TCP
(`--host/--port` are ignored, a stale socket file is removed). Owner-only file permissions are the
access control, which makes it the safest local transport: no port, no network exposure.

```bash
curl --unix-socket /path/y.sock http://localhost/v1/models
```
```python
import httpx, openai
client = openai.OpenAI(base_url="http://yunshu/v1", api_key="x",
                       http_client=httpx.Client(transport=httpx.HTTPTransport(uds="/path/y.sock")))
```
`YunshuStream(url, uds=path)` does the same for the WebSocket. (The OpenAI SDK's own realtime/responses
WebSocket clients open TCP connections; use a TCP port for them.)

## 5. HTTP/2

Not offered. uvicorn speaks HTTP/1.1 only; h2c needs hypercorn (or granian), i.e. a second server stack with
its own lifespan, WebSocket and disconnect semantics, which the disconnect-driven cancellation and the
multiplexed WebSocket already cover for the use cases HTTP/2 would (many concurrent streams on one
connection, no head-of-line blocking between requests). Revisit if a client needs h2c specifically.
