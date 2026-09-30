# Todo API

`python server.py --port PORT` serves JSON over HTTP on 127.0.0.1:PORT. State is in memory.
All request and response bodies are JSON (`Content-Type: application/json`). Ids are integers
starting at 1 and never reused.

A todo object is `{"id": int, "title": str, "done": bool}`.

| Method and path | Behaviour |
|---|---|
| `GET /health` | 200 `{"status": "ok"}` |
| `POST /todos` | Body `{"title": str}` (non-empty string after stripping whitespace; `done` optional bool, default false). 201 with the created todo. |
| `GET /todos` | 200 with a JSON array of all todos in id order. Optional query `?done=true` or `?done=false` filters by state. |
| `GET /todos/<id>` | 200 with the todo. |
| `PATCH /todos/<id>` | Body may contain `title` (non-empty string) and/or `done` (bool). Unknown fields are ignored. 200 with the updated todo. |
| `DELETE /todos/<id>` | 204 with an empty body. |

Errors are JSON `{"error": "<message>"}`:

- 400: body is not valid JSON, is not an object, or a field has the wrong type or an invalid value (also `?done=` with any value other than true/false).
- 404: unknown todo id or unknown path. An `<id>` that is not an integer is also 404.
- 405: a known path with an unsupported method (`Allow` header not required).
