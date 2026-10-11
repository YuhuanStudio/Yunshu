import type zh from "../zh-TW/errors.ts";
import type { Shape } from "../../types.ts";

const errors: Shape<typeof zh> = {
  "status.401": "Authentication failed. Check the access token.",
  "status.403": "You do not have permission for this action.",
  "status.404": "Not found. This engine version may not offer this endpoint.",
  "status.409":
    "The action conflicts with the engine's current state. Check and try again.",
  "status.429": "Too many requests or the engine is busy. Try again shortly.",
  "status.rejected": "The engine rejected this request. Check your input.",
  "status.server":
    "Internal engine error (HTTP {status}). Check the engine log.",
  "status.other": "The engine returned HTTP {status}.",
  "failure.network":
    "Cannot reach the engine. Check the server address and that the engine is running.",
  "failure.timeout": "The engine timed out. Try again shortly.",
  "failure.json": "The engine returned content that is not valid JSON.",
  "failure.empty": "The engine returned an empty response.",
  "failure.field": "The engine returned data in an unexpected format.",
  "offline.connecting.title": "Connecting to the engine",
  "offline.connecting.short": "Connecting",
  "offline.connecting.hint": "Connecting to the local engine.",
  "offline.unauthorized.title": "A valid access token is required",
  "offline.unauthorized.short": "Unauthorized",
  "offline.unauthorized.hint":
    "The engine rejected this key. Update it in Settings.",
  "offline.server.title": "Internal engine error",
  "offline.server.short": "Engine error",
  "offline.server.hint":
    "The engine answered HTTP {status}. Check the engine log.",
  "offline.http.title": "The engine returned HTTP {status}",
  "offline.http.short": "HTTP {status}",
  "offline.http.hint":
    "The engine answered the status query with HTTP {status}.",
  "offline.unreachable.title": "Cannot reach the engine",
  "offline.unreachable.short": "Unreachable",
  "offline.unreachable.hint":
    "No response from the engine. Check that the Yunshu server is running.",
  "address.invalid": "Enter a valid engine address.",
  "address.protocol": "The engine address must use HTTP or HTTPS.",
  "address.credentials":
    "The engine address cannot contain a user name or password.",
  "address.query": "The engine address cannot contain a query or fragment.",
  "address.path": "Invalid API path.",
  "stream.interrupted": "Generation stopped: the engine reported an error.",
  "stream.failed": "Generation failed",
  "operation.partial":
    "The action partly completed: {warning}. Check the latest engine state.",
  "operation.warmupSkipped":
    "The model is loaded; this type has no text warm-up.",
  "operation.cancelSent": "Cancel request sent.",
  "operation.done": "Done.",
  "operation.timeout":
    "The server did not answer in time. The action may still be running; refresh the state.",
  "operation.saveAddressFailed":
    "Could not save the server address; this session still uses the new setting.",
};
export default errors;
