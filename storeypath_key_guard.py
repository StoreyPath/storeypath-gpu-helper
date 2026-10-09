"""The key on every call but /health, for the vLLM build of StoreyPath's GPU helper.

vLLM's own --api-key guards only the paths under /v1, /v2, /inference and /cohere;
its other routes (/invocations, /tokenize, /pause, /metrics, /version and more)
answer without it. start.sh loads this in front of them all (vllm serve
--middleware storeypath_key_guard.KeyGuard) when the helper has a key: a call must
carry "Authorization: Bearer <key>" unless it is GET /health, which Studio and
Docker's health check ask without one, as with the llama.cpp build.

The key is read from VLLM_API_KEY, where start.sh puts it (vLLM's own check reads
it there too).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets

OPEN_PATHS = ("/health",)


class KeyGuard:
    """Pure ASGI middleware: 401 for an HTTP call without the key, a closed socket
    (policy violation) for a WebSocket one."""

    def __init__(self, app) -> None:
        self.app = app
        key = os.environ.get("VLLM_API_KEY", "")
        self.key_hash = hashlib.sha256(key.encode()).digest() if key else None

    def _has_key(self, scope) -> bool:
        for name, value in scope.get("headers") or ():
            if name.lower() == b"authorization":
                scheme, _, token = value.decode("latin-1").partition(" ")
                return scheme.lower() == "bearer" and secrets.compare_digest(
                    hashlib.sha256(token.strip().encode()).digest(), self.key_hash)
        return False

    async def __call__(self, scope, receive, send) -> None:
        kind = scope["type"]
        if self.key_hash is None or kind not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        path = scope["path"].removeprefix(scope.get("root_path", "")) or "/"
        if path.rstrip("/") in OPEN_PATHS or self._has_key(scope):
            return await self.app(scope, receive, send)
        if kind == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        body = json.dumps({"error": "Unauthorized"}).encode()
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode()),
                                (b"www-authenticate", b"Bearer")]})
        await send({"type": "http.response.body", "body": body})
