"""Read-only capture proxy in front of the real DeepSeek endpoint.

The acceptance run must prove *for real* that the three looks came back from
DeepSeek (call count, raw request payloads, raw responses) and must be able to
report the provider's own token usage. Nothing in the app records outbound
provider traffic, so this process sits between the backend and
``https://api.deepseek.com`` and logs both directions verbatim.

It is an observability shim, not part of the product:

* the request body, auth header and content type are forwarded untouched;
* the upstream status, content type and body are returned untouched;
* every log write is wrapped so a logging failure can never break the call;
* on an upstream error it returns the upstream's own error body when it has
  one, and a plain 502 with the reason when the connection itself failed.

``--self-test`` runs the whole path offline against a canned upstream, so the
plumbing is proven before any billable request is made.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

_UPSTREAM = os.environ.get("CAPTURE_UPSTREAM", "")
_LOG = os.environ.get("CAPTURE_LOG", "")
_PORT = int(os.environ.get("CAPTURE_PORT", "8099"))


def _log_record(record: dict) -> None:
    if not _LOG:
        return
    try:
        with open(_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    except Exception:  # pragma: no cover - logging must never break the call
        pass


def _forward(headers, body: bytes) -> tuple[int, str, bytes]:
    outbound = {"Content-Type": headers.get("Content-Type", "application/json")}
    if headers.get("Authorization"):
        outbound["Authorization"] = headers["Authorization"]
    try:
        with httpx.Client(timeout=600.0, follow_redirects=True) as client:
            response = client.post(_UPSTREAM, content=body, headers=outbound)
        return (
            response.status_code,
            response.headers.get("content-type", "application/json"),
            response.content,
        )
    except Exception as exc:  # connection-level failure
        payload = json.dumps(
            {"error": {"message": f"capture proxy could not reach the upstream: {exc!r}"}}
        ).encode("utf-8")
        return 502, "application/json", payload


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # health probe
        if self.path.rstrip("/") in {"", "/__health"}:
            body = json.dumps(
                {"ok": True, "upstream": _UPSTREAM, "log": _LOG, "port": _PORT}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        record: dict = {"path": self.path, "bytes": len(body)}
        try:
            record["request"] = json.loads(body.decode("utf-8"))
        except Exception:
            record["request"] = body.decode("utf-8", "replace")
        status, content_type, payload = _forward(self.headers, body)
        record["status"] = status
        try:
            record["response"] = json.loads(payload.decode("utf-8"))
        except Exception:
            record["response"] = payload.decode("utf-8", "replace")
        _log_record(record)

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:  # keep the proxy's own output quiet
        return


def _serve(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _self_test() -> int:
    """Prove pass-through + logging against a canned upstream, fully offline."""
    canned = {
        "id": "fake-1",
        "choices": [{"message": {"role": "assistant", "content": '{"ok": true}'}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }

    class _Fake(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            blob = json.dumps(canned).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def log_message(self, *args) -> None:
            return

    fake = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    fake_port = fake.server_address[1]

    global _UPSTREAM, _LOG
    _UPSTREAM = f"http://127.0.0.1:{fake_port}/chat/completions"
    _LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proxy_selftest.jsonl")
    if os.path.exists(_LOG):
        os.remove(_LOG)
    proxy = _serve(_PORT)
    try:
        response = httpx.post(
            f"http://127.0.0.1:{_PORT}/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer test-key"},
            timeout=30.0,
        )
        assert response.status_code == 200, response.status_code
        assert response.json() == canned, response.text
        lines = [json.loads(line) for line in open(_LOG, encoding="utf-8")]
        assert len(lines) == 1, lines
        assert lines[0]["request"]["model"] == "x", lines
        assert lines[0]["response"]["usage"]["total_tokens"] == 10, lines
        health = httpx.get(f"http://127.0.0.1:{_PORT}/__health", timeout=10.0)
        assert health.json()["ok"] is True
        print("SELF-TEST PASS: pass-through, request logging, response logging, health")
        return 0
    finally:
        proxy.shutdown()
        fake.shutdown()


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    if not _UPSTREAM:
        raise SystemExit("CAPTURE_UPSTREAM is required")
    print(f"capture proxy on http://127.0.0.1:{_PORT} -> {_UPSTREAM}", flush=True)
    print(f"logging to {_LOG or '(disabled)'}", flush=True)
    _serve(_PORT).serve_forever()
