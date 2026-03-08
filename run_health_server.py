from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.sirtrade.status import read_automation_status


class HealthHandler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        status = read_automation_status()
        if self.path == "/health":
            if status and status.get("ok"):
                self._send_json(200, {"status": "ok", "updated_at": status.get("updated_at")})
            else:
                self._send_json(503, {"status": "degraded", "updated_at": status.get("updated_at") if status else None})
            return

        if self.path == "/status":
            if status is None:
                self._send_json(404, {"status": "missing", "detail": "Automation status not found yet."})
            else:
                self._send_json(200, status)
            return

        self._send_json(404, {"error": "Not found"})

    def log_message(self, format: str, *args):
        return


def main() -> None:
    port = int(os.getenv("SIRTRADE_HEALTH_PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    print(f"Health server listening on 0.0.0.0:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
