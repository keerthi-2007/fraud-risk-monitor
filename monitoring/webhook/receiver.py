import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

LOG_PATH = Path("/app/notifications/alerts.jsonl")


class AlertHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/alerts":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                self.send_error(413)
                return

            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or not isinstance(
                payload.get("alerts"), list
            ):
                self.send_error(400)
                return

            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

            with LOG_PATH.open("a", encoding="utf-8") as log:
                for alert in payload["alerts"]:
                    record = {
                        "received_at": datetime.now(timezone.utc).isoformat(),
                        "status": alert.get("status"),
                        "alertname": alert.get("labels", {}).get("alertname"),
                        "severity": alert.get("labels", {}).get("severity"),
                        "summary": alert.get("annotations", {}).get("summary"),
                    }
                    log.write(json.dumps(record) + "\n")

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

        except (ValueError, TypeError, KeyError):
            self.send_error(400)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 8085), AlertHandler).serve_forever()
