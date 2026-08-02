"""Local stand-in for the Google Apps Script endpoint.

Lets you click through the real study page and watch answers land in a
CSV, without setting up Google first. It speaks the same contract as the
Apps Script receiver, so if the page works against this it will work
against the deployed one.

Run it, point `RESULTS_ENDPOINT` in docs/index.html at it, open the page,
answer a few buildings, then look at the CSV.

    python study/local_receiver.py
    # -> listening on http://localhost:8766/collect
    # -> writing study/responses/local_test.csv

Columns match the Sheet exactly: session, server_time, client_time,
image, choice -- so `score.py --sheet` reads this file unchanged.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

OUT = Path("study/responses/local_test.csv")
PORT = 8766


class Handler(BaseHTTPRequestHandler):
    def _cors(self) -> None:
        # The real page posts text/plain to dodge a preflight, but a
        # browser may still send OPTIONS depending on how it is served.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")

    def do_OPTIONS(self) -> None:      # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self) -> None:         # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as exc:
            self.send_response(400)
            self._cors()
            self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "error": str(exc)}).encode())
            return

        OUT.parent.mkdir(parents=True, exist_ok=True)
        new = not OUT.exists()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = body.get("rows") or []
        with OUT.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["session", "server_time", "client_time",
                            "img", "choice"])
            for r in rows:
                ts = r.get("ts")
                client = (datetime.fromtimestamp(ts / 1000, timezone.utc)
                          .isoformat(timespec="seconds") if ts else "")
                w.writerow([body.get("session", ""), now, client,
                            r.get("img", ""), r.get("choice", "")])

        print(f"  + {len(rows):2d} row(s) from session "
              f"{body.get('session', '?')[:8]} -> {OUT}", flush=True)
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True, "n": len(rows)}).encode())

    def log_message(self, *_):         # silence the default access log
        pass


if __name__ == "__main__":
    print(f"listening on http://localhost:{PORT}/collect")
    print(f"writing     {OUT}")
    print("\nset this in docs/index.html:")
    print(f'  const RESULTS_ENDPOINT = "http://localhost:{PORT}/collect";\n')
    HTTPServer(("", PORT), Handler).serve_forever()
