"""Dashboard server: API, file serving, path-traversal safety, half-written lines, snapshot.

    python tests/test_dashboard.py
"""

import json
import re
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dashboard  # noqa: E402
from diagnostics import RunRecorder  # noqa: E402
from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402


def get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def main():
    root = Path(tempfile.mkdtemp())
    (root / "secret.txt").write_text("nope")
    rec = RunRecorder("offline", root=root)
    frame = Image.new("RGB", (64, 64), (40, 40, 40))
    amap = np.zeros((32, 32), np.float32)
    amap[10:20, 10:20] = 0.9
    for i in (1, 2):
        rec.item({"item": i, "score": 0.2 * i, "action": "box", "confidence": 1.0, "label": "good", "status": "ok",
                  "ms": {"total": 10.0}, "spans": []},
                 [{"frame": frame, "amap": amap, "info": {"score": 0.2 * i, "map": {}}}])
    rec.event("warning", "test warning")
    rec.close()
    run = rec.dir.name
    with open(rec.dir / "items.jsonl", "a") as f:
        f.write('{"item": 3, "half')  # simulate a line mid-write

    srv = dashboard.serve("127.0.0.1", 0, root)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        code, body = get(base, "/")
        assert code == 200 and b"Sorter dashboard" in body
        code, body = get(base, "/api/runs")
        runs = json.loads(body)
        assert code == 200 and runs[0]["id"] == run and runs[0]["state"] == "done", runs
        code, body = get(base, f"/api/run?id={run}&since=0")
        d = json.loads(body)
        assert code == 200 and len(d["items"]) == 2 and d["events"][0]["msg"] == "test warning", d
        code, body = get(base, f"/api/run?id={run}&since=1")
        assert [i["item"] for i in json.loads(body)["items"]] == [2]
        thumb = json.loads(get(base, f"/api/run?id={run}")[1])["items"][0]["artifacts"][0]["thumb"]
        code, body = get(base, f"/files/{run}/{thumb}")
        assert code == 200 and body[:2] == b"\xff\xd8", "thumb should be a JPEG"
        for bad in ["/files/../secret.txt", f"/files/{run}/../secret.txt", f"/files/{run}/%2e%2e/secret.txt",
                    "/files/..%2fsecret.txt", "/api/run?id=..", "/api/run?id=../" + run, "/files/" + run,
                    f"/files/{run}/items/../../secret.txt", "/nope"]:
            code, body = get(base, bad)
            assert code == 404 and b"nope" not in body, (bad, code, body[:80])
    finally:
        srv.shutdown()

    out = dashboard.snapshot(rec.dir)
    html = out.read_text()
    assert "/*SNAPSHOT*/null" not in html
    data = json.loads(re.search(r"const SNAP = (\{.*?\});\n", html, re.S).group(1))
    assert data["id"] == run and len(data["items"]) == 2
    print("dashboard ok:", base, out)


if __name__ == "__main__":
    main()
