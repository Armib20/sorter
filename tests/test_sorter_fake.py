"""Sorter + real OpenVINO Detector + RunRecorder against the fake Viam machine (real SDK client path).

    python tests/test_sorter_fake.py
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from decider import RuleDecider  # noqa: E402
from detector import Detector  # noqa: E402
from diagnostics import RunRecorder  # noqa: E402
from robot import Robot  # noqa: E402
from sorter import Sorter  # noqa: E402
from tests.fake_machine import fake_machine  # noqa: E402


async def main():
    test = settings.TEST_IMAGES
    frames = [test / "good" / "000.png", test / "broken_large" / "000.png"]
    det = Detector()
    recorder = RunRecorder("test", root=tempfile.mkdtemp())
    async with fake_machine(images=frames) as (machine, calls):
        robot = await Robot(dry_run=False, connect=False).attach(machine)
        sorter = Sorter(robot, det, RuleDecider(), recorder)
        good = await sorter.process_item(views=1)
        bad = await sorter.process_item(views=1)
    recorder.close()

    assert good["action"] == "box" and bad["action"] == "reject", (good["action"], bad["action"])
    placed = [a[0] for c, m, a, k in calls if m == "move_to_position"]
    assert any(abs(p.y - settings.POSES["reject"]["y"]) < 1e-6 for p in placed), "never moved to the reject bin"

    run = recorder.dir
    lines = [json.loads(line) for line in (run / "items.jsonl").read_text().splitlines()]
    assert len(lines) == 2 and set(lines[0]["ms"]) >= {"pick", "capture", "detect", "decide", "place", "total"}
    spans = lines[1]["spans"]
    names = [(s["depth"], s["name"]) for s in spans]
    assert (0, "pick") in names and (1, "move") in names and (1, "gripper.grab") in names, names
    cam = next(s for s in spans if s["name"] == "camera")
    assert cam["depth"] == 1 and cam["attrs"]["used"] == "color" and "rpc_ms" in cam["attrs"], cam
    assert any(s["name"] == "move" and s["attrs"]["target"] == "reject" for s in spans)
    assert lines[1]["diag"][0]["map"]["bbox"] is not None, "defect item should have an anomaly region"
    for a in lines[1]["artifacts"][0].values():
        assert (run / a).exists(), a
    status = json.loads((run / "status.json").read_text())
    assert status["state"] == "done", status
    meta = json.loads((run / "run.json").read_text())
    assert meta["verb"] == "test" and "settings" in meta and "versions" in meta

    for s in spans:
        print(f"  {'  ' * s['depth']}{s['name']:14} +{s['start_ms']:7.1f} {s['dur_ms']:7.1f} ms  {s.get('attrs', '')}"[:140])
    print("sorter on fake machine ok:", [(r["score"], r["action"]) for r in lines], run)


if __name__ == "__main__":
    asyncio.run(main())
