"""Sorter + real OpenVINO Detector against the fake Viam machine (real SDK client path).

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
from robot import Robot  # noqa: E402
from sorter import Sorter  # noqa: E402
from tests.fake_machine import fake_machine  # noqa: E402


async def main():
    test = settings.TEST_IMAGES
    frames = [test / "good" / "000.png", test / "broken_large" / "000.png"]
    log_path = Path(tempfile.mkdtemp()) / "fake.jsonl"
    det = Detector()
    async with fake_machine(images=frames) as (machine, calls):
        robot = await Robot(dry_run=False, connect=False).attach(machine)
        sorter = Sorter(robot, det, RuleDecider(), log_path)
        good = await sorter.process_item(views=1)
        bad = await sorter.process_item(views=1)
    assert good["action"] == "box" and bad["action"] == "reject", (good["action"], bad["action"])
    placed = [a[0] for c, m, a, k in calls if m == "move_to_position"]
    assert any(abs(p.y - settings.POSES["reject"]["y"]) < 1e-6 for p in placed), "never moved to the reject bin"
    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(lines) == 2 and set(lines[0]["ms"]) >= {"pick", "capture", "detect", "decide", "place", "total"}
    print(json.dumps(lines[1]["ms"]))
    print("sorter on fake machine ok:", [(r["score"], r["action"]) for r in lines])


if __name__ == "__main__":
    asyncio.run(main())
