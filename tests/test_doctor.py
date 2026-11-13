"""doctor.py robot checks against the fake Viam machine (normal frame, then a dark one).

    python tests/test_doctor.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageEnhance  # noqa: E402

import settings  # noqa: E402
from detector import Detector  # noqa: E402
from doctor import Report, check_robot  # noqa: E402
from robot import Robot  # noqa: E402
from tests.fake_machine import fake_machine  # noqa: E402


async def run(frame, det):
    rep, out = Report(), Path(tempfile.mkdtemp())
    async with fake_machine(images=[frame]) as (machine, calls):
        robot = await Robot(dry_run=True, connect=False).attach(machine)
        await check_robot(rep, robot, det, out)
        moved = [m for c, m, *_ in calls if m.startswith("move")]
    assert not moved, f"doctor must never move the arm: {moved}"
    assert (out / "camera.jpg").exists() and (out / "camera_overlay.jpg").exists()
    rows = {}
    for r in rep.rows:  # several rows can share a check name (one per drift warning)
        rows.setdefault(r["check"], []).append(r)
    return rows


async def main():
    det = Detector()
    good = Image.open(settings.TEST_IMAGES / "good" / "000.png").convert("RGB")
    rows = await run(good, det)
    for k in ("arm 'arm'", "gripper 'gripper'", "camera 'camera'", "arm pose", "camera frame", "frame vs training"):
        assert [r["status"] for r in rows[k]] == ["ok"], (k, rows[k])
    dark = ImageEnhance.Brightness(good).enhance(0.5)
    rows = await run(dark, det)
    drift = rows["frame vs training"]
    assert all(r["status"] == "warn" for r in drift) and any("darker" in r["detail"] for r in drift), drift
    print("\ndoctor robot checks ok")


if __name__ == "__main__":
    asyncio.run(main())
