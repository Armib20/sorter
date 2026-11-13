"""Read-only health check of the SO-101 leader's serial bus: N joint + gripper reads, count failures.

    python leader_check.py            # 150 reads at ~20 Hz
    python leader_check.py --gripper  # live leader-gripper % for 30 s (open it fully, then squeeze)
Healthy: 0 failures, steady read times. Failures / missing gripper = power or a loose servo cable.
"""
import asyncio
import statistics
import sys
import time

import viam.components.arm  # noqa: F401  register before dialing
import viam.components.gripper  # noqa: F401
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.robot.client import RobotClient

from robot import load_credentials


async def main(n=150):
    c = load_credentials()
    m = await RobotClient.at_address(c["address"], RobotClient.Options.with_api_key(
        api_key=c["api_key"], api_key_id=c["api_key_id"]))
    arm, grip = Arm.from_robot(m, "leader-arm"), Gripper.from_robot(m, "leader-gripper")
    ok, fails, times, errs, jumps = 0, 0, [], {}, 0
    prev = None
    try:
        for i in range(n):
            t0 = time.perf_counter()
            try:
                j = (await arm.get_joint_positions()).values
                await grip.is_moving()
                ok += 1
                times.append((time.perf_counter() - t0) * 1000)
                if prev and max(abs(a - b) for a, b in zip(j, prev)) > 20:  # >20 deg in 50 ms while still = bad read
                    jumps += 1
                prev = j
            except Exception as e:  # noqa: BLE001
                fails += 1
                k = str(e).split("\n")[0][:90]
                errs[k] = errs.get(k, 0) + 1
            await asyncio.sleep(0.05)
        print(f"reads ok {ok}/{n}, failed {fails}, implausible jumps {jumps}")
        if times:
            print(f"read time median {statistics.median(times):.1f} ms, max {max(times):.1f} ms")
        for k, v in sorted(errs.items(), key=lambda kv: -kv[1]):
            print(f"  {v:3d}x {k}")
        print("joints now:", [round(v, 1) for v in (await arm.get_joint_positions()).values])
    finally:
        await m.close()


async def watch_gripper(seconds=30):
    """What teleop sees: open_close opens the follower above 60 %, grabs below 40 %, holds in between."""
    c = load_credentials()
    m = await RobotClient.at_address(c["address"], RobotClient.Options.with_api_key(
        api_key=c["api_key"], api_key_id=c["api_key_id"]))
    lo, hi = 100.0, 0.0
    try:
        g = Gripper.from_robot(m, "leader-gripper")
        t_end = time.time() + seconds
        while time.time() < t_end:
            pct = (await g.do_command({"command": "get_position"}))["position_percentage"]
            lo, hi = min(lo, pct), max(hi, pct)
            zone = "OPEN  (>60)" if pct > 60 else "GRAB  (<40)" if pct < 40 else "hold  (40-60)"
            bar = "#" * int(pct / 2.5)
            print(f"\r{pct:6.1f}% {zone}  |{bar:<40}|  seen {lo:.0f}-{hi:.0f}%", end="", flush=True)
            await asyncio.sleep(0.2)
        print(f"\nrange seen: {lo:.1f}% .. {hi:.1f}%  -> needs to cross both 40 and 60 for open_close to toggle")
    finally:
        await m.close()


if __name__ == "__main__":
    if "--gripper" in sys.argv:
        asyncio.run(watch_gripper())
    else:
        asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 150))
