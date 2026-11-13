"""Send the follower arm and/or the leader arm to their saved home poses.

Uses the machine's own `arm-position-saver` switches (erh:vmodutils): position 2 = "go to
saved pose", position 1 = "save current pose". One arm at a time, follower first; aborts
the leader move if the follower doesn't end up where expected; Ctrl-C / any error calls
stop() on the arms.

    python home.py              # follower (home-pose) then leader (leader-home)
    python home.py --follower   # only the robot arm
    python home.py --leader     # only the leader
    python home.py --save follower   # store the follower's CURRENT pose as its home (switch position 1)
"""

import argparse
import asyncio
import json
import math
import sys
import time

import viam.components.arm  # noqa: F401  register APIs before dialing
import viam.components.switch  # noqa: F401
from viam.app.viam_client import ViamClient
from viam.components.arm import Arm
from viam.components.switch import Switch
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions

from robot import load_credentials

PAIRS = {"follower": ("home-pose", "arm"), "leader": ("leader-home", "leader-arm")}
TOLERANCE_DEG = 3.0


async def saved_joints_deg(creds, switch_name):
    """Read the switch's saved joints (radians in config) through the app API, in degrees."""
    vc = await ViamClient.create_from_dial_options(DialOptions.with_api_key(creds["api_key"], creds["api_key_id"]))
    try:
        part_id = creds.get("part_id")
        if not part_id:
            return None
        part = await vc.app_client.get_robot_part(part_id)
        for comp in json.loads(part.robot_config_json).get("components", []):
            if comp["name"] == switch_name:
                j = comp.get("attributes", {}).get("joints")
                return [math.degrees(v) for v in j] if j else None
    finally:
        vc.close()


async def go_home(machine, which, expected=None, timeout=60):
    switch_name, arm_name = PAIRS[which]
    arm, sw = Arm.from_robot(machine, arm_name), Switch.from_robot(machine, switch_name)
    before = [round(v, 1) for v in (await arm.get_joint_positions()).values]
    print(f"[{which}] {arm_name} before: {before}")
    if expected:
        print(f"[{which}] target ({switch_name}): {[round(v, 1) for v in expected]}")
    t0 = time.perf_counter()
    try:
        await asyncio.wait_for(sw.set_position(2), timeout)   # "go to saved pose"
        while await arm.is_moving():                         # in case set_position returns early
            if time.perf_counter() - t0 > timeout:
                raise TimeoutError("arm still moving")
            await asyncio.sleep(0.2)
    except BaseException:
        await arm.stop()
        print(f"[{which}] STOPPED {arm_name}")
        raise
    after = [round(v, 1) for v in (await arm.get_joint_positions()).values]
    print(f"[{which}] {arm_name} after:  {after}   ({time.perf_counter() - t0:.1f} s)")
    if expected:
        err = max(abs(a - e) for a, e in zip(after, expected))
        print(f"[{which}] max joint error vs saved home: {err:.1f}°")
        return err <= TOLERANCE_DEG
    return True


async def main(args):
    creds = load_credentials()
    machine = await RobotClient.at_address(creds["address"], RobotClient.Options.with_api_key(
        api_key=creds["api_key"], api_key_id=creds["api_key_id"]))
    try:
        if args.save:
            switch_name, arm_name = PAIRS[args.save]
            await Switch.from_robot(machine, switch_name).set_position(1)  # "update config" with current pose
            print(f"saved current {arm_name} pose into {switch_name}")
            return
        order = [w for w in ("follower", "leader") if getattr(args, w)] or ["follower", "leader"]
        for which in order:
            expected = await saved_joints_deg(creds, PAIRS[which][0])
            if not await go_home(machine, which, expected):
                print(f"[{which}] did not reach home within {TOLERANCE_DEG}°: not continuing")
                sys.exit(1)
        print("done: both arms homed" if len(order) == 2 else "done")
    finally:
        await machine.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--follower", action="store_true")
    ap.add_argument("--leader", action="store_true")
    ap.add_argument("--save", choices=list(PAIRS), default=None)
    asyncio.run(main(ap.parse_args()))
