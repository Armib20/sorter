"""An in-process fake Viam machine (arm, gripper, camera) served over a grpclib
test channel, so robot.py is exercised through the real Viam SDK client stack
(RobotClient -> gRPC -> component services) with no viam-server.

    async with fake_machine(images=[...]) as (machine, calls):
        robot = await Robot(dry_run=False, connect=False).attach(machine)
        ...
        calls  # list of (component, method, args) the SDK delivered

Run directly for a self-test:  python tests/fake_machine.py
"""

import asyncio
import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402


def _make_fake(base, name, calls, overrides):
    async def stub(self, *a, **k):
        return None

    def recorder(method, impl):
        async def wrapped(self, *a, **k):
            k.pop("extra", None), k.pop("timeout", None)
            calls.append((name, method, a, k))
            return await impl(self, *a, **k)
        return wrapped

    body = {m: recorder(m, overrides.get(m, stub)) for m in base.__abstractmethods__}
    cls = type(f"Fake{base.__name__}", (base,), body)
    return cls(name)


@asynccontextmanager
async def fake_machine(images=None, arm=None, gripper=None, camera=None):
    import settings

    arm, gripper, camera = arm or settings.ARM_NAME, gripper or settings.GRIPPER_NAME, camera or settings.CAMERA_NAME
    from grpclib.testing import ChannelFor
    from viam.components.arm import Arm
    from viam.components.camera import Camera
    from viam.components.gripper import Gripper
    from viam.media.video import CameraMimeType, NamedImage
    from viam.proto.common import Pose, ResponseMetadata
    from viam.proto.component.arm import JointPositions
    from viam.resource.manager import ResourceManager
    from viam.resource.registry import Registry
    from viam.robot.client import RobotClient
    from viam.robot.service import RobotService

    calls = []
    frames = list(images or [Image.new("RGB", (64, 48), (0, 128, 0))])
    state = {"pose": Pose(x=1, y=2, z=3, o_z=-1), "joints": [0.0] * 6, "i": 0}

    async def get_end_position(self, *a, **k):
        return state["pose"]

    async def move_to_position(self, pose, *a, **k):
        state["pose"] = pose

    async def get_joint_positions(self, *a, **k):
        return JointPositions(values=state["joints"])

    async def move_to_joint_positions(self, positions, *a, **k):
        state["joints"] = list(positions.values)

    async def grab(self, *a, **k):
        return True

    async def get_images(self, *a, **k):
        img = frames[state["i"] % len(frames)]
        state["i"] += 1
        img = img if isinstance(img, Image.Image) else Image.open(img).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        depth = NamedImage("depth", b"\x00" * 16, CameraMimeType.from_string("image/vnd.viam.dep"))
        return [depth, NamedImage("color", buf.getvalue(), CameraMimeType.JPEG)], ResponseMetadata()

    comps = [
        _make_fake(Arm, arm, calls, dict(get_end_position=get_end_position, move_to_position=move_to_position,
                                         get_joint_positions=get_joint_positions,
                                         move_to_joint_positions=move_to_joint_positions)),
        _make_fake(Gripper, gripper, calls, dict(grab=grab)),
        _make_fake(Camera, camera, calls, dict(get_images=get_images)),
    ]
    manager = ResourceManager(comps)
    services = [RobotService(manager)] + [Registry.lookup_api(c.API).rpc_service(manager) for c in comps]
    async with ChannelFor(services) as channel:
        machine = await RobotClient.with_channel(channel, RobotClient.Options(refresh_interval=0, disable_sessions=True))
        try:
            yield machine, calls
        finally:
            await machine.close()


async def _selftest():
    import settings
    from robot import Robot

    settings.CAMERA_SOURCE = "color"
    settings.NUM_VIEWS = 2
    async with fake_machine() as (machine, calls):
        robot = await Robot(dry_run=False, connect=False).attach(machine)
        print(await robot.describe())
        assert await robot.pick("pick") is True
        imgs = await robot.capture()
        assert len(imgs) == 2 and imgs[0].size == (64, 48)
        await robot.place("reject")
        await robot.move_to({"joints": [10, 20, 30, 40, 50, 60]})
        methods = [(c, m) for c, m, *_ in calls]
        for c, m, a, k in calls:
            k = {kk: v for kk, v in k.items() if kk != "metadata"}
            print(f"  {c:8} {m:24} {' '.join(str(a).split())} {k or ''}"[:150])
        assert ("gripper", "grab") in methods and ("gripper", "open") in methods
        assert methods.count(("camera", "get_images")) == 2
        assert ("arm", "move_to_joint_positions") in methods
        last_pose = [a for c, m, a, k in calls if m == "move_to_position"][-1][0]
        reject = settings.POSES["reject"]
        assert abs(last_pose.z - (reject["z"] + settings.APPROACH_MM)) < 1e-6, "place should retreat above the bin"
    print("fake machine self-test ok")


if __name__ == "__main__":
    asyncio.run(_selftest())
