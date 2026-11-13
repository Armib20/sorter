"""Thin async wrapper over the Viam Python SDK: connect, pick, place, capture.

    async with Robot(dry_run=False) as robot:      # reads api-key.json
        await robot.pick("pick")
        images = await robot.capture(views=2)      # list of PIL images
        await robot.place("reject")

Modes
  connected + dry_run=False   real motion and camera
  connected + dry_run=True    real camera, motion is only logged (safe perception test)
  not connected (no address)  fully offline: motion logged, capture() serves images
                              from `image_source` (e.g. a folder of test images)

api-key.json (gitignored):
  {"address": "my-machine-main.xxxx.viam.cloud", "api_key": "...", "api_key_id": "..."}
(Viam app -> your machine -> CONNECT -> API keys / Code sample.)

CLI:
    python robot.py --dry-run          # connect, list resources, print current pose
    python robot.py --snap             # save one frame to /tmp/snap.png
"""

import argparse
import asyncio
import io
import itertools
import json
import logging
import time
from pathlib import Path

from PIL import Image

import settings
from diagnostics import maybe_span

log = logging.getLogger("robot")


def resolve_pose(pose):
    """A pose name (key of settings.POSES) or a pose dict -> pose dict."""
    if isinstance(pose, str):
        if pose not in settings.POSES:
            raise KeyError(f"unknown pose {pose!r}; known: {sorted(settings.POSES)}")
        return dict(settings.POSES[pose])
    return dict(pose)


def offset_pose(pose, offset):
    """Add an offset like {"z": 80} or {"theta": 90} to a pose. For joint poses,
    "theta" rotates the last joint (the wrist)."""
    pose = dict(pose)
    if "joints" in pose:
        joints = list(pose["joints"])
        if "theta" in offset:
            joints[-1] += offset["theta"]
        for i, d in enumerate(offset.get("joints", [])):
            joints[i] += d
        pose["joints"] = joints
        return pose
    for k, v in offset.items():
        pose[k] = pose.get(k, 0) + v
    return pose


def load_credentials(path=None, address=None):
    path = Path(path or settings.API_KEY_FILE)
    creds = json.loads(path.read_text()) if path.exists() else {}
    creds["address"] = address or settings.ROBOT_ADDRESS or creds.get("address", "")
    return creds


class Robot:
    def __init__(self, dry_run=True, address=None, key_file=None, connect=True, image_source=None):
        self.dry_run = dry_run
        self.creds = load_credentials(key_file, address) if connect else {}
        self.machine = None
        self.arm = self.gripper = self.camera = self.motion = None
        # offline frames: an iterable of paths / PIL images, cycled
        self._frames = itertools.cycle(list(image_source)) if image_source else None
        self.holding = False
        self.trace = None  # diagnostics.Trace for the current item (set by Sorter); spans per call

    # ---------------------------------------------------------------- lifecycle
    @property
    def connected(self):
        return self.machine is not None

    async def connect(self, timeout=20):
        if not self.creds.get("address"):
            log.info("no robot address: running fully offline (dry run)")
            self.dry_run = True
            return self
        # Import component APIs BEFORE dialing: RobotClient only builds clients for resource
        # types registered at connect time (otherwise from_robot raises ResourceNotFoundError).
        import viam.components.arm  # noqa: F401
        import viam.components.camera  # noqa: F401
        import viam.components.gripper  # noqa: F401
        import viam.services.motion  # noqa: F401
        from viam.robot.client import RobotClient

        if not (self.creds.get("api_key") and self.creds.get("api_key_id")):
            raise SystemExit(f"{settings.API_KEY_FILE} needs api_key and api_key_id")
        opts = RobotClient.Options.with_api_key(api_key=self.creds["api_key"], api_key_id=self.creds["api_key_id"])
        log.info("connecting to %s ...", self.creds["address"])
        machine = await asyncio.wait_for(RobotClient.at_address(self.creds["address"], opts), timeout)
        await self.attach(machine)
        return self

    async def attach(self, machine):
        """Bind components on an already-connected RobotClient (used by connect() and tests)."""
        from viam.components.arm import Arm
        from viam.components.camera import Camera
        from viam.components.gripper import Gripper

        self.machine = machine
        names = {(r.subtype, r.name) for r in machine.resource_names}

        def get(cls, name):
            return cls.from_robot(machine, name) if (cls.API.resource_subtype, name) in names else None

        self.arm = get(Arm, settings.ARM_NAME)
        self.gripper = get(Gripper, settings.GRIPPER_NAME)
        self.camera = get(Camera, settings.CAMERA_NAME)
        if settings.USE_MOTION_SERVICE:
            from viam.services.motion import MotionClient
            self.motion = MotionClient.from_robot(machine, settings.MOTION_NAME)
        missing = [n for n, c in [(settings.ARM_NAME, self.arm), (settings.GRIPPER_NAME, self.gripper),
                                  (settings.CAMERA_NAME, self.camera)] if c is None]
        if missing:
            log.warning("not found on machine: %s (available: %s)", missing, sorted(f"{t}:{n}" for t, n in names))
        return self

    async def close(self):
        if self.machine is not None:
            await self.machine.close()
            self.machine = None

    async def __aenter__(self):
        return await self.connect()

    async def __aexit__(self, *exc):
        await self.close()

    # ----------------------------------------------------------------- motion
    def _span(self, name, **attrs):
        return maybe_span(self.trace, name, **attrs)

    async def move_to(self, pose, label=None):
        label = label or (pose if isinstance(pose, str) else "custom")
        pose = resolve_pose(pose)
        mode = "joints" if "joints" in pose else ("motion" if self.motion is not None else "cartesian")
        with self._span("move", target=label, mode=mode, dry_run=self.dry_run or self.arm is None):
            if self.dry_run or self.arm is None:
                log.info("DRY-RUN move_to %s %s", label, pose)
                return
            from viam.proto.common import Pose, PoseInFrame
            from viam.proto.component.arm import JointPositions

            if mode == "joints":
                await self.arm.move_to_joint_positions(JointPositions(values=pose["joints"]))
            elif mode == "motion":
                target = settings.GRIPPER_NAME if self.gripper is not None else settings.ARM_NAME
                await self.motion.move(component_name=target,
                                       destination=PoseInFrame(reference_frame=settings.POSE_FRAME, pose=Pose(**pose)))
            else:
                await self.arm.move_to_position(Pose(**pose))

    async def _approach(self, pose, label):
        """Hover APPROACH_MM above a Cartesian pose; joint poses go direct."""
        if "joints" not in pose and settings.APPROACH_MM:
            await self.move_to(offset_pose(pose, {"z": settings.APPROACH_MM}), label=f"{label}+approach")

    async def open_gripper(self):
        with self._span("gripper.open", dry_run=self.dry_run or self.gripper is None):
            if self.dry_run or self.gripper is None:
                log.info("DRY-RUN gripper open")
            else:
                await self.gripper.open()
        self.holding = False

    async def grab(self) -> bool:
        with self._span("gripper.grab", dry_run=self.dry_run or self.gripper is None) as attrs:
            if self.dry_run or self.gripper is None:
                log.info("DRY-RUN gripper grab")
                self.holding = True
            else:
                self.holding = bool(await self.gripper.grab())
            attrs["holding"] = self.holding
        return self.holding

    async def pick(self, pose="pick") -> bool:
        """Open, approach from above, descend, grab, lift. Returns whether something was grabbed."""
        label = pose if isinstance(pose, str) else "pick"
        pose = resolve_pose(pose)
        await self.open_gripper()
        await self._approach(pose, label)
        await self.move_to(pose, label=label)
        grabbed = await self.grab()
        await self._approach(pose, label)
        if not grabbed:
            log.warning("grab reported nothing held at %s", pose)
        return grabbed

    async def place(self, name):
        """Carry the held item to POSES[name] (e.g. "box" or "reject"), release, retreat."""
        pose = resolve_pose(name)
        await self._approach(pose, name)
        await self.move_to(pose, label=name)
        await self.open_gripper()
        await self._approach(pose, name)

    async def stop(self):
        if self.arm is not None and not self.dry_run:
            await self.arm.stop()

    # ----------------------------------------------------------------- camera
    def set_frames(self, frames):
        """Offline mode: frames that get_image() serves (cycled) when there is no camera."""
        self._frames = itertools.cycle(list(frames))

    async def get_image(self) -> Image.Image:
        with self._span("camera", source="offline" if self.camera is None else settings.CAMERA_NAME) as attrs:
            if self.camera is None:
                if self._frames is None:
                    raise RuntimeError("no camera and no offline image_source")
                frame = next(self._frames)
                img = frame if isinstance(frame, Image.Image) else Image.open(frame).convert("RGB")
                attrs["size"] = list(img.size)
                if not isinstance(frame, Image.Image):
                    attrs["file"] = Path(frame).name
                return img

            from viam.media.video import CameraMimeType

            kw = {"filter_source_names": [settings.CAMERA_SOURCE]} if settings.CAMERA_SOURCE else {}
            t0 = time.perf_counter()
            images, _ = await self.camera.get_images(**kw)
            if not images and kw:  # source name wrong: fall back to everything
                attrs["source_fallback"] = True
                images, _ = await self.camera.get_images()
            attrs["rpc_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            attrs["sources"] = [f"{i.name}:{i.mime_type}" for i in images]
            for img in images:  # first colour image; depth frames are skipped
                if img.mime_type in (CameraMimeType.JPEG, CameraMimeType.PNG):
                    t0 = time.perf_counter()
                    frame = Image.open(io.BytesIO(img.data)).convert("RGB")
                    attrs.update(used=img.name, bytes=len(img.data), size=list(frame.size),
                                 decode_ms=round((time.perf_counter() - t0) * 1000, 2))
                    # crop live frames only: datasets are saved already cropped, so
                    # training, offline runs and the live loop all see the same pixels
                    if settings.CROP:
                        frame = frame.crop(tuple(settings.CROP))
                        attrs["crop"] = list(settings.CROP)
                    return frame
            raise RuntimeError(f"no JPEG/PNG image from camera; got {attrs['sources']}")

    async def capture(self, views=None) -> list[Image.Image]:
        """One frame per view. views: int (first N of settings.VIEW_OFFSETS) or a list of offsets.
        With more than one view the arm moves the held item / wrist camera between frames."""
        views = settings.NUM_VIEWS if views is None else views
        offsets = settings.VIEW_OFFSETS[:views] if isinstance(views, int) else list(views)
        frames = []
        for i, off in enumerate(offsets):
            if len(offsets) > 1 or off:
                await self.move_to(offset_pose(resolve_pose("inspect"), off), label=f"inspect{off or ''}")
                if self.connected and not self.dry_run and settings.SETTLE_S:
                    with self._span("settle", s=settings.SETTLE_S):
                        await asyncio.sleep(settings.SETTLE_S)
            frames.append(await self.get_image())
        return frames

    # ------------------------------------------------------------ inspection
    async def describe(self) -> dict:
        """Resources and the arm's current pose / joints (paste into settings.POSES)."""
        out = {"connected": self.connected, "dry_run": self.dry_run}
        if not self.connected:
            return out
        out["resources"] = sorted(f"{r.subtype}:{r.name}" for r in self.machine.resource_names)
        if self.arm is not None:
            p = await self.arm.get_end_position()
            out["pose"] = {k: round(getattr(p, k), 2) for k in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")}
            j = await self.arm.get_joint_positions()
            out["joints"] = {"joints": [round(v, 2) for v in j.values]}
        return out


async def _cli(args):
    async with Robot(dry_run=args.dry_run, address=args.address) as robot:
        print(json.dumps(await robot.describe(), indent=2))
        if args.snap and robot.camera is not None:
            t0 = time.perf_counter()
            img = await robot.get_image()
            img.save(args.snap)
            print(f"saved {args.snap} {img.size} in {(time.perf_counter() - t0) * 1000:.0f} ms")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--address", default=None)
    ap.add_argument("--dry-run", action="store_true", help="never move the arm")
    ap.add_argument("--snap", nargs="?", const="/tmp/snap.png", default=None, help="save one camera frame")
    asyncio.run(_cli(ap.parse_args()))


if __name__ == "__main__":
    main()
