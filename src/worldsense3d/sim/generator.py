from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import pybullet as p
except ImportError as exc:  # pragma: no cover
    raise ImportError("pybullet is required for simulation: pip install pybullet") from exc


COLORS = [
    (0.90, 0.15, 0.15, 1.0),
    (0.15, 0.35, 0.95, 1.0),
    (0.15, 0.75, 0.30, 1.0),
    (0.95, 0.75, 0.12, 1.0),
    (0.65, 0.20, 0.85, 1.0),
]


@dataclass
class SimulationConfig:
    min_objects: int = 3
    max_objects: int = 5
    frames: int = 9
    context_frames: int = 4
    frame_stride: int = 30
    physics_hz: int = 240
    image_size: int = 128
    camera_distance: float = 3.0
    camera_yaw: float = 45.0
    camera_pitch: float = -50.0
    near: float = 0.05
    far: float = 5.0
    force_min: float = 4.0
    force_max: float = 12.0
    force_duration_steps: int = 45


class EpisodeGenerator:
    def __init__(self, cfg: SimulationConfig, seed: int = 7) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.client = p.connect(p.DIRECT)
        p.setPhysicsEngineParameter(fixedTimeStep=1.0 / cfg.physics_hz, physicsClientId=self.client)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client)

    def close(self) -> None:
        if p.isConnected(self.client):
            p.disconnect(self.client)

    def _create_object(self, index: int, position: np.ndarray) -> int:
        shape_kind = index % 3
        size = float(self.rng.uniform(0.09, 0.14))
        if shape_kind == 0:
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size] * 3, physicsClientId=self.client)
            visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[size] * 3, rgbaColor=COLORS[index], physicsClientId=self.client
            )
        elif shape_kind == 1:
            collision = p.createCollisionShape(p.GEOM_SPHERE, radius=size, physicsClientId=self.client)
            visual = p.createVisualShape(
                p.GEOM_SPHERE, radius=size, rgbaColor=COLORS[index], physicsClientId=self.client
            )
        else:
            collision = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=size, height=size * 2, physicsClientId=self.client
            )
            visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=size,
                length=size * 2,
                rgbaColor=COLORS[index],
                physicsClientId=self.client,
            )
        mass = float(self.rng.uniform(0.4, 1.5))
        body = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position.tolist(),
            physicsClientId=self.client,
        )
        p.changeDynamics(
            body,
            -1,
            lateralFriction=float(self.rng.uniform(0.25, 0.85)),
            restitution=float(self.rng.uniform(0.0, 0.25)),
            physicsClientId=self.client,
        )
        return body

    def _camera(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.cfg
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0.15],
            distance=cfg.camera_distance,
            yaw=cfg.camera_yaw,
            pitch=cfg.camera_pitch,
            roll=0,
            upAxisIndex=2,
        )
        proj = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=1.0,
            nearVal=cfg.near,
            farVal=cfg.far,
        )
        _, _, rgba, zbuf, seg = p.getCameraImage(
            cfg.image_size,
            cfg.image_size,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            physicsClientId=self.client,
        )
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(cfg.image_size, cfg.image_size, 4)[..., :3]
        zbuf = np.asarray(zbuf, dtype=np.float32).reshape(cfg.image_size, cfg.image_size)
        depth = (cfg.far * cfg.near) / (cfg.far - (cfg.far - cfg.near) * zbuf)
        seg = np.asarray(seg, dtype=np.int32).reshape(cfg.image_size, cfg.image_size)
        return rgb, depth.astype(np.float32), seg

    def _state(self, body_ids: list[int]) -> np.ndarray:
        out = np.zeros((self.cfg.max_objects, 13), dtype=np.float32)
        for i, body in enumerate(body_ids):
            pos, quat = p.getBasePositionAndOrientation(body, physicsClientId=self.client)
            lin, ang = p.getBaseVelocity(body, physicsClientId=self.client)
            out[i] = np.asarray([*pos, *quat, *lin, *ang], dtype=np.float32)
        return out

    def _semantic_segmentation(self, raw: np.ndarray, body_ids: list[int]) -> np.ndarray:
        ids = raw & ((1 << 24) - 1)
        out = np.zeros_like(ids, dtype=np.uint8)
        for i, body in enumerate(body_ids, start=1):
            out[ids == body] = i
        return out

    def generate(self) -> dict[str, np.ndarray]:
        cfg = self.cfg
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client)
        p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.client)
        plane = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.35, 1.35, 0.04], physicsClientId=self.client)
        plane_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[1.35, 1.35, 0.04], rgbaColor=[0.72, 0.72, 0.72, 1], physicsClientId=self.client
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=plane,
            baseVisualShapeIndex=plane_vis,
            basePosition=[0, 0, -0.04],
            physicsClientId=self.client,
        )

        n = int(self.rng.integers(cfg.min_objects, cfg.max_objects + 1))
        body_ids: list[int] = []
        positions: list[np.ndarray] = []
        for i in range(n):
            for _ in range(100):
                xy = self.rng.uniform(-0.75, 0.75, size=2)
                if all(np.linalg.norm(xy - q[:2]) > 0.35 for q in positions):
                    break
            pos = np.array([xy[0], xy[1], 0.18], dtype=np.float32)
            positions.append(pos)
            body_ids.append(self._create_object(i, pos))

        for _ in range(60):
            p.stepSimulation(physicsClientId=self.client)

        target_index = int(self.rng.integers(0, n))
        theta = float(self.rng.uniform(0, 2 * math.pi))
        direction = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=np.float32)
        force = float(self.rng.uniform(cfg.force_min, cfg.force_max))
        action = np.array(
            [
                target_index / max(cfg.max_objects - 1, 1),
                direction[0],
                direction[1],
                force / cfg.force_max,
                cfg.force_duration_steps / cfg.physics_hz,
                (cfg.context_frames - 1) / max(cfg.frames - 1, 1),
            ],
            dtype=np.float32,
        )

        rgbs, depths, segs, states = [], [], [], []
        force_left = cfg.force_duration_steps
        for frame in range(cfg.frames):
            rgb, depth, seg_raw = self._camera()
            rgbs.append(rgb)
            depths.append(depth)
            segs.append(self._semantic_segmentation(seg_raw, body_ids))
            states.append(self._state(body_ids))

            for _ in range(cfg.frame_stride):
                if frame >= cfg.context_frames - 1 and force_left > 0:
                    p.applyExternalForce(
                        body_ids[target_index],
                        -1,
                        (direction * force).tolist(),
                        [0, 0, 0],
                        p.WORLD_FRAME,
                        physicsClientId=self.client,
                    )
                    force_left -= 1
                p.stepSimulation(physicsClientId=self.client)

        object_mask = np.zeros(cfg.max_objects, dtype=np.float32)
        object_mask[:n] = 1.0
        return {
            "rgb": np.stack(rgbs),
            "depth": np.stack(depths),
            "segmentation": np.stack(segs),
            "states": np.stack(states),
            "action": action,
            "object_mask": object_mask,
            "target_index": np.asarray(target_index, dtype=np.int64),
            "num_objects": np.asarray(n, dtype=np.int64),
        }

    def save_episode(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.generate())
