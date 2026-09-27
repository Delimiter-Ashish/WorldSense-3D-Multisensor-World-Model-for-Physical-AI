from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as p

from .generator import COLORS, SimulationConfig


@dataclass(frozen=True)
class ObjectSpec:
    shape_kind: int
    size: float
    mass: float
    friction: float
    restitution: float
    position: tuple[float, float, float]


@dataclass(frozen=True)
class SceneSpec:
    objects: tuple[ObjectSpec, ...]
    target_index: int
    direction: tuple[float, float, float]
    force: float


class CounterfactualEpisodeGenerator:
    def __init__(self, cfg: SimulationConfig, seed: int = 7) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.client = p.connect(p.DIRECT)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / cfg.physics_hz,
            physicsClientId=self.client,
        )
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client)

    def close(self) -> None:
        if p.isConnected(self.client):
            p.disconnect(self.client)

    def _sample_scene(self) -> SceneSpec:
        cfg = self.cfg
        n = int(self.rng.integers(cfg.min_objects, cfg.max_objects + 1))

        positions: list[np.ndarray] = []
        objects: list[ObjectSpec] = []

        for i in range(n):
            for _ in range(100):
                xy = self.rng.uniform(-0.75, 0.75, size=2)
                if all(np.linalg.norm(xy - q[:2]) > 0.35 for q in positions):
                    break

            pos = np.array([xy[0], xy[1], 0.18], dtype=np.float32)
            positions.append(pos)

            objects.append(
                ObjectSpec(
                    shape_kind=i % 3,
                    size=float(self.rng.uniform(0.09, 0.14)),
                    mass=float(self.rng.uniform(0.4, 1.5)),
                    friction=float(self.rng.uniform(0.25, 0.85)),
                    restitution=float(self.rng.uniform(0.0, 0.25)),
                    position=(float(pos[0]), float(pos[1]), float(pos[2])),
                )
            )

        target_index = int(self.rng.integers(0, n))
        theta = float(self.rng.uniform(0.0, 2.0 * math.pi))
        direction = (
            float(math.cos(theta)),
            float(math.sin(theta)),
            0.0,
        )
        force = float(self.rng.uniform(cfg.force_min, cfg.force_max))

        return SceneSpec(
            objects=tuple(objects),
            target_index=target_index,
            direction=direction,
            force=force,
        )

    def _reset_world(self) -> None:
        p.resetSimulation(physicsClientId=self.client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / self.cfg.physics_hz,
            physicsClientId=self.client,
        )
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client)

        plane = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[1.35, 1.35, 0.04],
            physicsClientId=self.client,
        )
        plane_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[1.35, 1.35, 0.04],
            rgbaColor=[0.72, 0.72, 0.72, 1.0],
            physicsClientId=self.client,
        )
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=plane,
            baseVisualShapeIndex=plane_vis,
            basePosition=[0.0, 0.0, -0.04],
            physicsClientId=self.client,
        )

    def _create_object(
        self,
        index: int,
        spec: ObjectSpec,
        mass_scale: float,
        friction_scale: float,
        target_index: int,
    ) -> int:
        size = spec.size

        if spec.shape_kind == 0:
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[size] * 3,
                physicsClientId=self.client,
            )
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[size] * 3,
                rgbaColor=COLORS[index],
                physicsClientId=self.client,
            )
        elif spec.shape_kind == 1:
            collision = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=size,
                physicsClientId=self.client,
            )
            visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=size,
                rgbaColor=COLORS[index],
                physicsClientId=self.client,
            )
        else:
            collision = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=size,
                height=size * 2,
                physicsClientId=self.client,
            )
            visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=size,
                length=size * 2,
                rgbaColor=COLORS[index],
                physicsClientId=self.client,
            )

        is_target = index == target_index
        mass = spec.mass * (mass_scale if is_target else 1.0)
        friction = spec.friction * (friction_scale if is_target else 1.0)

        body = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=list(spec.position),
            physicsClientId=self.client,
        )

        p.changeDynamics(
            body,
            -1,
            lateralFriction=friction,
            restitution=spec.restitution,
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

        rgb = np.asarray(rgba, dtype=np.uint8).reshape(
            cfg.image_size, cfg.image_size, 4
        )[..., :3]

        zbuf = np.asarray(zbuf, dtype=np.float32).reshape(
            cfg.image_size, cfg.image_size
        )

        depth = (cfg.far * cfg.near) / (
            cfg.far - (cfg.far - cfg.near) * zbuf
        )

        seg = np.asarray(seg, dtype=np.int32).reshape(
            cfg.image_size, cfg.image_size
        )

        return rgb, depth.astype(np.float32), seg

    def _state(self, body_ids: list[int]) -> np.ndarray:
        out = np.zeros((self.cfg.max_objects, 13), dtype=np.float32)

        for i, body in enumerate(body_ids):
            pos, quat = p.getBasePositionAndOrientation(
                body, physicsClientId=self.client
            )
            lin, ang = p.getBaseVelocity(
                body, physicsClientId=self.client
            )
            out[i] = np.asarray(
                [*pos, *quat, *lin, *ang],
                dtype=np.float32,
            )

        return out

    def _semantic_segmentation(
        self,
        raw: np.ndarray,
        body_ids: list[int],
    ) -> np.ndarray:
        ids = raw & ((1 << 24) - 1)
        out = np.zeros_like(ids, dtype=np.uint8)

        for i, body in enumerate(body_ids, start=1):
            out[ids == body] = i

        return out

    def _rollout(
        self,
        scene: SceneSpec,
        force_scale: float = 1.0,
        mass_scale: float = 1.0,
        friction_scale: float = 1.0,
        cancel_action: bool = False,
    ) -> dict[str, np.ndarray]:
        cfg = self.cfg
        self._reset_world()

        body_ids = [
            self._create_object(
                i,
                spec,
                mass_scale,
                friction_scale,
                scene.target_index,
            )
            for i, spec in enumerate(scene.objects)
        ]

        for _ in range(60):
            p.stepSimulation(physicsClientId=self.client)

        effective_force = 0.0 if cancel_action else scene.force * force_scale
        direction = np.asarray(scene.direction, dtype=np.float32)

        action = np.asarray(
            [
                scene.target_index / max(cfg.max_objects - 1, 1),
                direction[0],
                direction[1],
                effective_force / cfg.force_max,
                cfg.force_duration_steps / cfg.physics_hz,
                (cfg.context_frames - 1) / max(cfg.frames - 1, 1),
            ],
            dtype=np.float32,
        )

        rgbs = []
        depths = []
        segs = []
        states = []

        force_left = 0 if cancel_action else cfg.force_duration_steps

        for frame in range(cfg.frames):
            rgb, depth, seg_raw = self._camera()

            rgbs.append(rgb)
            depths.append(depth)
            segs.append(
                self._semantic_segmentation(seg_raw, body_ids)
            )
            states.append(self._state(body_ids))

            for _ in range(cfg.frame_stride):
                if frame >= cfg.context_frames - 1 and force_left > 0:
                    p.applyExternalForce(
                        body_ids[scene.target_index],
                        -1,
                        (direction * effective_force).tolist(),
                        [0, 0, 0],
                        p.WORLD_FRAME,
                        physicsClientId=self.client,
                    )
                    force_left -= 1

                p.stepSimulation(physicsClientId=self.client)

        object_mask = np.zeros(cfg.max_objects, dtype=np.float32)
        object_mask[: len(scene.objects)] = 1.0

        target = scene.objects[scene.target_index]

        return {
            "rgb": np.stack(rgbs),
            "depth": np.stack(depths),
            "segmentation": np.stack(segs),
            "states": np.stack(states),
            "action": action,
            "object_mask": object_mask,
            "target_mass": np.asarray(
                target.mass * mass_scale, dtype=np.float32
            ),
            "target_friction": np.asarray(
                target.friction * friction_scale, dtype=np.float32
            ),
            "effective_force": np.asarray(
                effective_force, dtype=np.float32
            ),
        }

    def generate_pair(self) -> dict[str, np.ndarray]:
        scene = self._sample_scene()

        factual = self._rollout(scene)

        intervention = str(
            self.rng.choice(
                ["force", "mass", "friction", "cancel_action"]
            )
        )

        force_scale = 1.0
        mass_scale = 1.0
        friction_scale = 1.0
        cancel_action = False
        intervention_scale = 1.0

        if intervention == "force":
            intervention_scale = float(
                self.rng.choice([0.5, 1.5])
            )
            force_scale = intervention_scale

        elif intervention == "mass":
            intervention_scale = float(
                self.rng.choice([0.5, 2.0])
            )
            mass_scale = intervention_scale

        elif intervention == "friction":
            intervention_scale = float(
                self.rng.choice([0.5, 1.5])
            )
            friction_scale = intervention_scale

        else:
            intervention_scale = 0.0
            cancel_action = True

        counterfactual = self._rollout(
            scene,
            force_scale=force_scale,
            mass_scale=mass_scale,
            friction_scale=friction_scale,
            cancel_action=cancel_action,
        )

        start = self.cfg.context_frames
        target = scene.target_index

        factual_future = factual["states"][start:, target, :3]
        counterfactual_future = counterfactual["states"][start:, target, :3]

        trajectory_shift = np.linalg.norm(
            counterfactual_future - factual_future,
            axis=-1,
        )

        intervention_codes = {
            "force": 0,
            "mass": 1,
            "friction": 2,
            "cancel_action": 3,
        }

        out: dict[str, np.ndarray] = {}

        for prefix, branch in (
            ("factual", factual),
            ("counterfactual", counterfactual),
        ):
            for key, value in branch.items():
                out[f"{prefix}_{key}"] = value

        out.update(
            {
                "target_index": np.asarray(
                    target, dtype=np.int64
                ),
                "num_objects": np.asarray(
                    len(scene.objects), dtype=np.int64
                ),
                "intervention_type": np.asarray(intervention),
                "intervention_code": np.asarray(
                    intervention_codes[intervention],
                    dtype=np.int64,
                ),
                "intervention_scale": np.asarray(
                    intervention_scale,
                    dtype=np.float32,
                ),
                "target_trajectory_shift": trajectory_shift.astype(
                    np.float32
                ),
                "mean_target_shift": np.asarray(
                    trajectory_shift.mean(),
                    dtype=np.float32,
                ),
            }
        )

        return out

    def save_pair(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.generate_pair())
