"""Native Isaac scene backend. Import only after Isaac AppLauncher starts."""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from isaaclab import sim
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim.utils.transforms import standardize_xform_ops
from omni.physics.tensors import create_simulation_view
from PIL import Image
from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from .paths import RIGHT_JOINTS, asset_path
from .scene_assets import collision_bounds, collision_support, remap_render_assets
from .scene_fixture import pose_matrix, transform_points
from .scene_kinematics import KinematicTree
from .scene_objects import install_objects
from .settling import SIDES, verify_settled


def _save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class NativeBimanualScene:
    """Two native torque hands, native arms/materials, GT initialization and target contacts."""

    def __init__(self, config, out, device="cuda:0"):
        self.config, self.out = config, out
        self.started = time.monotonic()
        source = asset_path(config["source_name"], config["metadata_directory"])

        def read(name):
            return json.loads((source / name).read_bytes())

        scenario, motors, initial = read("scenario.json"), read("hand_motor_config.json"), read("initial_pose.json")
        self.world = sim.SimulationContext(
            sim.SimulationCfg(
                dt=config["physics_dt"],
                device=device,
                use_fabric=True,
                physx=sim.PhysxCfg(
                    enable_external_forces_every_iteration=True,
                    max_velocity_iteration_count=4,
                    max_position_iteration_count=config["maximum_solver_iterations"],
                ),
            )
        )
        stage = self.world.stage
        layer = Sdf.Layer.FindOrOpen(str(asset_path(config["source_name"], config["scene_file"])))
        if layer is None:
            raise FileNotFoundError("restore the content-addressed native scene first")
        stage.GetRootLayer().TransferContent(layer)
        _save(out / "render_assets.json", remap_render_assets(stage, source))
        if any(config.get(key) for key in ("added_objects", "parked_source_objects", "removed_source_objects")):
            scenario = install_objects(stage, scenario, config)
            _save(
                out / "scene_variant.json",
                {
                    "objects": config.get("added_objects", []),
                    "parked": config.get("parked_source_objects", []),
                    "removed": config.get("removed_source_objects", []),
                    "scenario": scenario,
                },
            )

        self.fixture_path = None
        self.bare_tcp_support = {}
        self.fixture_sides = ()
        hardware = config.get("wrist_hardware")
        if hardware in ("removed", "left-only"):
            from .scene_wrist_mount import remove_wrist_hardware

            bare = ("right",) if hardware == "left-only" else ("right", "left")
            report = remove_wrist_hardware(stage, config["robot_prim"], bare)
            self.bare_tcp_support = {
                side: collision_support(
                    stage, config["robot_prim"] + "/TCP_Link_" + side[0].upper()
                )
                for side in bare
            }
        if "tabletop_seed" in config and hardware != "removed":
            from .scene_wrist_mount import install_mount, resolve_mount

            self.fixture_path = resolve_mount(config.get("wrist_mount"))
            if hardware == "left-only" and self.fixture_path is None:
                raise ValueError("left observer requires the registered wrist mount bundle")
            if self.fixture_path:
                self.fixture_sides = (
                    ("left",) if hardware == "left-only" else ("right", "left")
                )
                install_mount(
                    stage, config["robot_prim"], self.fixture_path, self.fixture_sides
                )
        if hardware in ("removed", "left-only"):
            if hardware == "left-only":
                tcp = stage.GetPrimAtPath(config["robot_prim"] + "/TCP_Link_L")
                prims = list(Usd.PrimRange(tcp, Usd.TraverseInstanceProxies()))
                report["left"] = {
                    "active_wrist_cameras": [
                        str(p.GetPath()) for p in prims if p.IsA(UsdGeom.Camera)
                    ],
                    "fixture_colliders": sum(
                        p.HasAPI(UsdPhysics.CollisionAPI)
                        and "WristCameraFixture" in str(p.GetPath())
                        for p in prims
                    ),
                }
            _save(out / "wrist_assembly.json", report)

        def mapping(key):
            return dict(zip(motors["joint_names"], motors[key][0]))

        initial_joints = initial["joint_positions"].copy()
        for name, value in config.get("initial_joint_positions", {}).items():
            if name not in initial_joints or not np.isfinite(value):
                raise ValueError(f"invalid initial joint position for {name}")
            initial_joints[name] = float(value)
        self.tree = KinematicTree.from_urdf(source / "robot_cad.urdf")
        if hardware == "left-only":
            left_arm = [f"Joint{i}_L" for i in range(1, 8)]
            bounds = self.tree.joint_limits(left_arm)
            angles = np.asarray([initial_joints[joint] for joint in left_arm])
            if np.any((angles < bounds[:, 0]) | (angles > bounds[:, 1])):
                raise ValueError("left observation pose exceeds native arm limits")
        self.robot = Articulation(
            ArticulationCfg(
                prim_path=config["robot_prim"],
                spawn=None,
                init_state=ArticulationCfg.InitialStateCfg(joint_pos=initial_joints),
                actuators={
                    "arm": ImplicitActuatorCfg(
                        joint_names_expr=["Joint[1-7]_.*"],
                        stiffness=config["arm_stiffness"],
                        damping=config["arm_damping"],
                    ),
                    "hand": IdealPDActuatorCfg(
                        joint_names_expr=["left_.*", "right_.*"],
                        stiffness=mapping("stiffness_Nm_rad"),
                        damping=mapping("damping_Nms_rad"),
                        effort_limit=mapping("effort_limits_Nm"),
                        effort_limit_sim=mapping("effort_limits_Nm"),
                        armature=motors["armature_kg_m2"],
                    ),
                },
            )
        )
        self.objects = {
            name: RigidObject(
                RigidObjectCfg(
                    prim_path=path,
                    spawn=None,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=tuple(scenario["objects"][name]["position"]),
                        rot=tuple(scenario["objects"][name]["quaternion_wxyz"]),
                    ),
                )
            )
            for name, path in config["object_prims"].items()
        }
        camera_prim = stage.GetPrimAtPath(config["camera_prim"])
        before = np.asarray(UsdGeom.Xformable(camera_prim).GetLocalTransformation())
        if not standardize_xform_ops(camera_prim):
            raise RuntimeError("cannot adapt native camera transform")
        np.testing.assert_allclose(
            np.asarray(UsdGeom.Xformable(camera_prim).GetLocalTransformation()), before, atol=1e-6, rtol=0
        )
        data_types = ["rgb", "distance_to_image_plane"]
        if "tabletop_seed" not in config:
            data_types.append("instance_id_segmentation_fast")
        self.camera = Camera(
            CameraCfg(
                prim_path=config["camera_prim"],
                spawn=None,
                width=config["images"]["width"],
                height=config["images"]["height"],
                data_types=data_types,
                update_period=0.0,
                update_latest_camera_pose=True,
                colorize_instance_id_segmentation=False,
            )
        )
        self.wrist_devices = {}
        paths = {}
        if hardware == "left-only":
            paths["left"] = config["robot_prim"] + "/TCP_Link_L/wrist2_camera"
        elif config.get("camera_mode") != "head" and config.get(
            "head_wrist_validation"
        ):
            paths["right"] = config["head_wrist_validation"]["camera_prim"]
            if config.get("active_hands") in ("both", "left"):
                paths["left"] = config["robot_prim"] + "/TCP_Link_L/wrist2_camera"
        if paths:
            from .head_wrist_geometry import usd_camera_intrinsics

            for side, wrist_path in paths.items():
                wrist_prim = stage.GetPrimAtPath(wrist_path)
                if not wrist_prim or not wrist_prim.IsA(UsdGeom.Camera):
                    raise ValueError(f"missing native {side} wrist camera at {wrist_path}")
                wrist_local = np.asarray(UsdGeom.Xformable(wrist_prim).GetLocalTransformation()).copy()
                eye = wrist_local.T @ np.diag([1.0, -1.0, -1.0, 1.0])
                # Native camera optics with a world-space render proxy synchronized
                # from measured TCP at exposure, independent of USD/Fabric update lag.
                proxy_path = "/World/DexBridge" + side.title() + "WristRGBD"
                proxy = UsdGeom.Camera.Define(stage, proxy_path).GetPrim()
                for key in (
                    "clippingRange",
                    "focalLength",
                    "horizontalAperture",
                    "verticalAperture",
                    "horizontalApertureOffset",
                    "verticalApertureOffset",
                    "projection",
                ):
                    proxy.GetAttribute(key).Set(wrist_prim.GetAttribute(key).Get())
                # The Camera sensor's convenience matrix assumes centered, square pixels.
                # This USD camera is calibrated with an off-center principal point.
                focal = wrist_prim.GetAttribute("focalLength").Get()
                ha = wrist_prim.GetAttribute("horizontalAperture").Get()
                va = wrist_prim.GetAttribute("verticalAperture").Get()
                hoff = wrist_prim.GetAttribute("horizontalApertureOffset").Get()
                voff = wrist_prim.GetAttribute("verticalApertureOffset").Get()
                k = usd_camera_intrinsics(480, 640, focal, ha, va, hoff, voff)
                standardize_xform_ops(proxy)
                sensor = Camera(
                    CameraCfg(
                        prim_path=proxy_path,
                        spawn=None,
                        width=480,
                        height=640,
                        data_types=data_types,
                        update_period=0.0,
                        update_latest_camera_pose=True,
                        colorize_instance_id_segmentation=False,
                    )
                )
                self.wrist_devices[side] = (sensor, eye, k)
            if "right" in self.wrist_devices:
                self.wrist_camera, self.tcp_from_wrist_optical, self.wrist_K = self.wrist_devices["right"]
        self.observer_camera = None
        observer = config["images"].get("observer")
        if observer:
            # Independent RGB recording of the contact area, outside the controller inputs.
            observer_path = "/World/DexBridgeObserver"
            prim = UsdGeom.Camera.Define(stage, observer_path)
            prim.CreateFocalLengthAttr(observer.get("focal_length_mm", 24.0))
            prim.CreateHorizontalApertureAttr(20.0)
            prim.CreateClippingRangeAttr((0.025, 8.0))
            standardize_xform_ops(prim.GetPrim())
            self.observer_camera = Camera(
                CameraCfg(
                    prim_path=observer_path,
                    spawn=None,
                    width=768,
                    height=512,
                    data_types=["rgb"],
                    update_period=0.0,
                )
            )
            (out / "observer_frames").mkdir()
        # Reporting only: do not alter geometry, materials, inertia or solver settings.
        for prim in Usd.PrimRange(stage.GetPrimAtPath(config["robot_prim"])):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName().startswith(("left_", "right_")):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        self.world._disable_app_control_on_stop_handle = True
        self.world.reset()
        self.world._disable_app_control_on_stop_handle = True
        if observer:
            self.observer_camera.set_world_poses(
                np.asarray([observer["position_m"]], dtype="float32"),
                np.asarray([observer["rotation_wxyz_ros"]], dtype="float32"),
                convention="ros",
            )
        sdk_names = RIGHT_JOINTS
        self.hand_names = {side: [name.replace("right_", side + "_", 1) for name in sdk_names] for side in SIDES}
        if any(len(names) != 22 for names in self.hand_names.values()) or self.robot.num_joints != 58:
            raise ValueError("native scene joint contract changed")
        self.ids = {
            side: [self.robot.joint_names.index(name) for name in names] for side, names in self.hand_names.items()
        }
        self.limits = {
            side: self.robot.data.joint_pos_limits[0, ids].cpu().numpy() for side, ids in self.ids.items()
        }
        self.hand_motor = self.robot.actuators["hand"]
        motor_ids = self.hand_motor.joint_indices.cpu()
        if torch.count_nonzero(
            self.robot.root_physx_view.get_dof_stiffnesses()[:, motor_ids]
        ) or torch.count_nonzero(self.robot.root_physx_view.get_dof_dampings()[:, motor_ids]):
            raise ValueError("native torque hand unexpectedly has position/velocity drives")
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"], False, True)
        self.hand_bounds = {side: {} for side in SIDES}
        for side in SIDES:
            for name in self.robot.body_names:
                if name.startswith(side + "_"):
                    bounds = collision_bounds(stage, config["robot_prim"] + "/" + name, cache)
                    if bounds is not None:
                        self.hand_bounds[side][name] = bounds
        self.arm_mask_bounds = {}
        if "tabletop_seed" in config:
            for name in self.robot.body_names:
                if name.startswith(("Link", "TCP_Link", "Base_")) and name.endswith(("_R", "_L")):
                    bounds = collision_bounds(stage, config["robot_prim"] + "/" + name, cache)
                    if bounds is not None:
                        self.arm_mask_bounds[name] = bounds
        self.forearm_support = {}
        if config.get("check_forearm_table"):
            for side in ("R", "L"):
                for j in range(4, 8):
                    name = f"Link{j}_{side}"
                    points = collision_support(stage, config["robot_prim"] + "/" + name)
                    # Keep the surface of each physical CAD collider at millimetre resolution.
                    _, ids = np.unique(np.round(points / 0.002), axis=0, return_index=True)
                    self.forearm_support[name] = points[ids]
        self.object_bounds = {
            name: collision_bounds(stage, path, cache) for name, path in config["object_prims"].items()
        }
        if any(bounds is None for bounds in self.object_bounds.values()) or not all(self.hand_bounds.values()):
            raise ValueError("missing native collision envelopes")
        self.hand_support = {
            name: collision_support(stage, config["robot_prim"] + "/" + name)
            for bounds in self.hand_bounds.values()
            for name in bounds
        }
        self.object_support = {
            name: collision_support(stage, path) for name, path in config["object_prims"].items()
        }
        np.savez_compressed(out / "collision_support.npz", **self.hand_support, **self.object_support)
        table_range = cache.ComputeWorldBound(stage.GetPrimAtPath(config["table_prim"])).ComputeAlignedRange()
        self.table_bounds = np.array([table_range.GetMin(), table_range.GetMax()])
        self.body_ids = {
            name: self.robot.body_names.index(name) for side in SIDES for name in self.hand_bounds[side]
        }
        self.palm_ids = {side: self.robot.body_names.index(f"{side}_hand_C_MC") for side in SIDES}
        self.tensor_view = create_simulation_view("torch")
        self.tensor_view.set_subspace_roots("/")
        watched = [config["robot_prim"] + "/" + name for name in self.body_ids]
        filters = list(config["object_prims"].values())
        self.contact_view = self.tensor_view.create_rigid_contact_view(watched, [filters] * len(watched), 32768)
        self.table_contact_view = None
        if config.get("record_hand_table_contact"):
            table_shapes = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(config["table_prim"]))
                            if p.HasAPI(UsdPhysics.CollisionAPI)]
            if not table_shapes:
                raise ValueError("physical tabletop collider missing")
            self.table_contact_view = self.tensor_view.create_rigid_contact_view(
                watched, [table_shapes] * len(watched), 32768
            )
        self.contact_names = list(self.body_ids)
        self.contact_filters = list(config["object_prims"])
        self.time, self.frame_number, self.step_number = 0.0, 0, 0
        self.frames = out / "frames"
        self.frames.mkdir()
        if hardware == "left-only":
            (out / "left_observer_frames").mkdir()
        self.png_options = {"compress_level": 1} if "tabletop_seed" in config else {}
        self.settling_receipt = None
        self.frame_times = []
        self.arm_ids = {
            name: index for index, name in enumerate(self.robot.joint_names) if name.startswith("Joint")
        }
        self.target = self.robot.data.default_joint_pos.clone()
        if config.get("initial_joint_positions"):
            self.robot.reset()
            zero_velocity = torch.zeros_like(self.target)
            self.robot.write_joint_state_to_sim(self.target, zero_velocity)
            self.robot.set_joint_position_target(self.target)
            self.robot.set_joint_velocity_target(zero_velocity)
            self.robot.write_data_to_sim()
        else:
            self.robot.write_joint_state_to_sim(self.target, torch.zeros_like(self.target))
            self.robot.reset()
        for obj in self.objects.values():
            obj.write_root_state_to_sim(obj.data.default_root_state)
            obj.reset()
        self.wrist_targets = {
            side: pose_matrix(
                self.robot.data.body_pos_w[0, i].cpu().numpy(), self.robot.data.body_quat_w[0, i].cpu().numpy()
            )
            for side, i in self.palm_ids.items()
        }
        if config.get("initial_joint_positions"):
            self.world.forward()
        self._record_initial_state()
        self.capture()

    def _record_initial_state(self):
        if self.settling_receipt is not None:
            raise ValueError("initial state cannot be overwritten after action admission")
        self.initial_snapshot = self.snapshot()
        self.initial_heights = {name: state[2] for name, state in self.initial_snapshot["objects"].items()}
        self.object_references = {
            name: pose_matrix(state[:3], state[3:7]) for name, state in self.initial_snapshot["objects"].items()
        }

    def snapshot(self):
        return {
            "joint_names": self.robot.joint_names,
            "joint_pos": self.robot.data.joint_pos[0].cpu().tolist(),
            "body_names": self.robot.body_names,
            "body_pos": self.robot.data.body_pos_w[0].cpu().tolist(),
            "body_quat_wxyz": self.robot.data.body_quat_w[0].cpu().tolist(),
            "objects": {name: obj.data.root_state_w[0].cpu().tolist() for name, obj in self.objects.items()},
        }

    def sync_wrist_camera(self, side):
        """Drive the render proxy from the physical TCP. Controller extrinsics use encoder FK."""
        sensor, eye, _ = self.wrist_devices[side]
        body = self.robot.body_names.index("TCP_Link_" + side[0].upper())
        physical = (
            pose_matrix(
                self.robot.data.body_pos_w[0, body].cpu().numpy(),
                self.robot.data.body_quat_w[0, body].cpu().numpy(),
            )
            @ eye
        )
        quat = np.roll(Rotation.from_matrix(physical[:3, :3]).as_quat(), 1)
        sensor.set_world_poses(
            physical[None, :3, 3].astype("float32"),
            quat[None].astype("float32"),
            convention="ros",
        )

    def capture(self):
        record_left = self.config.get("wrist_hardware") == "left-only"
        if record_left:
            self.sync_wrist_camera("left")
        self.world.render()
        if record_left:
            side_sensor = self.wrist_devices["left"][0]
            side_sensor.update(self.config["physics_dt"], force_recompute=True)
            rgb = side_sensor.data.output["rgb"][0, ..., :3].cpu().numpy()
            Image.fromarray(rgb).save(self.out / "left_observer_frames" / f"{self.frame_number:06d}.png", **self.png_options)
        self.camera.update(self.config["physics_dt"], force_recompute=True)
        image = self.camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        Image.fromarray(image).save(self.frames / f"{self.frame_number:06d}.png", **self.png_options)
        if self.observer_camera is not None:
            self.observer_camera.update(self.config["physics_dt"], force_recompute=True)
            rgb = self.observer_camera.data.output["rgb"][0, ..., :3].cpu().numpy()
            Image.fromarray(rgb).save(self.out / "observer_frames" / f"{self.frame_number:06d}.png", **self.png_options)
        self.frame_times.append({"frame": self.frame_number, "sim_time": self.time})
        self.frame_number += 1

    def _measure(self):
        state = self.snapshot()
        forces, _points, _normals, _separations, counts, starts = self.contact_view.get_contact_data(
            self.config["physics_dt"]
        )
        forces, counts, starts = (value.cpu().numpy() for value in (forces, counts, starts))
        contact_counts = {side: 0 for side in SIDES}
        pad_contacts = {}
        validation = self.config.get("head_wrist_validation", {})
        record_links = validation.get("max_grasp_attempts") or validation.get("record_link_forces")
        record_surfaces = validation.get("multi_finger_handle") or validation.get("tactile_seating")
        link_forces = {}
        tactile_surfaces = {}
        palm_contact_local = None
        for side in SIDES:
            column = self.contact_filters.index(self.config["assignment"][side])
            for index, name in enumerate(self.contact_names):
                if name.startswith(side + "_"):
                    start, count = int(starts[index, column]), int(counts[index, column])
                    if count:
                        hits = int(
                            np.count_nonzero(forces[start : start + count] > self.config["minimum_contact_force_N"])
                        )
                        contact_counts[side] += hits
                        if record_links and side == "right" and hits:
                            normal = forces[start : start + count]
                            link_forces[name] = float(normal[normal > self.config["minimum_contact_force_N"]].sum())
                            if record_surfaces:
                                strongest = start + int(np.argmax(normal))
                                tactile_surfaces[name] = (
                                    _points[strongest].cpu().numpy().tolist(),
                                    _normals[strongest].cpu().numpy().tolist(),
                                )
                            if name == "right_hand_C_MC":
                                mask = normal.reshape(-1) > self.config["minimum_contact_force_N"]
                                points = _points[start : start + count].cpu().numpy()[mask]
                                body = self.body_ids[name]
                                palm_pose = pose_matrix(state["body_pos"][body], state["body_quat_wxyz"][body])
                                location = np.average(points, axis=0, weights=normal.reshape(-1)[mask])
                                palm_contact_local = ((location - palm_pose[:3, 3]) @ palm_pose[:3, :3]).tolist()
                        if self.config.get("head_wrist_validation") and name.endswith("_elastomer") and hits:
                            pad_contacts[name] = hits
        penetration = 0.0
        for name, index in self.body_ids.items():
            points = transform_points(
                pose_matrix(state["body_pos"][index], state["body_quat_wxyz"][index]), self.hand_support[name]
            )
            lower, upper = points.min(axis=0), points.max(axis=0)
            overlap_xy = np.all(upper[:2] >= self.table_bounds[0, :2]) and np.all(
                lower[:2] <= self.table_bounds[1, :2]
            )
            if overlap_xy and upper[2] >= self.table_bounds[0, 2]:
                penetration = max(penetration, float(self.table_bounds[1, 2] - lower[2]))
        sample = {"time": self.time, "finite": True, "hand_table_penetration_m": penetration}
        if self.config.get("head_wrist_validation"):
            sample["pad_contacts"] = pad_contacts
        if record_links:
            sample["link_normal_N"] = link_forces
            sample["palm_contact_local_m"] = palm_contact_local
            if record_surfaces:
                sample["link_contact_surfaces"] = tactile_surfaces
        for side in SIDES:
            index = self.palm_ids[side]
            palm = pose_matrix(state["body_pos"][index], state["body_quat_wxyz"][index])
            goal = self.wrist_targets[side]
            name = self.config["assignment"][side]
            object_state = state["objects"][name]
            obj_pose = pose_matrix(object_state[:3], object_state[3:7])
            points = transform_points(obj_pose, self.object_support[name])
            sample[side] = {
                "contact_count": contact_counts[side],
                "wrist_position_error_m": float(np.linalg.norm(palm[:3, 3] - goal[:3, 3])),
                "wrist_rotation_error_rad": float(Rotation.from_matrix(goal[:3, :3].T @ palm[:3, :3]).magnitude()),
                "joint_error_rad": float(
                    torch.max(
                        torch.abs(self.robot.data.joint_pos[0, self.ids[side]] - self.target[0, self.ids[side]])
                    )
                ),
                "clearance_m": float(points[:, 2].min() - self.table_bounds[1, 2]),
                "rise_m": float(object_state[2] - self.initial_heights[name]),
                "object_translation_m": float(
                    np.linalg.norm(obj_pose[:3, 3] - self.object_references[name][:3, 3])
                ),
                "object_rotation_rad": float(
                    Rotation.from_matrix(self.object_references[name][:3, :3].T @ obj_pose[:3, :3]).magnitude()
                ),
                "object_linear_speed_m_s": float(np.linalg.norm(object_state[7:10])),
                "object_angular_speed_rad_s": float(np.linalg.norm(object_state[10:13])),
            }
            if record_links and side == "right":
                sample[side]["object_pose_wxyz"] = list(object_state[:7])
        sample["finite"] = bool(
            np.isfinite(state["joint_pos"]).all()
            and all(np.isfinite(value).all() for value in state["objects"].values())
        )
        return sample

    def advance_dynamics(self):
        self.robot.set_joint_position_target(self.target)
        self.robot.set_joint_velocity_target(torch.zeros_like(self.target))
        self.robot.set_joint_effort_target(
            self.robot.root_physx_view.get_gravity_compensation_forces().clamp(
                -self.robot.data.joint_effort_limits, self.robot.data.joint_effort_limits
            )
        )
        self.robot.write_data_to_sim()
        self.world.step(render=False)
        self.robot.update(self.config["physics_dt"])
        for obj in self.objects.values():
            obj.update(self.config["physics_dt"])
        self.time += self.config["physics_dt"]
        self.step_number += 1

    def _step(self):
        self.advance_dynamics()
        sample = self._measure()
        if self.step_number % round(1 / self.config["physics_dt"] / self.config["images"]["fps"]) == 0:
            self.capture()
        return sample

    def settle(self):
        if self.settling_receipt is not None:
            raise ValueError("settling is initialization-only; references cannot be reset during actions")
        settings = self.config["settling"]
        start, samples = self.time, []
        for _ in range(round(settings["maximum_seconds"] / self.config["physics_dt"])):
            samples.append(self._step())
            verification = verify_settled(samples, start, settings)
            if verification["passed"]:
                break
        snapshot = self.snapshot()
        self.settling_receipt = {
            "stage": "settling",
            "verification": verification,
            "samples": samples,
            "final": snapshot,
        }
        if verification["passed"]:
            self.object_references = {
                name: pose_matrix(state[:3], state[3:7]) for name, state in snapshot["objects"].items()
            }
        _save(self.out / "settling.json", self.settling_receipt)
        return self.settling_receipt
