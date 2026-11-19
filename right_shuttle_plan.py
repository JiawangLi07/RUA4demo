"""Check right-hand pinch candidates against a saved live scene and robot mesh."""

from argparse import ArgumentParser
from pathlib import Path
import json

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from async_policy import FastLeftIK
from hardware import DEPLOYMENT, arm_settings, hand_settings
from kinematics import URDFKinematics


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("observation", type=Path)
    args = parser.parse_args()
    run = args.observation.resolve()
    (run / "right_shuttle_plan.json").unlink(missing_ok=True)
    obs = json.loads((run / "observation.json").read_text())
    scene = json.loads((run / "right_shuttle_scene.json").read_text())
    state = obs["joints"]
    model = mj.MjModel.from_xml_path(str(run / "right_scene_model.xml"))
    data = mj.MjData(model)
    addresses = {}
    for side, suffix in (("left", "L"), ("right", "R")):
        for i, angle in enumerate(state["arm_deg"][side], 1):
            data.qpos[model.joint(f"Joint{i}_{suffix}").qposadr[0]] = np.deg2rad(angle)
        names = [model.joint(i).name for i in range(model.njnt)
                 if model.joint(i).name.startswith(side + "_")]
        addresses[side] = [model.joint(name).qposadr[0] for name in names]
        data.qpos[addresses[side]] = state["hand_rad"][side]
    mj.mj_forward(model, data)
    wrist = data.site("right_tcp").xpos.copy()
    rotation = data.site("right_tcp").xmat.reshape(3, 3).copy()
    up = np.array(scene["table_up"])
    center = np.array(scene["cork_center"])
    radius = scene["cork_radius"]
    plane_d = scene["table_d"]
    right_geoms, left_geoms = [], []
    for i in range(model.ngeom):
        geom = model.geom(i)
        if geom.type[0] != mj.mjtGeom.mjGEOM_MESH or "collision_" in geom.name:
            continue
        if geom.name.startswith("right_") or ("_R_visual" in geom.name and "Base" not in geom.name):
            right_geoms.append(i)
        if geom.name.startswith("left_") or ("_L_visual" in geom.name and "Base" not in geom.name):
            left_geoms.append(i)
    hand_geoms = [i for i in right_geoms if model.geom(i).name.startswith("right_")]
    pad_geoms = {name: model.geom(f"right_{name}_elastomer_visual").id
                 for name in ("thumb", "index")}

    def vertices(i):
        mesh = model.geom(i).dataid[0]
        start = model.mesh_vertadr[mesh]
        stop = start + model.mesh_vertnum[mesh]
        points = model.mesh_vert[start:stop:3]
        return points @ data.geom_xmat[i].reshape(3, 3).T + data.geom_xpos[i]

    measured = np.array(state["hand_rad"]["right"])
    root = Path(__file__).resolve().parent
    taught = np.array(json.loads((root / "config/cube_station.json").read_text())[
        "hand_grasps_rad"]["teleop_pinch"])
    bounds_hand = np.array([(j.min_rad, j.max_rad) for j in hand_settings().joints])

    def shape(alpha):
        q = np.clip(measured + alpha * (taught - measured), bounds_hand[:, 0], bounds_hand[:, 1])
        data.qpos[addresses["right"]] = q
        mj.mj_forward(model, data)
        tips = np.array([data.site(f"right_{name}_tip").xpos for name in ("thumb", "index")])
        local_tips = (tips - wrist) @ rotation
        cloud = np.vstack([vertices(i) for i in hand_geoms])
        pads = {name: (vertices(i) - wrist) @ rotation for name, i in pad_geoms.items()}
        return q, local_tips, (cloud - wrist) @ rotation, pads

    candidates = []
    summary = []
    for alpha in (0.8, 1.0, 1.15, 1.3, 1.45, 1.6):
        closing, tips, closed_cloud, pads = shape(alpha)
        opening, _, open_cloud, _ = shape(max(0.0, alpha - 0.55))
        midpoint = tips.mean(axis=0)
        axis = tips[1] - tips[0]
        summary.append({"alpha": alpha, "tip_gap_mm": float(np.linalg.norm(axis) * 1000)})
        axis /= np.linalg.norm(axis)
        for heading in np.linspace(0, 2 * np.pi, 18, endpoint=False):
            horizontal = np.array([np.cos(heading), np.sin(heading), 0.0])
            horizontal -= (horizontal @ up) * up
            horizontal /= np.linalg.norm(horizontal)
            alignment = Rotation.align_vectors([horizontal], [rotation @ axis])[0].as_matrix()
            for roll in np.linspace(-np.pi, np.pi, 24, endpoint=False):
                target_rotation = Rotation.from_rotvec(roll * horizontal).as_matrix() @ alignment @ rotation
                for above in (0.0, 0.006, 0.012):
                    target = center + above * up - target_rotation @ midpoint
                    close_world = closed_cloud @ target_rotation.T + target
                    open_world = open_cloud @ target_rotation.T + target
                    floor = float(min(np.min(close_world @ up + plane_d),
                                      np.min(open_world @ up + plane_d)))
                    close_gap = float(np.min(np.linalg.norm(close_world - center, axis=1)) - radius)
                    open_gap = float(np.min(np.linalg.norm(open_world - center, axis=1)) - radius)
                    if floor < 0.010 or open_gap < 0.008 or not -0.002 <= close_gap <= 0.004:
                        continue
                    contacts = []
                    for pad in pads.values():
                        pad_world = pad @ target_rotation.T + target
                        offsets = pad_world - center
                        lengths = np.linalg.norm(offsets, axis=1)
                        nearest = int(np.argmin(lengths))
                        if not -0.002 <= lengths[nearest] - radius <= 0.004:
                            break
                        contacts.append(offsets[nearest] / lengths[nearest])
                    if len(contacts) != 2 or contacts[0] @ contacts[1] > -0.4:
                        continue
                    angle = Rotation.from_matrix(target_rotation @ rotation.T).magnitude()
                    score = np.linalg.norm(target - wrist) + 0.06 * angle
                    candidates.append({"score": score, "alpha": alpha, "wrist": target,
                                       "rotation": target_rotation, "opening": opening,
                                       "closing": closing, "floor": floor,
                                       "open_gap": open_gap, "close_gap": close_gap})
    print("PINCH_GEOMETRY", json.dumps(summary), flush=True)
    print("GEOMETRIC_CANDIDATES", len(candidates), flush=True)
    candidates.sort(key=lambda row: row["score"])
    limits = np.array([(j.min_deg, j.max_deg) for j in arm_settings().joints])
    ik = FastLeftIK(URDFKinematics(DEPLOYMENT / "assets/robot/marvin_dual_arm.urdf"), limits, "right")
    initial = np.deg2rad(state["arm_deg"]["right"])
    arm_addresses = [model.joint(f"Joint{i}_R").qposadr[0] for i in range(1, 8)]
    start_pose = ik.forward_jacobian(initial)[0]
    checked = []
    for index, candidate in enumerate(candidates[:36]):
        previous = start_pose.copy()
        q = initial.copy()
        paths = []
        failure = None
        for phase, offset in (("pregrasp", 0.065), ("grasp", 0.0), ("lift", 0.05)):
            goal = np.eye(4)
            goal[:3, :3] = candidate["rotation"]
            goal[:3, 3] = candidate["wrist"] + offset * up
            count = max(2, int(np.ceil(np.linalg.norm(goal[:3, 3] - previous[:3, 3]) / 0.003)))
            rotations = Slerp([0, 1], Rotation.from_matrix([previous[:3, :3], goal[:3, :3]]))
            path = [q.copy()]
            for fraction in np.linspace(0, 1, count + 1)[1:]:
                pose = previous.copy()
                pose[:3, :3] = rotations(fraction).as_matrix()
                pose[:3, 3] += fraction * (goal[:3, 3] - previous[:3, 3])
                try:
                    q = ik.solve(pose, q)
                except ValueError:
                    failure = f"{phase}_ik"
                    break
                path.append(q.copy())
            paths.append((phase, path))
            if failure:
                break
            previous = goal
        if failure:
            checked.append({"candidate": index, "failure": failure})
            continue
        table_gap, other_gap = float("inf"), float("inf")
        for phase, path in paths:
            hand_pose = (candidate["closing"] if phase == "lift" else
                         measured if phase == "pregrasp" else candidate["opening"])
            data.qpos[addresses["right"]] = hand_pose
            for q in path:
                data.qpos[arm_addresses] = q
                mj.mj_forward(model, data)
                cloud = np.vstack([vertices(i) for i in right_geoms])
                above_table = ((cloud[:, 1] < -0.20) & (cloud[:, 1] > -1.0)
                               & (np.abs(cloud[:, 0]) < 0.65))
                if above_table.any():
                    table_gap = min(table_gap, float(np.min(cloud[above_table] @ up + plane_d)))
                for a in right_geoms:
                    for b in left_geoms:
                        other_gap = min(other_gap, float(mj.mj_geomDistance(model, data, a, b, 0.1, None)))
        checked.append({"candidate": index, "floor_mm": table_gap * 1000,
                        "interarm_mm": other_gap * 1000})
        if table_gap < 0.010 or other_gap < 0.04:
            continue
        plan = {"source_run": str(run), "source_frame": obs["frame_id"], "side": "right",
                "cork_center": center.tolist(), "opening_rad": candidate["opening"].tolist(),
                "closing_rad": candidate["closing"].tolist(), "table_up": up.tolist(),
                "min_table_clearance_m": table_gap, "min_interarm_clearance_m": other_gap,
                "paths_deg": {phase: np.rad2deg(path).tolist() for phase, path in paths},
                "right_start_deg": state["arm_deg"]["right"],
                "left_start_deg": state["arm_deg"]["left"]}
        (run / "right_shuttle_plan.json").write_text(json.dumps(plan, indent=2))
        print("FEASIBLE_PLAN", json.dumps({"candidate": index, "floor_mm": table_gap * 1000,
                                           "interarm_mm": other_gap * 1000}), flush=True)
        break
    report = {"geometric_candidates": len(candidates), "checked": checked,
              "plan": str(run / "right_shuttle_plan.json") if (run / "right_shuttle_plan.json").exists() else None}
    (run / "right_shuttle_search.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
