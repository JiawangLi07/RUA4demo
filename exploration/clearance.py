"""Check the measured hand and sampled grasp trajectory in the mount frame."""
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog

from hardware import hand_settings
from .station import NoPlan


class HandClearance:
    def __init__(self, station, parked_arm, parked_hand, *, buffered=True):
        profile = station.clearance
        if not profile:
            raise NoPlan("No measured table and hand clearance profile for this station")
        self.normal = station.up
        self.height = float(profile["plane_height_m"])
        self.margin = float(profile["table_margin_m"])
        self.finger_margin = float(profile.get("finger_table_margin_m", self.margin))
        self.separation = float(profile["interarm_margin_m"])
        if (not np.isfinite([self.height, self.margin, self.finger_margin, self.separation]).all()
                or not 0 <= self.finger_margin <= self.margin
                or min(self.margin, self.separation) <= 0):
            raise NoPlan("Invalid hand clearance profile")
        self.buffered = buffered
        if not buffered:
            self.margin = self.finger_margin = self.separation = 0.
        model = mj.MjModel.from_xml_path(str(Path(__file__).resolve().parents[1] / "config/station_geometry.xml"))
        self.model = model
        self.data = mj.MjData(model)
        mj.mj_forward(model, self.data)
        mount = self.data.body("dual_arm_mount")
        if not (np.allclose(mount.xpos, 0) and np.allclose(mount.xmat.reshape(3, 3), np.eye(3))):
            raise NoPlan("Expected model coordinates in dual_arm_mount")
        names = [joint.name.split("_", 1)[1] for joint in hand_settings().joints]
        self.meshes = {}
        self.digit_meshes = set()
        self.opposite = []
        for side in ("left", "right"):
            suffix = side[0].upper()
            arm = [model.joint(f"Joint{i}_{suffix}").qposadr[0] for i in range(1, 8)]
            hand = [model.joint(f"{side}_{name}").qposadr[0] for name in names]
            active = side == station.arm
            if active:
                self.arm, self.hand = arm, hand
            else:
                self.data.qpos[arm] = np.deg2rad(parked_arm)
                self.data.qpos[hand] = parked_hand
            for i in range(model.ngeom):
                geom = model.geom(i)
                name = geom.name or ""
                if not name.endswith("_visual") or geom.type[0] != mj.mjtGeom.mjGEOM_MESH:
                    continue
                if not active and (name.startswith(side + "_") or name.endswith(f"_{suffix}_visual")):
                    self.opposite.append(i)
                if active and name.startswith(side + "_"):
                    mesh = geom.dataid[0]
                    offset = model.mesh_vertadr[mesh]
                    vertices = model.mesh_vert[offset:offset + model.mesh_vertnum[mesh]]
                    self.meshes[i] = vertices[ConvexHull(vertices).vertices]
                    if name.startswith(tuple(f"{side}_{digit}_" for digit in
                                             ("thumb", "index", "middle", "ring", "pinky"))):
                        self.digit_meshes.add(i)
        if not self.opposite or not self.meshes or not all(
            model.geom(f"{station.arm}_{digit}_elastomer_visual").id in self.meshes
            for digit in ("thumb", "index", "middle", "ring", "pinky")
        ):
            raise NoPlan("Incomplete hand geometry")

        self.pairs = self.nonadjacent_pairs()
        hand_indices = {address: index for index, address in enumerate(self.hand)}
        self.finger_dependencies = {}
        for geom in self.meshes:
            body = int(model.geom_bodyid[geom])
            indices = set()
            while body:
                start = model.body_jntadr[body]
                for joint in range(start, start + model.body_jntnum[body]):
                    address = model.jnt_qposadr[joint]
                    if address in hand_indices:
                        indices.add(hand_indices[address])
                body = int(model.body_parentid[body])
            self.finger_dependencies[geom] = indices
        self.error_bounds = None

    def set_pose(self, arm, hand):
        if not np.isfinite(arm).all() or not np.isfinite(hand).all():
            raise NoPlan("Nonfinite clearance input")
        self.data.qpos[self.arm] = np.deg2rad(arm)
        self.data.qpos[self.hand] = hand
        mj.mj_forward(self.model, self.data)

    def table_gaps(self):
        return np.array([(vertices @ (self.data.geom_xmat[i].reshape(3, 3).T @ self.normal)).min()
                         + self.data.geom_xpos[i] @ self.normal - self.height
                         for i, vertices in self.meshes.items()])

    def finger_error_bounds(self, radians):
        """Bound vertex displacement for bounded finger angle errors, wrist fixed."""
        bounds = []
        for i, vertices in self.meshes.items():
            world = vertices @ self.data.geom_xmat[i].reshape(3, 3).T + self.data.geom_xpos[i]
            body = int(self.model.geom_bodyid[i])
            last_anchor = None
            radius = total = 0.
            while body:
                for joint in reversed(range(self.model.body_jntadr[body],
                                            self.model.body_jntadr[body] + self.model.body_jntnum[body])):
                    if self.model.jnt_qposadr[joint] not in self.hand:
                        continue
                    anchor = self.data.xanchor[joint]
                    if last_anchor is None:
                        radius = np.linalg.norm(world - anchor, axis=1).max()
                    else:
                        radius += np.linalg.norm(last_anchor - anchor)
                    total += 2 * np.sin(radians / 2) * radius
                    last_anchor = anchor
                body = int(self.model.body_parentid[body])
            bounds.append(total)
        return np.asarray(bounds)

    def nonadjacent_pairs(self):
        # An intermediate virtual hinge may separate physically adjoining links.
        by_body = {int(self.model.geom_bodyid[i]) for i in self.meshes}
        pairs = []
        for i in self.meshes:
            body = int(self.model.geom_bodyid[i])
            parent = int(self.model.body_parentid[body])
            while parent and parent not in by_body:
                parent = int(self.model.body_parentid[parent])
            for j in self.meshes:
                if i >= j:
                    continue
                other = int(self.model.geom_bodyid[j])
                upper = int(self.model.body_parentid[other])
                while upper and upper not in by_body:
                    upper = int(self.model.body_parentid[upper])
                if body != other and body != upper and other != parent:
                    pairs.append((i,j))
        return pairs

    def audit_finger_path(self, arm, start, waypoints, radians):
        self.set_pose(arm, start)
        bounds = self.finger_error_bounds(radians)
        displacement = dict(zip(self.meshes, bounds))
        pairs = self.pairs
        minimum = float("inf")
        closest = None
        first = np.array(start)
        count = 0
        end_gap = None
        for end in [start, *waypoints]:
            end = np.asarray(end)
            steps = max(1,int(np.ceil(np.max(abs(end-first)) / .005)))
            for q in np.linspace(first, end, steps+1)[1:]:
                self.set_pose(arm, q)
                count += 1
                end_gap = float(min(self.table_gaps()-bounds))
                for i, j in pairs:
                    gap = mj.mj_geomDistance(self.model, self.data, i, j, .1, None)
                    lower = gap-displacement[i]-displacement[j]
                    if lower <= 0:
                        certified = self.separating_gap(i,j,radians)
                        if certified > 0:
                            lower = certified
                    if lower < minimum:
                        minimum = lower
                        closest = [self.model.geom(i).name, self.model.geom(j).name]
            first = end
        return {"checked_pairs":len(pairs),"sampled_poses":count,
                "assumed_joint_error_rad": radians, "pair_lower_bound_mm":minimum*1000,
                "limiting_pair":closest,"endpoint_table_lower_bound_mm":end_gap*1000,
                "endpoint_passes_with_assumed_error":bool(end_gap > self.margin),
                "scope":"fixed wrist; modeled non-adjacent hand links; angular error is a scenario, not a measured enable transient"}

    def separating_gap(self, i, j, radians):
        """Certify a separating plane over the same revolute joint error box."""
        def vertices(k):
            return self.meshes[k] @ self.data.geom_xmat[k].reshape(3,3).T + self.data.geom_xpos[k]

        def chain(k):
            result = []
            body = int(self.model.geom_bodyid[k])
            while body:
                first = self.model.body_jntadr[body]
                for joint in reversed(range(first, first + self.model.body_jntnum[body])):
                    if self.model.jnt_qposadr[joint] in self.hand:
                        result.append(joint)
                body = int(self.model.body_parentid[body])
            return result[::-1]

        a, b = vertices(i), vertices(j)
        # Maximize nominal separation with bounded plane normal; then verify its actual support.
        rows = np.r_[np.c_[a,-np.ones(len(a)),np.zeros(len(a))],
                     np.c_[-b,np.ones(len(b)),np.ones(len(b))]]
        sol = linprog([0,0,0,0,-1], A_ub=rows,b_ub=np.zeros(len(rows)),
                      bounds=[(-1,1)]*3+[(None,None)]*2, method="highs")
        if not sol.success or sol.x[-1] <= 0:
            return 0.
        normal=sol.x[:3]/np.linalg.norm(sol.x[:3])
        ca, cb = chain(i), chain(j)
        while ca and cb and ca[0] == cb[0]:
            ca, cb = ca[1:], cb[1:]

        def projected_error(v, joints):
            # Each fixed segment is rotated by its preceding joints. Its second
            # directional derivative is bounded by length * (sum |dq|)^2.
            # Taylor's remainder bounds nonlinear terms, including mixed joints.
            err = np.zeros(len(v))
            for k, joint in enumerate(joints):
                anchor = self.data.xanchor[joint]
                axis = self.data.xaxis[joint]
                err += radians * abs((v-anchor) @ np.cross(normal,axis))
                end = self.data.xanchor[joints[k+1]] if k+1 < len(joints) else v
                err += .5*((k+1)*radians)**2 * np.linalg.norm(end-anchor,axis=-1)
            return err

        return float(np.min(b@normal-projected_error(b,cb))-np.max(a@normal+projected_error(a,ca)))

    def other_arm_gap(self, i):
        cap = self.separation + .001
        return min(mj.mj_geomDistance(self.model, self.data, i, j, cap, None) for j in self.opposite)

    def report(self, arm, hand):
        self.set_pose(arm, hand)
        links = []
        for i, gap in zip(self.meshes, self.table_gaps()):
            other = self.other_arm_gap(i)
            links.append({"link": self.model.geom(i).name, "table_mm": round(float(gap)*1000, 3),
                          "other_arm_mm_capped": round(float(other)*1000, 3),
                          "passes": bool(gap > self.margin and other > self.separation)})
        return {"links": links, "limiting": min(links, key=lambda r:r["table_mm"]),
                "passed": all(row["passes"] for row in links), "table_margin_mm": self.margin*1000,
                "interarm_margin_mm": self.separation*1000,
                "scope": "modeled palm and digits; camera mount, cables and hidden contacts unobserved"}

    def pose(self, arm, hand, *, near_table=False, retreat_gaps=None, moving_fingers=None):
        self.set_pose(arm, hand)
        for i, gap in zip(self.meshes, self.table_gaps()):
            margin = self.finger_margin if near_table and i in self.digit_meshes else self.margin
            recovering = retreat_gaps is not None and i in retreat_gaps
            if recovering:
                margin = retreat_gaps[i]
            if gap <= 0 or (gap < margin - .00005 if recovering else gap <= margin):
                raise NoPlan(f"{self.model.geom(i).name} table clearance {gap:.4f} m <= {margin:.4f} m")
            for j in self.opposite:
                gap = mj.mj_geomDistance(self.model, self.data, i, j, self.separation + 0.001, None)
                if gap <= self.separation:
                    raise NoPlan(f"{self.model.geom(i).name} too close to {self.model.geom(j).name}")

        moving = None if moving_fingers is None else set(moving_fingers)
        if self.buffered and self.error_bounds is None and (moving is None or moving):
            self.error_bounds = dict(zip(self.meshes, self.finger_error_bounds(.015)))
        for i, j in self.pairs:
            if not self.buffered or (moving is not None and not moving.intersection(
                    self.finger_dependencies[i] ^ self.finger_dependencies[j])):
                # A shared rigid transform cannot change the distance in a pair.
                distance = mj.mj_geomDistance(self.model, self.data, i, j, .001, None)
                if distance <= 0:
                    raise NoPlan(f"Hand self collision: {self.model.geom(i).name} / {self.model.geom(j).name}")
                continue
            threshold = self.error_bounds[i] + self.error_bounds[j]
            distance = mj.mj_geomDistance(self.model, self.data, i, j, threshold + .001, None)
            if distance <= threshold and self.separating_gap(i,j,.015) <= 1e-6:
                raise NoPlan(f"Unverified self clearance: {self.model.geom(i).name} / {self.model.geom(j).name}")

    def path(self, paths, opened, closed):
        # The return follows the same configurations in reverse, including release.
        previous = np.r_[paths[0][0], opened]
        self.pose(previous[:7], previous[7:])
        for arm_path, fingers in ((paths[0], opened), (paths[1], opened),
                                  ([paths[1][-1]], closed), (paths[2], closed)):
            for arm in arm_path:
                target = np.r_[arm, fingers]
                delta = np.abs(target - previous)
                steps = max(1, int(np.ceil(max(delta[:7].max() / 0.25, delta[7:].max() / 0.01))))
                for q in np.linspace(previous, target, steps + 1)[1:]:
                    self.pose(q[:7], q[7:])
                previous = target
