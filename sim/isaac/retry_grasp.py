"""Repeat physical grasp attempts in a single native episode with fresh observations."""

import numpy as np

from .direct_pregrasp import DirectGrasp, WristVisibility, path_clear
from .head_wrist_geometry import hand_shapes
from .head_wrist_trial import FINGERS, Rejected, save


def taper_bands(surface, attempt):
    """Cross sections measured from the narrower end of a side-lying tapered object."""
    xyz = surface.world
    _, vec = np.linalg.eigh(np.cov(xyz[:, :2].T))
    along = vec[:, -1]
    normal = np.array([-along[1], along[0]])
    t = xyz[:, :2] @ along
    lo, hi = np.quantile(t, [0.006, 0.994])
    f = (t - lo) / (hi - lo)
    widths = [np.ptp(np.quantile(xyz[np.abs(f - u) < 0.09, :2] @ normal, [0.03, 0.97])) for u in (0.15, 0.85)]
    if widths[0] > widths[1]:
        f = 1 - f
    locations = ((0.17, 0.25, 0.34), (0.17, 0.28, 0.38), (0.53, 0.65, 0.38))
    bands = []
    for rank, u in enumerate(locations[min(attempt - 1, 2)]):
        strip = xyz[np.abs(f - u) < 0.065]
        if len(strip) < 15:
            continue
        low, high = np.quantile(strip[:, 2], [0.025, 0.975])
        z = low + 0.55 * (high - low)
        mid = strip[np.abs(strip[:, 2] - z) < max(0.003, 0.2 * (high - low))]
        if len(mid) < 8:
            continue
        a, b = np.quantile(mid[:, :2] @ normal, [0.02, 0.98])
        xy = np.median(mid[:, :2] @ along) * along + ((a + b) / 2) * normal
        bands.append((rank * 0.003, np.r_[xy, z], normal, b - a, high, low))
    return bands


class RetryGrasp(DirectGrasp):
    def event(self, stage, **details):
        if stage in ("head_mesh", "head_exposure") and self.observation_dir.name == "after_release":
            details["observation_subdir"] = "after_release"
        super().event(stage, attempt=getattr(self, "attempt_number", 0), **details)

    def contact_bands(self):
        if self.cfg.get("side_lying_taper"):
            return taper_bands(self.surface, self.attempt_number)
        return None

    def options(self, role, coarse=None):
        if self.cfg.get("multi_finger_handle"):
            from .handle_grasp import handle_grasps

            return handle_grasps(self, coarse)
        if self.family == "side_wall" and (
            self.cfg.get("whole_palm", True) or "cylindrical_side" in self.used_postures
        ):
            from .power_wrap import wrap_grasps

            options = wrap_grasps(self, coarse)
            return options
        return super().options(role, coarse)

    def close_and_lift(self, grasp, qcontact, qclose, low, hi):
        if self.cfg.get("require_palm_hold") and not self.selected_grasp.details.get("power_wrap"):
            raise Rejected("this attempt requires distributed palm contact")
        if self.selected_grasp.details.get("tool_handle"):
            from .handle_grasp import close_handle

            close_handle(self, grasp, qcontact)
        elif self.selected_grasp.details.get("power_wrap"):
            from .power_wrap import close_envelope

            close_envelope(self, grasp, qcontact, qclose)
        else:
            super().close_and_lift(grasp, qcontact, qclose, low, hi)

    def object_pose(self):
        return self.scene.objects[self.object_name].data.root_state_w[0, :7].cpu().numpy().copy()

    def local_retreat(self, hand):
        """Back along the last observed approach and prefer a D405 view on that short path."""
        start = self.palm()
        to_view = self.selected_view[:3, 3] - start[:3, 3]
        length = np.linalg.norm(to_view)
        shape = hand_shapes(self.scene, hand)
        k = self.scene.wrist_K.copy()
        k[:2, 2] -= 0.5
        camera = WristVisibility(np.linalg.solve(start, self.wrist_optical()), k, shape, self.scene.table_bounds)
        points = self.surface.world[::max(1, len(self.surface.world) // 170)]
        choices = []
        for distance in np.unique(np.minimum([0.0, 0.012, 0.024, 0.036, 0.048], length)):
            goal = start.copy()
            goal[:3, 3] += to_view * distance / max(length, 1e-8)
            if not path_clear(self.surface, start, goal, [shape] * 6):
                continue
            try:
                self.joint_path(start, goal)
            except Rejected:
                continue
            visible = camera.fraction(goal, points)
            # If this hand blocks the camera, retain room to rotate into the next hand shape.
            choices.append(((visible < 0.21, -min(visible, 0.21), distance), goal, visible))
        if choices:
            _, goal, visible = min(choices, key=lambda row: row[0])
            return goal, visible
        return None

    def wait_for_rest(self):
        s = self.scene
        self.event("await_free_rest")
        arm = s.target[0, self.arm_ids].cpu().numpy().copy()
        hand = s.target[0, self.hand_ids].cpu().numpy().copy()
        goal = self.palm()
        stable = 0
        for _ in range(round(2.0 / self.dt)):
            row = self.command(arm, hand, goal, allow_contact=True)
            r = row["right"]
            free = (
                not r["contact_count"]
                and r["clearance_m"] < 0.005
                and r["object_linear_speed_m_s"] < 0.012
                and r["object_angular_speed_rad_s"] < 0.20
            )
            stable = stable + 1 if free else 0
            if stable >= round(0.15 / self.dt):
                return stable
        return 0

    def recover(self, start_palm, start_hand):
        """Unload before opening. Return home only when a nearby observation cannot be made."""
        grip = self.selected_grasp
        failed_palm = self.palm()
        self.retry_route = "local"
        fallback = None
        if grip is not None:
            hand = self.scene.target[0, self.hand_ids].cpu().numpy().copy()
            last = self.samples[-1] if self.samples else {}
            if last.get("hand_contacts") and last.get("clearance_m", 0) > 0.005:
                down = self.palm()
                down[2, 3] -= min(last["clearance_m"], 0.13)
                self.move("return_to_support", down, hand, 0.7, allow_contact=True)
            self.held.clear()
            entered = any(
                e["attempt"] == self.attempt_number and e["stage"] == "approach_contact_band" for e in self.events
            )
            if entered:
                # If this leg stopped early its planned endpoint may be inside the object.
                reached = self.attempts[-1]["terminal_stage"] != "approach_contact_band"
                self.move("release_grip", grip.pre if reached else self.palm(), grip.opened, 0.72, allow_contact=True)
            elif last.get("hand_contacts"):
                self.move("release_grip", self.palm(), grip.opened, 0.72, allow_contact=True)
            # The object may have shifted. Preserve both original exposures for this attempt.
            saved_dir = self.observation_dir
            self.observation_dir = saved_dir / "after_release"
            self.observation_dir.mkdir()
            try:
                self.observe("head")
            except Rejected as exc:
                fallback = str(exc)
            finally:
                self.observation_dir = saved_dir
            if fallback is None:
                hand = self.scene.target[0, self.hand_ids].cpu().numpy().copy()
                choice = self.local_retreat(hand)
                if choice is None:
                    fallback = "no clearance along the short retreat"
                else:
                    goal, visible = choice
                    self.event("local_retreat_selected", predicted_visible_fraction=float(visible))
                    if np.linalg.norm(goal[:3, 3] - self.palm()[:3, 3]) > 0.002:
                        self.move("local_retreat", goal, hand, 0.45, allow_contact=True)
        stable = self.wait_for_rest() if fallback is None else 0
        if not stable and grip is not None:
            self.retry_route = "previous_view"
            self.event("retry_wider_view", reason=fallback or "object still moving or touching the hand")
            self.move("withdraw_hand", self.selected_view, grip.opened, 0.72, allow_contact=True)
            stable = self.wait_for_rest()
            if not stable:
                self.retry_route = "home"
                self.move("clear_for_reobservation", start_palm, start_hand, 0.90, allow_contact=True)
                stable = self.wait_for_rest()
        if not stable:
            raise Rejected("object did not settle free of the hand after release")
        end = self.palm()
        self.event(
            "reobservation_ready",
            route=self.retry_route,
            free_ticks=stable,
            palm=end.tolist(),
            distance_from_failure_m=float(np.linalg.norm(end[:3, 3] - failed_palm[:3, 3])),
            distance_from_home_m=float(np.linalg.norm(end[:3, 3] - start_palm[:3, 3])),
            object_pose_wxyz=self.object_pose().tolist(),
        )

    def execute(self):
        if not self.cfg.get("use_wrist"):
            raise Rejected("multi-attempt grasp requires the wrist camera")
        maximum = int(self.cfg["max_grasp_attempts"])
        if not 1 <= maximum <= 8:
            raise ValueError("max_grasp_attempts must be between 1 and 8")
        if not self.scene.settle()["verification"]["passed"]:
            raise Rejected("initial contact-free settling failed")
        self.event("settled", object_pose_wxyz=self.object_pose().tolist())
        self.used_postures = set()
        self.retry_route = None
        self.attempts = []
        self.outcome["attempts"] = self.attempts
        start_palm = self.palm()
        start_hand = self.actual()[self.hand_ids]
        for number in range(1, maximum + 1):
            self.attempt_number = number
            self.observation_dir = self.out / f"attempt_{number:02d}"
            self.observation_dir.mkdir()
            self.selected_grasp = None
            self.selected_view = None
            self.held.clear()
            self.streak = dict.fromkeys(FINGERS, 0)
            self.surface.fit = None
            self.approach_origin = self.object_pose()[:3]
            first_step = self.scene.step_number
            self.event("attempt_begin", object_pose_wxyz=self.object_pose().tolist(), step=first_step)
            fail = None
            try:
                self.attempt_grasp()
            except Rejected as exc:
                fail = str(exc)
            posture = self.selected_grasp.details["posture"] if self.selected_grasp is not None else None
            info = dict(
                number=number,
                posture=posture,
                passed=fail is None,
                reason=fail or "held for 2 s",
                terminal_stage=self.stage,
                first_step=first_step,
                last_step=self.scene.step_number,
                final_object_pose_wxyz=self.object_pose().tolist(),
            )
            self.attempts.append(info)
            save(self.out / "attempts.json", self.attempts)
            if fail is None:
                return
            self.event("attempt_failed", reason=fail, failed_stage=info["terminal_stage"], posture=posture)
            if posture:
                self.used_postures.add(posture)
            if (
                "invalid state/table penetration" in fail
                or number == maximum
                or (posture is None and not self.cfg.get("side_lying_taper"))
            ):
                raise Rejected(fail)
            self.recover(start_palm, start_hand)
        raise Rejected("attempt budget exhausted")
