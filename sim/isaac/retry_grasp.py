"""Repeat physical grasp attempts in a single native episode with fresh observations."""

import numpy as np

from .direct_pregrasp import DirectGrasp
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

    def recover(self, start_palm, start_hand):
        """Return a loaded object to support, unload, then retract through the observed approach."""
        s = self.scene
        grip = self.selected_grasp
        if grip is not None:
            hand = s.target[0, self.hand_ids].cpu().numpy().copy()
            last = self.samples[-1] if self.samples else {}
            if last.get("hand_contacts") and last.get("clearance_m", 0) > 0.005:
                down = self.palm()
                down[2, 3] -= min(last["clearance_m"], 0.13)
                self.move("return_to_support", down, hand, 0.7, allow_contact=True)
            self.held.clear()
            if any(
                e["attempt"] == self.attempt_number and e["stage"] == "approach_contact_band" for e in self.events
            ):
                self.move("release_grip", grip.pre, grip.opened, 0.72, allow_contact=True)
                self.move("withdraw_hand", self.selected_view, grip.opened, 0.72, allow_contact=True)
            self.move("clear_for_reobservation", start_palm, start_hand, 0.90, allow_contact=True)
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
                self.event("reobservation_ready", object_pose_wxyz=self.object_pose().tolist(), free_ticks=stable)
                return
        raise Rejected("object did not settle free of the hand after release")

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
