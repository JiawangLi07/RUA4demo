"""Deterministic feedback fixture for the cube trial and revision pipeline."""

from .grasp_program import OFFSET_LIMIT_M, verdict


class OfflineTrials:
    grasps = ("pinch", "wide")

    def __init__(self, *, single_cell=False):
        self.single_cell = single_cell

    def run(self, program, out, layout, *, cancelled):
        robo = FixtureRobot(layout, cancelled)
        samples = program.run(robo, single_cell=self.single_cell,
                              path=out / "grasp_cube.py" if self.single_cell else None)
        return {
            **verdict(robo.before, samples),
            "source": "offline_fixture",
            "before_m": robo.before,
            "lift_samples_m": samples,
            "contact": robo.contact,
            "reset_ok": True,
            "commands": robo.commands,
        }


class FixtureRobot:
    def __init__(self, layout, cancelled):
        self.before = (0.42, round(0.004 * layout - 0.006, 3), 0.14)
        self.cancelled = cancelled
        self.commands = []
        self.contact = {}

    def _step(self, name):
        if self.cancelled():
            raise RuntimeError("Trial connection closed")
        self.commands.append(name)

    def prepare(self, offset_m, lift_m, grasp):
        self._step("prepare")
        self.offset_m, self.lift_m, self.grasp = offset_m, lift_m, grasp

    def locate_target(self):
        self._step("locate_target")
        return self.before

    def approach(self):
        self._step("approach")

    def confirm_target(self):
        self._step("confirm_target")

    def descend(self):
        self._step("descend")

    def close_hand(self):
        self._step("close_hand")
        dx, dy, dz = self.offset_m
        x_error = 0.008 - dx
        self.contact = {
            "lateral_contact_side": "center" if abs(x_error) <= 0.003 else ("positive_x" if x_error > 0 else "negative_x"),
            "closure": "centered" if abs(x_error) <= 0.003 else "off_center",
        }
        self.attached = abs(x_error) <= 0.003 and abs(dy) <= 0.004 and abs(dz) <= 0.004

    def lift(self):
        self._step("lift")
        x, y, z = self.before
        if not self.attached:
            return [self.before, self.before]
        return [(x, y, z + self.lift_m),
                (x, y, z + self.lift_m - 0.001) if self.grasp == "pinch" else self.before]


def scripted_learner(request):
    """Use the recorded error in the fixture, including the cold start."""
    program = {**request["previous"]}
    if request["evidence"]:
        trial = request["evidence"][-1]
        side = trial["contact"]["lateral_contact_side"]
        step = {"positive_x":1, "center":0, "negative_x":-1}[side] * OFFSET_LIMIT_M * 2 / 3
        program["offset_m"] = [max(-OFFSET_LIMIT_M,min(OFFSET_LIMIT_M,program["offset_m"][0] + step)),
                               program["offset_m"][1], 0.0]
        program["grasp"] = "pinch"
        program["recheck"] = True
        reason = f"Try a lateral offset following the contact side after {trial['reason']}."
    else:
        reason = "Start at the localized center with the configured basic hand pose."
    return {"program": program, "reason": reason}
