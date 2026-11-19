from types import SimpleNamespace

import numpy as np

from exploration import live_cell, pinch_retry
from exploration.live_cell import CELL
from exploration.pinch_retry import rank_finger_probes, retry_pinch
from exploration.station import NoPlan


def decision(pattern, contacting=()):
    return {
        "confirmed": pattern == "bilateral",
        "reason": "bilateral_contact_confirmed" if pattern == "bilateral" else "bilateral_contact_not_sustained",
        "contact_pattern": pattern,
        "contacting_digits": list(contacting),
    }


class InPlaceRobot:
    def __init__(self):
        self.actions = []

    def capture_tactile_baseline(self):
        self.actions.append("baseline")

    def reopen_hand(self, *, keep_enabled):
        assert keep_enabled
        self.actions.append("open")

    def relocalize_for_regrasp(self):
        return self.locate_target()

    def locate_target(self):
        self.actions.append("locate")
        return "new live target"

    def prepare(self):
        self.actions.append("prepare")

    def approach(self):
        self.actions.append("approach")

    def close_hand(self):
        self.actions.append("close")
        return False

    def retry_grasp(self):
        return retry_pinch(decision("none"), self)["confirmed"]

    def align_in_place(self, target):
        assert target == "new live target"
        self.actions.append("align")

    def close_and_measure(self):
        self.actions.append("close")
        return decision("bilateral", ("thumb", "index"))

    def tighten_grasp(self):
        self.actions.append("tighten")
        return True

    def lift(self):
        self.actions.append("lift")


def test_cell_regrasp_relocalizes_before_alignment_and_lift():
    robot = InPlaceRobot()
    exec(CELL, {"robo": robot})
    assert robot.actions == ["baseline", "locate", "prepare", "approach", "close",
                             "open", "locate", "align", "close", "tighten", "lift"]


class FingerProbeRobot:
    candidates = (("right_index_PIP", .75), ("right_index_PIP", 1.5),
                  ("right_index_MCP_FE", .75), ("right_index_MCP_FE", 1.5))

    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.probes = []

    def next_probe(self, anchor, free, tried, *, blocked):
        assert not blocked
        assert (anchor, free) == ("thumb", "index")
        return next((candidate for candidate in self.candidates if candidate not in tried), None)

    def probe_free_finger(self, anchor, free, candidate):
        assert (anchor, free) == ("thumb", "index")
        self.probes.append(candidate)
        return next(self.outcomes)


def test_anchored_retry_only_probes_free_finger():
    initial = decision("single", ("thumb",))
    stalled = {**initial, "reason": "waypoint_tracking_stalled", "stalled_digits": ["index"]}
    robot = FingerProbeRobot([stalled, decision("bilateral", ("thumb", "index"))])
    result = retry_pinch(stalled, robot)
    assert result["confirmed"] and len(robot.probes) == 2


def test_anchor_loss_or_bad_feedback_aborts_without_a_second_probe():
    initial = decision("single", ("thumb",))
    for after in (decision("none"), {"confirmed": False, "reason": "stale_tactile_frames"}):
        robot = FingerProbeRobot([after])
        assert not retry_pinch(initial, robot)["confirmed"] and len(robot.probes) == 1


def test_unstable_contact_probes_both_digits_then_holds_the_stable_one():
    actions = []
    states = iter([decision("unstable"), decision("single", ("index",)),
                   decision("bilateral", ("thumb", "index"))])

    def select(anchor, free, tried, **kwargs):
        for joint in ("IP", "MCP_FE"):
            candidate = (f"right_{free}_{joint}", .75)
            if candidate not in tried:
                return candidate

    def probe(anchor, free, candidate):
        actions.append((anchor, free))
        return next(states)

    driver = SimpleNamespace(probe_free_finger=probe, next_probe=select)
    result = retry_pinch(decision("unstable"), driver)
    assert result["confirmed"]
    assert actions == [(None, "thumb"), (None, "index"), ("index", "thumb")]


def test_retry_can_succeed_after_more_than_three_probes():
    initial = decision("single", ("thumb",))
    robot = FingerProbeRobot([initial] * 3 + [decision("bilateral", ("thumb", "index"))])
    assert retry_pinch(initial, robot)["confirmed"] and len(robot.probes) == 4


def test_stall_without_contact_tries_other_joints_then_keeps_new_anchor():
    none = decision("none")
    index = decision("single", ("index",))
    stalled = {**none, "reason": "waypoint_tracking_stalled",
               "stalled_joints": {"right_thumb_CMC_AA": .015}, "stalled_digits": ["thumb"]}
    moves = []

    def next_probe(anchor, free, tried, *, blocked):
        assert blocked == {"right_thumb_CMC_AA"}
        assert free == "thumb"
        available = [(f"right_thumb_{joint}", .75) for joint in ("CMC_FE", "MCP_FE", "IP")]
        assert anchor == ("index" if len(tried) == 2 else None)
        return next(proposal for proposal in available if proposal not in tried)

    def probe(anchor, free, candidate):
        moves.append(candidate)
        return [none, index, decision("bilateral", ("thumb", "index"))][len(moves) - 1]

    def relieve(actual):
        assert actual == stalled
        return none

    robot = SimpleNamespace(relieve_stall=relieve, next_probe=next_probe, probe_free_finger=probe)
    result = retry_pinch(stalled, robot)
    assert result["confirmed"] and len(moves) == 3
    assert [step["action"] for step in result["retries"]] == ["relieve_stall"] + ["finger_probe"] * 3


def test_retry_stops_when_candidates_are_exhausted_or_no_closure_was_commanded():
    initial = decision("single", ("thumb",))
    robot = FingerProbeRobot([initial] * len(FingerProbeRobot.candidates))
    result = retry_pinch(initial, robot)
    assert not result["confirmed"] and result["retry_blocked"] == "no_clear_finger_probe"
    assert len(robot.probes) == len(FingerProbeRobot.candidates)
    result = retry_pinch({"confirmed": False, "reason": "awaiting_post_close_feedback"}, None)
    assert result["retries"] == []


def test_geometry_ranks_only_clear_moves_of_the_free_finger():
    class Geometry:
        model = SimpleNamespace(geom=lambda _: SimpleNamespace(id=0))
        meshes = {0: np.zeros((1, 3))}

        def __init__(self):
            self.data = SimpleNamespace(geom_xmat=[np.eye(3)], geom_xpos=np.zeros((1, 3)))

        def set_pose(self, arm, hand):
            self.data.geom_xpos[0, 0] = hand[1]

        def pose(self, arm, hand, *, near_table=False, moving_fingers=None):
            if hand[1] > .02:
                raise NoPlan("finger collision")
            self.set_pose(arm, hand)

    geometry = Geometry()
    moves = rank_finger_probes(
        geometry, [0] * 7, [0., 0.], ["right_thumb_IP", "right_index_PIP"],
        [(-1, 1), (-1, 1)], "index", [.05, 0, 0],
    )
    assert moves == [("right_index_PIP", .75)]
    assert geometry.data.geom_xpos[0, 0] == 0


def test_live_anchored_probe_needs_no_camera_or_sam3(tmp_path, monkeypatch):
    arm = SimpleNamespace(read_joint_position_deg=lambda _: [0.] * 7)
    hand = SimpleNamespace(
        cfg=SimpleNamespace(joints=[SimpleNamespace(name="right_thumb_IP", min_rad=-1, max_rad=1)]),
        read_state=lambda: SimpleNamespace(joint_pos=[0.]),
    )
    robot = live_cell.LiveGrasp({}, np.eye(4), None, arm, {"right": hand}, tmp_path,
                              lambda: False, None)
    robot.geometry = object()

    def rank(*args, **kwargs):
        assert args[-1] == "thumb"
        assert kwargs == {"target": None, "anchor": "index", "near_table": True, "blocked": (),
                          "inward_only": False}
        return [("right_thumb_IP", -.75), ("right_thumb_IP", .75)]

    monkeypatch.setattr(live_cell, "rank_finger_probes", rank)
    assert robot.next_finger_probe("index", "thumb", (("right_thumb_IP", -.75),)) == ("right_thumb_IP", .75)


def test_anchored_geometry_prefers_closer_pad_but_can_try_other_clear_moves(monkeypatch):
    model = SimpleNamespace(geom=lambda name: SimpleNamespace(id=0 if "thumb" in name else 1))
    geometry = SimpleNamespace(model=model, data=SimpleNamespace(q=0.))

    def pose(arm, hand, **kwargs):
        if hand[0] > .02:
            raise NoPlan("too close to table")
        geometry.data.q = hand[0]

    geometry.set_pose = geometry.pose = pose
    monkeypatch.setattr(pinch_retry.mj, "mj_geomDistance",
                        lambda m, d, free, anchor, cap, out: .03 - d.q)
    ranked = rank_finger_probes(
        geometry, [0.] * 7, [0., 0.], ["right_thumb_IP", "right_index_PIP"],
        [(-1., 1.), (-1., 1.)], "thumb", anchor="index", near_table=True,
    )
    assert ranked[0] == ("right_thumb_IP", .75)
    assert ("right_thumb_IP", -.75) in ranked
    assert ("right_thumb_IP", 1.5) not in ranked
