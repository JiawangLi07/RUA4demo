"""The enveloping score and in-episode retry transitions are distinct from fingertip closure."""

from types import SimpleNamespace

import numpy as np

from sim.isaac.direct_pregrasp import WristVisibility
from sim.isaac.head_wrist_trial import HeadWristTrial, Rejected
from sim.isaac.power_wrap import envelope_contacts, phalangeal_support, tactile_cylinder_center
from sim.isaac.retry_grasp import RetryGrasp, taper_bands
from sim.isaac.score import retry_distances


def test_wrap_requires_a_palm_and_distributed_loaded_phalanges():
    tips = {"right_thumb_elastomer": 1.0, "right_index_elastomer": 0.8, "right_middle_elastomer": 0.6}
    assert not envelope_contacts(tips)
    assert not envelope_contacts({**tips, "right_hand_C_MC": 1.2})
    genuine = {**tips, "right_hand_C_MC": 1.2, "right_index_MP": 0.25}
    assert envelope_contacts(genuine)
    assert not envelope_contacts({**genuine, "right_hand_C_MC": 0.0})
    assert not envelope_contacts(
        {"right_hand_C_MC": 1, "right_index_PP": 1, "right_index_MP": 2, "right_index_DP": 3}
    )


def test_side_taper_starts_near_the_smaller_end_in_both_directions():
    angle, x = np.meshgrid(np.linspace(0, 2 * np.pi, 49), np.linspace(-0.045, 0.045, 91))
    radius = 0.025 - (x + 0.045) * 0.15
    xyz = np.c_[x.ravel(), (radius * np.cos(angle)).ravel(), 0.744 + (radius * np.sin(angle)).ravel()]
    for sign in (-1, 1):
        points = xyz * np.array([sign, sign, 1]) + [0.64, -0.20, 0]
        bands = taper_bands(SimpleNamespace(world=points), 1)
        assert bands and (bands[0][1][0] - 0.64) * sign > 0.017
        assert 0.020 < bands[0][3] < 0.038


def test_retries_reobserve_from_the_moved_object_and_keep_all_attempts(tmp_path):
    class Trial(RetryGrasp):
        def object_pose(self):
            return np.array([self.scene.time, 0, 0.74, 1, 0, 0, 0])

        def palm(self):
            return np.eye(4)

        def actual(self):
            return np.zeros(22)

        def attempt_grasp(self):
            assert self.surface.fit is None
            assert self.approach_origin[0] == self.scene.time
            self.surface.fit = self.scene.time
            self.scene.time += 1
            self.scene.step_number += 240
            if self.attempt_number == 1:
                self.selected_grasp = SimpleNamespace(details={"posture": "tripod"})
                self.event("feedback_closure")
                raise Rejected("contact lost")
            assert self.used_postures == {"tripod"}
            self.selected_grasp = SimpleNamespace(details={"posture": "tip"})
            self.event("success")

        def recover(self, _palm, _hand):
            self.event("release_grip")
            self.scene.time += 0.5
            self.scene.step_number += 120
            self.event("reobservation_ready")

    trial = Trial.__new__(Trial)
    trial.scene = SimpleNamespace(time=0.0, step_number=0, settle=lambda: {"verification": {"passed": True}})
    trial.hand_ids = list(range(22))
    trial.surface = SimpleNamespace(fit=123)
    trial.out = tmp_path
    trial.cfg = {"use_wrist": True, "max_grasp_attempts": 3}
    trial.events = []
    trial.outcome = {}
    trial.held = set()
    trial.execute()
    assert [x["passed"] for x in trial.attempts] == [False, True]
    assert [x["first_step"] for x in trial.attempts] == [0, 360]
    assert (tmp_path / "attempt_01").is_dir() and (tmp_path / "attempt_02").is_dir()


def test_tactile_waist_uses_wall_contacts_and_rejects_cap_normals():
    prior = np.array([0.60, -0.21, 0.805])
    radius = 0.027
    centre = prior[:2] + [0.004, -0.002]
    contacts = {
        "palm": ([centre[0] - radius, centre[1], 0.81], [-1, 0, 0]),
        "index": ([centre[0], centre[1] + radius, 0.80], [0, -1, 0]),
        "top": ([0.61, -0.21, 0.84], [0, 0, 1]),
    }
    np.testing.assert_allclose(tactile_cylinder_center(contacts, prior, radius), centre)


def test_camera_behind_the_table_edge_cannot_use_hidden_points():
    camera = np.eye(4)
    camera[:3, 3] = [0.52, 0, 0.70]
    bounds = np.array([[0.49, -0.9, 0.69], [1.33, 0.6, 0.72]])
    K = np.array([[300, 0, 240], [0, 300, 320], [0, 0, 1]])
    points = np.array([[0.52, 0, 0.85], [0.53, 0.01, 0.86]])
    imager = WristVisibility(camera, K, {}, bounds)
    assert imager.fraction(np.eye(4), points) == 0
    camera[2, 3] = 0.74
    assert imager.fraction(np.eye(4), points) == 1


def test_local_retreat_uses_the_nearest_clear_reachable_view(monkeypatch):
    from sim.isaac import retry_grasp

    trial = RetryGrasp.__new__(RetryGrasp)
    trial.scene = SimpleNamespace(wrist_K=np.eye(3), table_bounds=None)
    trial.palm = lambda: np.eye(4)
    trial.wrist_optical = trial.palm
    trial.selected_view = np.eye(4)
    trial.selected_view[0, 3] = 0.20
    trial.surface = SimpleNamespace(world=np.zeros((50, 3)), clear=lambda shape, pose, margin: True)
    monkeypatch.setattr(retry_grasp, "hand_shapes", lambda scene, hand: {})
    monkeypatch.setattr(WristVisibility, "fraction", lambda self, pose, points: 10 * pose[0, 3])

    def reachable(start, goal):
        if np.isclose(goal[0, 3], 0.024):
            raise Rejected("unreachable")

    trial.joint_path = reachable
    pose, fraction = trial.local_retreat(np.zeros(22))
    np.testing.assert_allclose(pose[:3, 3], [0.036, 0, 0])
    assert fraction >= 0.21
    trial.surface.clear = lambda *args: False
    assert trial.local_retreat(np.zeros(22)) is None


def test_retry_distance_uses_measured_palm_and_interpolates_event_boundaries():
    home = np.eye(4)
    end, view = home.copy(), home.copy()
    end[0, 3], view[0, 3] = 0.09, 0.05
    view[:2, :2] = [[0, -1], [1, 0]]
    events = [dict(stage="move_to_pregrasp", start=home, attempt=1),
              dict(stage="attempt_failed", sim_time=0.5, attempt=1),
              dict(stage="reobservation_ready", sim_time=2.5, attempt=1, route="local", palm=end),
              dict(stage="pregrasp_arrived", sim_time=3.8, attempt=2, measured_palm=view),
              dict(stage="wrist_mesh", sim_time=4.0, attempt=2)]
    records = []
    for t, x in enumerate([0.14, 0.12, 0.08, 0.10, 0.05]):
        palm = home.copy()
        palm[0, 3] = x
        records.append(dict(time=t, palm=palm, goal=home))
    row, = retry_distances(events, records)
    assert row["route"] == "local"
    np.testing.assert_allclose([row["maximum_displacement_m"], row["path_length_m"],
                                row["end_distance_from_home_m"]], [0.05, 0.06, 0.09])
    assert np.isclose(row["maximum_displacement_through_next_wrist_m"], 0.08)
    assert np.isclose(row["turn_to_next_wrist_deg"], 90)


def test_recovery_keeps_the_hand_nearby_and_records_fallbacks(tmp_path):
    class Target:
        def __getitem__(self, key):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros(22)

    class Trial(RetryGrasp):
        def event(self, stage, **metadata):
            self.events.append(dict(stage=stage, attempt=1, **metadata))

        def palm(self):
            return self.location.copy()

        def move(self, stage, pose, hand, seconds, **options):
            self.event(stage)
            self.location = pose.copy()

        def observe(self, role):
            assert role == "head" and self.observation_dir.name == "after_release"

        def local_retreat(self, hand):
            return self.local

        def wait_for_rest(self):
            return next(self.free)

        def object_pose(self):
            return np.array([0, 0, 0, 1, 0, 0, 0])

    for route, free in (("local", [36]), ("previous_view", [0, 36]), ("home", [0, 0, 36])):
        trial = Trial.__new__(Trial)
        home = np.eye(4)
        trial.location = home.copy()
        trial.location[0, 3] = 0.20
        trial.scene = SimpleNamespace(target=Target())
        trial.observation_dir = tmp_path / route
        trial.observation_dir.mkdir()
        trial.attempt_number, trial.hand_ids, trial.held = 1, list(range(22)), {"thumb"}
        trial.events = [dict(attempt=1, stage="approach_contact_band")]
        trial.attempts = [dict(terminal_stage="test_lift_check")]
        trial.selected_view = trial.palm()
        trial.selected_view[0, 3] = 0.10
        trial.selected_grasp = SimpleNamespace(pre=trial.palm(), opened=np.zeros(22))
        trial.samples = [dict(hand_contacts=2, clearance_m=0.018)]
        nearby = trial.palm()
        nearby[0, 3] -= 0.025
        trial.local, trial.free = (nearby, 0.4), iter(free)
        trial.recover(home, np.zeros(22))
        stages = [e["stage"] for e in trial.events]
        assert stages.index("return_to_support") < stages.index("release_grip") < stages.index("local_retreat")
        assert ("clear_for_reobservation" in stages) == (route == "home")
        end = trial.events[-1]
        assert end["route"] == route and end["free_ticks"] == 36 and not trial.held
        if route == "local":
            assert np.isclose(end["distance_from_failure_m"], 0.025)
            assert np.isclose(end["distance_from_home_m"], 0.175)


def test_phalangeal_probe_requires_thumb_opposition_and_several_loaded_digits():
    distributed = {
        "right_thumb_elastomer": 1.0,
        "right_index_MP": 1.0,
        "right_ring_MP": 1.2,
        "right_middle_elastomer": 1.0,
    }
    assert phalangeal_support(distributed)
    assert not envelope_contacts(distributed)
    assert not phalangeal_support({k: v for k, v in distributed.items() if "thumb" not in k})
    assert not phalangeal_support({k.replace("_MP", "_elastomer"): v for k, v in distributed.items()})


def test_loaded_thumb_base_relaxes_only_until_the_palm_shares_load():
    trial = HeadWristTrial.__new__(HeadWristTrial)
    trial.cfg = {"regulate_power_lift": True, "share_palm_load": True}
    trial.dt = 1 / 240
    trial.digit_indices = {"thumb": [0, 1]}
    trial.loaded_hand_reference = np.array([0.85, 0.30])
    trial.scene = SimpleNamespace(
        hand_names={"right": ["right_thumb_CMC_FE", "right_thumb_MCP_FE"]},
        limits={"right": np.array([[-0.17, 1.92], [-0.52, 1.40]])},
    )
    loads = {"right_thumb_MC": 19.0, "right_thumb_elastomer": 1.0}
    trial.samples = [{"link_normal_N": loads}]
    target = trial.loaded_hand_reference.copy()
    trial.regulate_loaded_fingers(target)
    assert target[0] < trial.loaded_hand_reference[0] and target[1] > trial.loaded_hand_reference[1]
    loads["right_hand_C_MC"] = 2.0
    shared = target.copy()
    trial.regulate_loaded_fingers(target)
    np.testing.assert_array_equal(target, shared)
