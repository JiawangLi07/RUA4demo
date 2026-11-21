"""The enveloping score and in-episode retry transitions are distinct from fingertip closure."""

from types import SimpleNamespace

import numpy as np

from sim.isaac.direct_pregrasp import WristVisibility
from sim.isaac.head_wrist_trial import HeadWristTrial, Rejected
from sim.isaac.power_wrap import envelope_contacts, phalangeal_support, tactile_cylinder_center
from sim.isaac.retry_grasp import RetryGrasp, taper_bands


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
