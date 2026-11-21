import numpy as np
from scipy.spatial.transform import Rotation

from sim.isaac.handle_grasp import handle_sections
from sim.isaac.handle_contacts import handle_observations, tool_supported


def test_handle_sections_follow_the_thick_end_after_a_table_rotation():
    pos, theta = np.meshgrid(np.linspace(-0.075, 0.075, 100), np.linspace(0, 2 * np.pi, 48))
    radius = np.where(pos < -0.035, 0.004, 0.020)
    pts = np.c_[(radius * np.cos(theta)).ravel(), pos.ravel(), (0.74 + radius * np.sin(theta)).ravel()]
    for yaw in (0, 1.8, -0.9):
        rot = Rotation.from_euler("z", yaw)
        shifted = rot.apply(pts) + [0.61, -0.16, 0]
        origin, axis, _, bands = handle_sections(shifted)
        for s in (-0.061, 0.0, 0.04):
            pt = rot.apply([0, s, 0.74])[:2] + [0.61, -0.16]
            station = (pt - origin) @ axis
            assert bool(bands[0, 0] <= station <= bands[-1, 0]) == (s >= 0.0)


def test_two_digits_or_several_links_on_one_finger_cannot_pass():
    links = {"right_thumb_DP": 1.3, "right_index_MP": 0.8, "right_index_elastomer": 1.5, "right_hand_C_MC": 8}
    assert not tool_supported(links)
    links["right_middle_elastomer"] = 0.07
    assert not tool_supported(links)
    links["right_middle_elastomer"] = 0.12
    assert tool_supported(links)


def test_handle_audit_excludes_shaft_and_same_side_contacts_after_lift():
    grip = {
        "handle_axis_xy": [0, 1],
        "handle_origin_xy": [0, 0],
        "handle_bounds_m": [-0.03, 0.06],
        "handle_sections_m": [[-0.03, 0, 0.03, 0.016, 0.014], [0.06, 0, 0.03, 0.016, 0.014]],
        "patch_world_m": [0, 0, 0.03],
    }
    reference = [0, 0, 0.03, 1, 0, 0, 0]
    loads = {f"right_{d}_elastomer": 1 for d in ("thumb", "index", "middle", "ring")}
    pts = [[-0.016, 0, 0.13], [0.016, -0.01, 0.13], [0.016, 0.02, 0.13], [0.016, -0.06, 0.13]]
    row = {
        "object_pose_wxyz": [0, 0, 0.13, 1, 0, 0, 0],
        "link_normal_N": loads,
        "contact_surfaces": {k: [p, [1, 0, 0]] for k, p in zip(loads, pts)},
    }
    result = handle_observations([row], grip, reference)
    assert result["minimum_digits_on_handle"] == 3 and "ring" not in result["on_handle_digit_ticks"]
    assert result["opposed_on_handle_ticks"] == 1
    pts[0][0] = 0.014
    assert handle_observations([row], grip, reference)["opposed_on_handle_ticks"] == 0
    # An axial match inside a thin core or above the observed handle cannot count.
    pts[0][0] = -0.016
    pts[1][0] = 0.003
    pts[2][2] = 0.17
    result = handle_observations([row], grip, reference)
    assert result["on_handle_digit_ticks"] == {"thumb": 1}
