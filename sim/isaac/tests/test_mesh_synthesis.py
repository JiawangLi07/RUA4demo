import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sim.isaac.head_wrist_surface import RegisteredSurface
from sim.isaac.paths import RIGHT_JOINTS
from sim.isaac.scene_objects import primitive_mesh
from sim.isaac.tabletop_catalog import geometry_family, load_registry
from sim.isaac.tabletop_control import swept_clear
from sim.isaac.tabletop_meshfit import HandContacts, closure_path, gravity_residual
from sim.isaac.tabletop_registry import prepare


def training_catalog(path):
    transform = {"offset_bbox": [0.2, -0.7, 0.6], "rotation": np.eye(3).tolist()}
    knot = {k: transform for k in ("pregrasp", "grasp", "squeeze")}
    knot.update({k: [0.0] * 22 for k in ("q_pre", "q_grasp", "q_squeeze")})
    knot.update(
        mode="training",
        shape="box_like",
        up=[0, 0, 1],
        validated=True,
        bbox_dims=[0.07, 0.04, 0.03],
    )
    types = {
        str(i): {
            "roles": ["right_thumb_elastomer", "right_index_elastomer"],
            "samples": [knot],
        }
        for i in range(36)
    }
    with gzip.open(path, "wt") as file:
        json.dump(
            {
                "schema": "rua.shared-hand-postures.v1",
                "joint_names": RIGHT_JOINTS,
                "types": types,
            },
            file,
        )


def test_metric_mesh_registry_is_independent_of_names_and_rejects_saved_poses(tmp_path):
    training = tmp_path / "training.gz"
    training_catalog(training)
    v, f = primitive_mesh({"kind": "box", "size_m": [0.057, 0.040, 0.033]})
    original = tmp_path / "shaped.npz"
    np.savez(
        original,
        visual_vertices=v,
        visual_faces=f,
        visual_colors=[[0.1, 0.2, 0.3]],
        collision_count=1,
        collision_000_vertices=v,
        collision_000_faces=f,
    )
    renamed = tmp_path / "bottle.npz"
    renamed.write_bytes(original.read_bytes())
    first, second = tmp_path / "first", tmp_path / "second"
    assert prepare([original], training, first, None, "z", 0.08) == prepare(
        [renamed], training, second, None, "z", 0.1
    )
    registry = first / "registry.json"
    objects, shared = load_registry(registry)
    key = next(iter(objects))
    assert set(objects[key]) == {"prior"} and key.startswith("part_")
    bundle = {"visual": (v, f, None), "collisions": [(v, f)]}
    s = RegisteredSurface(bundle, sample_faces=True)
    assert len(s.local) > 500
    s.world = s.local + [0, 0, -s.local[:, 2].min()]
    s.table = 0
    _, before = shared.query(s, np.eye(4))
    s.world[:, 0] *= 1.8
    _, after = shared.query(s, np.eye(4))
    assert len(before) == len(after) == 36 and not np.allclose(
        before[0]["grasp"], after[0]["grasp"]
    )
    data = json.loads(registry.read_bytes())
    data["objects"][key]["seeds"] = ["cached object pose"]
    registry.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="only a metric mesh"):
        load_registry(registry)


def test_opposed_friction_cones_support_lift_but_one_wall_does_not():
    def contact(x, n):
        return {"goal": np.array([x, 0.0, 0.0]), "normal": np.array([n, 0.0, 0.0])}

    opposed = [contact(-0.02, -1), contact(0.02, 1)]
    single_side = [contact(-0.02, -1), contact(-0.019, -1)]
    assert gravity_residual(opposed, np.zeros(3)) < 0.001
    assert gravity_residual(single_side, np.zeros(3)) > 0.5


def test_contact_jacobian_tracks_each_joint_and_wrist_rotation():
    from scipy.spatial.transform import Rotation

    h = HandContacts.__new__(HandContacts)
    origin = np.eye(4)
    origin[:3, 3] = [0.005, 0.019, 0.013]
    h.chains = {
        "pad": [
            SimpleNamespace(
                name="finger",
                origin=origin,
                kind="revolute",
                axis=np.array([0.0, 1.0, 0.0]),
            )
        ]
    }
    h.columns = {"finger": 3}
    case = {
        "link": "pad",
        "anchor": np.array([0.013, 0.002, 0.032]),
        "hand_normal": np.array([1.0, 0.0, 0.0]),
    }
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec([0.12, 0.40, -0.18]).as_matrix()
    q = np.zeros(22)
    q[3] = 0.4
    pt, n, j, jn = h.point_jac(case, pose, q)
    eps = 1e-6
    for ix in (0, 4, 9):
        m = pose.copy()
        v = q.copy()
        if ix < 3:
            m[ix, 3] += eps
        elif ix < 6:
            m[:3, :3] = (
                Rotation.from_rotvec(np.eye(3)[ix - 3] * eps).as_matrix() @ m[:3, :3]
            )
        else:
            v[ix - 6] += eps
        new, normal, _, _ = h.point_jac(case, m, v)
        np.testing.assert_allclose((new - pt) / eps, j[:, ix], atol=1e-6)
        np.testing.assert_allclose((normal - n) / eps, jn[:, ix], atol=1e-6)


def test_closure_is_computed_from_surface_width_without_sensor_input():
    # Two independent sliding pads with analytically known FK and opposite normals.
    def point_jac(site, palm, q):
        index = int("index" in site["link"])
        sign = 2 * index - 1
        jac = np.zeros((3, 8))
        jac[0, 6 + index] = -sign * 0.04
        point = np.array([sign * (0.04 - 0.04 * q[index]), 0.0, 0.05])
        return point, -site["normal"], jac, np.zeros_like(jac)

    hand = SimpleNamespace(point_jac=point_jac)
    results = []
    for width in (0.04, 0.06):
        sites = [
            {
                "link": "right_" + digit + "_elastomer",
                "goal": np.array([sign * width / 2, 0, 0.05]),
                "normal": np.array([sign, 0, 0]),
                "weight": 1,
            }
            for digit, sign in (("thumb", -1), ("index", 1))
        ]
        path, preload, gaps = closure_path(
            hand, np.eye(4), np.zeros(2), sites, np.array([[0, 1], [0, 1]]), 0.0
        )
        assert np.all((path >= 0) & (path <= 1))
        assert np.abs(np.diff(path, axis=0)).max() <= 0.0750001
        assert all(abs(gaps[k] + d) < 0.0002 for k, d in preload.items())
        results.append(path[-1])
    assert np.all(results[0] > results[1] + 0.12)


def test_width_sections_detect_a_sparsely_meshed_shaft():
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    tool, rod = [], []
    for x in np.linspace(0, 0.16, 90):
        radius = 0.004 if x < 0.06 else 0.016
        angles = theta if x < 0.06 else np.tile(theta, 12)
        tool.extend(
            np.c_[
                np.full(len(angles), x),
                radius * np.cos(angles),
                0.016 + radius * np.sin(angles),
            ]
        )
        rod.extend(
            np.c_[
                np.full(len(theta), x),
                0.009 * np.cos(theta),
                0.009 + 0.009 * np.sin(theta),
            ]
        )
    assert (
        geometry_family(SimpleNamespace(world=np.asarray(tool), table=0.0)) == "handled"
    )
    assert (
        geometry_family(SimpleNamespace(world=np.asarray(rod), table=0.0)) == "slender"
    )


def test_observation_transfer_checks_occupied_space_between_safe_endpoints():
    q0 = np.array([0.0, 0.0])
    q1 = np.array([0.6, 0.1])
    visited = []

    def free(q, f):
        visited.append((q.copy(), f))
        return abs(q[0] - 0.3) > 0.04

    assert free(q0, 0) and free(q1, 1)
    assert not swept_clear([q0, q1], free)
    assert any(0.4 < fraction < 0.6 for _, fraction in visited)
    assert swept_clear([q0, [0.1, 0.3], [0.12, 0.4]], lambda q, f: q[0] < 0.2)


def test_handle_balance_uses_component_volumes():
    from sim.isaac.tabletop_handle import volume_center

    box, _ = primitive_mesh({"kind": "box", "size_m": [0.01, 0.01, 0.01]})
    large = 2 * box + [0.09, 0, 0]
    np.testing.assert_allclose(volume_center([box, large]), [0.08, 0, 0], atol=1e-9)


def test_one_visible_end_cannot_certify_five_centimetres(monkeypatch):
    from sim.isaac import tabletop_vision

    world = np.array(
        [
            [x, y, z]
            for x in np.linspace(-0.075, 0.075, 21)
            for y in (-0.008, 0, 0.008)
            for z in (0, 0.008, 0.016)
        ]
    )
    rest, held = np.eye(4), np.eye(4)
    held[2, 3] = 0.055
    color = tabletop_vision.appearance(np.full((1, 3), 120, np.uint8))

    def evaluate(samples):
        rgb = np.full(samples.shape, 120, np.uint8)
        monkeypatch.setattr(
            tabletop_vision, "foreground", lambda *args: (0.0, samples, rgb, None)
        )
        return tabletop_vision.match_height(None, None, 0.0, world, rest, held, color)

    raised = world + held[:3, 3]
    end = evaluate(raised[raised[:, 0] > 0.035])
    assert end["carried_pixels"] >= 12 and "geometry_clearance_m" not in end
    assert evaluate(raised)["geometry_clearance_m"] == pytest.approx(0.055)
