import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sim.isaac.head_wrist_surface import RegisteredSurface
from sim.isaac.paths import RIGHT_JOINTS
from sim.isaac.scene_objects import primitive_mesh
from sim.isaac.tabletop_catalog import load_registry
from sim.isaac.tabletop_control import Controller
from sim.isaac.tabletop_meshfit import HandContacts, gravity_residual
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


def test_load_on_one_finger_preserves_its_target_during_lift():
    c = Controller.__new__(Controller)
    c.hand = np.array([0.5, 0.6])
    c.io = SimpleNamespace(
        q=lambda: np.array([0.29, 0.60]),
        hand=[0, 1],
        model=SimpleNamespace(limits={"right": np.array([[0, 1], [0, 1]])}),
    )
    c.grip_guard = {
        "base": c.hand.copy(),
        "extra": np.zeros(2),
        "axis": np.array([0.3, 0.3]),
        "digits": [[0], [1]],
    }
    result = c.maintained_hand()
    assert result[0] == 0.5 and result[1] > 0.6
