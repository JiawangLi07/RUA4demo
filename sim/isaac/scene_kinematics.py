"""Host-side URDF FK/IK; no simulator, mesh loading, policies, or recorded actions.

Transforms map link coordinates into the URDF root frame (not an assumed world
frame). Lengths are meters, angles radians. URDF origins use Rz(yaw) Ry(pitch)
Rx(roll); joint axes are expressed in that origin's frame. Geometry, dynamics,
safety-controller limits and actuator limits are deliberately not interpreted.
The caller supplies the station URDF path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from os import PathLike
from xml.etree import ElementTree as ET

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .scene_fixture import rigid_transform


@dataclass(frozen=True, slots=True)
class PoseSolution:
    """Actual FK residuals, never a substituted target; evaluations counts FK calls.

    ``converged`` means both requested geometric tolerances are met, not merely
    that scipy stopped. Failure is local non-convergence, not proof of global
    unreachability. The mapping preserves every supplied non-active seed value.
    """

    joint_positions: dict[str, float]
    position_error_m: float
    rotation_error_rad: float
    converged: bool
    evaluations: int


@dataclass(frozen=True, slots=True)
class _Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: NDArray[np.float64]
    axis: NDArray[np.float64]
    limits: tuple[float, float]


def _one(element: ET.Element, tag: str, required: bool = False) -> ET.Element | None:
    matches = element.findall(tag)
    if len(matches) > 1 or (required and not matches):
        raise ValueError(f"{element.get('name', element.tag)}: expected one {tag}")
    return matches[0] if matches else None


def _attribute(element: ET.Element, name: str) -> str:
    value = element.get(name)
    if value is None or not value.strip():
        raise ValueError(f"{element.tag}: missing {name}")
    return value


def _vector(value: str) -> NDArray[np.float64]:
    result = np.array([float(part) for part in value.split()], dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("URDF xyz/rpy/axis requires three finite numbers")
    return result


def _scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name}: finite numeric value required")
    return float(value)


class KinematicTree:
    """Validated, single-root URDF tree with lazily cached root-to-link chains.

    Construct with :meth:`from_urdf`. All moving joints on a requested chain
    require explicit positions: omitted values never silently become zero.
    Fixed joint positions may be omitted, or supplied as zero. Unknown names
    are errors. FK accepts measured positions outside limits; IK rejects such
    seeds rather than silently clipping fingers or other non-active joints.
    """

    def __init__(self, links: set[str], joints: dict[str, _Joint], root_link: str):
        self.root_link = root_link
        self._links = frozenset(links)
        self._joints = joints
        self._parents = {joint.child: joint for joint in joints.values()}
        self._chains: dict[str, tuple[_Joint, ...]] = {root_link: ()}

    @classmethod
    def from_urdf(cls, path: str | PathLike[str]) -> KinematicTree:
        """Parse XML data only, rejecting ambiguous topology and mimic joints.

        Fixed-joint zero axes are legal and ignored (as in the source scene).
        Moving axes default to URDF's +X; explicit nonzero axes are normalized.
        Revolute/prismatic joints require finite ordered lower/upper limits;
        continuous joints have no positional bounds, fixed joints have [0, 0].
        """
        try:
            robot = ET.parse(path).getroot()
        except ET.ParseError as exc:
            raise ValueError(f"malformed URDF XML: {exc}") from exc
        if robot.tag != "robot":
            raise ValueError("URDF root must be robot")
        links: set[str] = set()
        for element in robot.findall("link"):
            name = _attribute(element, "name")
            if name in links:
                raise ValueError(f"duplicate link: {name}")
            links.add(name)
        if not links:
            raise ValueError("URDF has no links")
        joints: dict[str, _Joint] = {}
        parents: dict[str, str] = {}
        for element in robot.findall("joint"):
            name = _attribute(element, "name")
            if name in joints:
                raise ValueError(f"duplicate joint: {name}")
            kind = _attribute(element, "type")
            if kind not in {"fixed", "revolute", "continuous", "prismatic"}:
                raise ValueError(f"{name}: unsupported joint type {kind}")
            if element.find("mimic") is not None:
                raise ValueError(f"{name}: mimic joints are unsupported")
            parent_element = _one(element, "parent", required=True)
            child_element = _one(element, "child", required=True)
            assert parent_element is not None and child_element is not None
            parent = _attribute(parent_element, "link")
            child = _attribute(child_element, "link")
            if parent not in links or child not in links:
                raise ValueError(f"{name}: missing parent/child link")
            if child in parents:
                raise ValueError(f"{child}: multiple parent joints")
            parents[child] = parent
            origin_element = _one(element, "origin")
            origin = np.eye(4)
            if origin_element is not None:
                origin[:3, 3] = _vector(origin_element.get("xyz", "0 0 0"))
                origin[:3, :3] = Rotation.from_euler("xyz", _vector(origin_element.get("rpy", "0 0 0"))).as_matrix()
            axis_element = _one(element, "axis")
            axis = _vector(_attribute(axis_element, "xyz") if axis_element is not None else "1 0 0")
            if kind != "fixed":
                # Scale before normalization to avoid overflow for finite axes.
                scale = float(np.max(np.abs(axis)))
                if scale == 0:
                    raise ValueError(f"{name}: zero joint axis")
                axis /= scale
                axis /= np.linalg.norm(axis)
            limit = _one(element, "limit", required=kind in {"revolute", "prismatic"})
            bounds = (0.0, 0.0) if kind == "fixed" else (-np.inf, np.inf)
            if kind in {"revolute", "prismatic"}:
                assert limit is not None
                bounds = (float(_attribute(limit, "lower")), float(_attribute(limit, "upper")))
                if not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
                    raise ValueError(f"{name}: invalid position limits")
            joints[name] = _Joint(name, kind, parent, child, origin, axis, bounds)
        roots = links - parents.keys()
        if len(roots) != 1:
            raise ValueError("URDF must have exactly one root; disconnected or cyclic tree")
        root = next(iter(roots))
        connected = {root}
        for link in links:
            visiting: set[str] = set()
            current = link
            while current not in connected:
                if current in visiting:
                    raise ValueError(f"cycle in chain at {current}")
                visiting.add(current)
                if current not in parents:
                    raise ValueError(f"missing chain to root for {current}")
                current = parents[current]
            connected.update(visiting)
        return cls(links, joints, root)

    def _chain(self, link_name: str) -> tuple[_Joint, ...]:
        if link_name not in self._links:
            raise ValueError(f"unknown link: {link_name}")
        if link_name not in self._chains:
            reverse = []
            current = link_name
            while current != self.root_link:
                joint = self._parents[current]
                reverse.append(joint)
                current = joint.parent
            self._chains[link_name] = tuple(reversed(reverse))
        return self._chains[link_name]

    def _positions(self, positions: Mapping[str, float], bounded: bool = False) -> dict[str, float]:
        if not isinstance(positions, Mapping):
            raise ValueError("joint_positions must be a mapping")
        result = {}
        for name, raw in positions.items():
            if name not in self._joints:
                raise ValueError(f"unknown joint: {name}")
            joint = self._joints[name]
            value = _scalar(raw, name)
            if joint.kind == "fixed" and value != 0:
                raise ValueError(f"{name}: fixed joint position must be zero")
            if bounded and not joint.limits[0] <= value <= joint.limits[1]:
                raise ValueError(f"{name}: seed outside native position limits")
            result[name] = value
        return result

    @staticmethod
    def _require_chain(chain: tuple[_Joint, ...], positions: Mapping[str, float]) -> None:
        for joint in chain:
            if joint.kind != "fixed" and joint.name not in positions:
                raise ValueError(f"missing joint position: {joint.name}")

    @staticmethod
    def _pose(chain: tuple[_Joint, ...], positions: Mapping[str, float]) -> NDArray[np.float64]:
        transform = np.eye(4)
        for joint in chain:
            transform = transform @ joint.origin
            if joint.kind == "prismatic":
                transform[:3, 3] += transform[:3, :3] @ (joint.axis * positions[joint.name])
            elif joint.kind in {"revolute", "continuous"}:
                transform[:3, :3] = (
                    transform[:3, :3] @ Rotation.from_rotvec(joint.axis * positions[joint.name]).as_matrix()
                )
        return transform

    def pose(self, link_name: str, joint_positions: Mapping[str, float]) -> NDArray[np.float64]:
        """Return the root-from-link 4x4 transform; do not clamp measured q."""
        chain = self._chain(link_name)
        positions = self._positions(joint_positions)
        self._require_chain(chain, positions)
        return self._pose(chain, positions)

    def joint_limits(self, names: Sequence[str]) -> NDArray[np.float64]:
        """Return ordered Nx2 URDF bounds (rad or m); continuous is +/-inf."""
        if isinstance(names, str) or len(set(names)) != len(names):
            raise ValueError("joint names must be a sequence without duplicates")
        for name in names:
            if name not in self._joints:
                raise ValueError(f"unknown joint: {name}")
        return np.array([self._joints[name].limits for name in names], dtype=float).reshape(-1, 2)

    def solve_pose(
        self,
        link_name: str,
        target_4x4: object,
        seed_positions: Mapping[str, float],
        active_joint_names: Sequence[str],
        position_tolerance_m: float = 0.002,
        rotation_tolerance_rad: float = 0.02,
    ) -> PoseSolution:
        """Deterministic bounded local IK, changing only named chain joints.

        Residuals are translation and SO(3) log error, each divided by its
        tolerance. One scipy TRF solve uses at most 200 objective evaluations
        (finite-difference Jacobian calls additionally counted in the result).
        No restarts, collision checking, motion planning, or execution occur.
        Empty/locked active sets still return honest FK errors. Supply all
        fingers/other-arm values in the seed to retain them in the result.
        """
        target = rigid_transform(target_4x4, "target_4x4")
        pos_tol = _scalar(position_tolerance_m, "position_tolerance_m")
        rot_tol = _scalar(rotation_tolerance_rad, "rotation_tolerance_rad")
        if pos_tol <= 0 or rot_tol <= 0:
            raise ValueError("pose tolerances must be positive")
        chain = self._chain(link_name)
        positions = self._positions(seed_positions, bounded=True)
        self._require_chain(chain, positions)
        bounds = self.joint_limits(active_joint_names)
        movable = {joint.name for joint in chain if joint.kind != "fixed"}
        if any(name not in movable for name in active_joint_names):
            raise ValueError("active joints must be moving joints on the target chain")
        free = [index for index, (lower, upper) in enumerate(bounds) if lower < upper]
        names = [active_joint_names[index] for index in free]
        evaluations = 0

        def errors(values: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            nonlocal evaluations
            positions.update(zip(names, map(float, values)))
            actual = self._pose(chain, positions)
            evaluations += 1
            return (
                actual[:3, 3] - target[:3, 3],
                Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec(),
            )

        def residual(values: NDArray[np.float64]) -> NDArray[np.float64]:
            translation, rotation = errors(values)
            return np.concatenate((translation / pos_tol, rotation / rot_tol))

        values = np.array([positions[name] for name in names])
        if names:
            solution = least_squares(
                residual,
                values,
                bounds=(bounds[free, 0], bounds[free, 1]),
                method="trf",
                max_nfev=200,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            values = solution.x
        translation, rotation = errors(values)
        position_error = float(np.linalg.norm(translation))
        rotation_error = float(np.linalg.norm(rotation))
        return PoseSolution(
            positions,
            position_error,
            rotation_error,
            position_error <= pos_tol and rotation_error <= rot_tol,
            evaluations,
        )
