"""Server-side Cartesian planning in dual_arm_mount, meters and xyzw quaternions."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .station import NoPlan


def cartesian_path(ik, measured_deg, vector, relative, *, verify_steps=True):
    q=np.deg2rad(np.asarray(measured_deg,dtype=float))
    initial=ik.forward_jacobian(q)[0]
    vector=np.asarray(vector,dtype=float)
    if vector.shape not in ((3,), (7,)) or not np.isfinite(vector).all():
        raise NoPlan("Expected xyz meters or xyz plus xyzw quaternion")
    if relative and vector.shape!=(3,):
        raise NoPlan("Relative motion uses xyz displacement in mount axes")
    target=initial.copy()
    target[:3,3]=initial[:3,3]+vector[:3] if relative else vector[:3]
    if vector.shape==(7,):
        if not np.isclose(np.linalg.norm(vector[3:]),1,atol=1e-4):
            raise NoPlan("Orientation quaternion must have unit norm")
        target[:3,:3]=Rotation.from_quat(vector[3:]).as_matrix()
    distance=np.linalg.norm(target[:3,3]-initial[:3,3])
    angle=Rotation.from_matrix(target[:3,:3]@initial[:3,:3].T).magnitude()
    count=max(1,int(np.ceil(max(distance/.0025,angle/np.deg2rad(1)))))
    if count>4000:
        raise NoPlan("Requested Cartesian path exceeds robot-scale planning range")
    rotations=Slerp([0,1],Rotation.from_matrix([initial[:3,:3],target[:3,:3]]))
    path=[np.rad2deg(q)]
    for fraction in np.linspace(0,1,count+1)[1:]:
        pose=initial.copy()
        pose[:3,3]=(1-fraction)*initial[:3,3]+fraction*target[:3,3]
        pose[:3,:3]=rotations(fraction).as_matrix()
        try:
            solved=ik.solve(pose,q)
        except ValueError as exc:
            raise NoPlan("IK rejected proposed Cartesian path") from exc
        joint_delta_deg = np.max(np.abs(np.rad2deg(solved - q)))
        if verify_steps:
            actual = ik.forward_jacobian(solved)[0]
            position_error = np.linalg.norm(actual[:3, 3] - pose[:3, 3])
            rotation_error = Rotation.from_matrix(actual[:3, :3] @ pose[:3, :3].T).magnitude()
            if joint_delta_deg > 1.5 or position_error > .0002 or rotation_error > .01:
                raise NoPlan("IK discontinuity or pose residual")
        subdivisions = max(1, int(np.ceil(joint_delta_deg / .4)))
        path.extend(np.rad2deg(v) for v in np.linspace(q,solved,subdivisions+1)[1:])
        q=solved
    return np.array(path),target
