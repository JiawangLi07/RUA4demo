"""Non-actuating local geometry and hand calibration diagnostics."""
import mujoco as mj
import numpy as np
from scipy.spatial import ConvexHull

from hardware import hand_settings


class TableNotVisible(ValueError):
    """The frame does not provide enough tabletop depth for a fresh plane fit."""


def fixed_arm_escape(geometry, arm, hand):
    """Search bounded finger changes with a stationary wrist; check sampled segments."""
    start = np.asarray(hand, dtype=float)
    limits = np.array([(j.min_rad, j.max_rad) for j in hand_settings().joints])
    if not np.all((start >= limits[:, 0]) & (start <= limits[:, 1])):
        return {"found": False, "reason": "measured fingers outside configured joint limits"}
    lower = np.maximum(limits[:, 0], start - .15)
    upper = np.minimum(limits[:, 1], start + .15)
    geometry.set_pose(arm, start)
    gaps = geometry.table_gaps()
    before = float(min(gaps))
    if before <= 0 or any(geometry.other_arm_gap(i) <= geometry.separation for i in geometry.meshes):
        return {"found": False, "reason": "start intersects modeled table or violates interarm margin",
                "start_min_mm": before*1000, "motion_issued": False}
    reserve = geometry.finger_error_bounds(.015)
    desired = geometry.margin + reserve
    current = start.copy()
    path = []
    # Squared deficits reward simultaneous improvements on several limiting digits.
    def score(g):
        return np.linalg.norm(np.maximum(desired + .0005 - g, 0))
    for _ in range(32):
        if np.all(gaps > desired):
            break
        best = score(gaps)
        selected = None
        for index in range(len(current)):
            for sign in (-1, 1):
                trial = current.copy()
                trial[index] += sign*.025
                if not lower[index] <= trial[index] <= upper[index]:
                    continue
                geometry.set_pose(arm, trial)
                end = geometry.table_gaps()
                if score(end) >= best - .00001:
                    continue
                # Track all low links through each sampled segment.
                floor = np.minimum(gaps, geometry.margin)
                safe = True
                for q in np.linspace(current, trial, 5)[1:]:
                    geometry.set_pose(arm, q)
                    if (np.any(geometry.table_gaps() < floor - 1e-8) or
                            any(geometry.other_arm_gap(i) <= geometry.separation for i in geometry.meshes)):
                        safe = False
                        break
                if safe:
                    selected, best = (trial, end), score(end)
        if selected is None:
            break
        current, gaps = selected
        path.append(current.tolist())
    return {"found": bool(min(gaps) > geometry.margin), "arm_fixed_deg": list(arm),
            "path_audit":geometry.audit_finger_path(arm, start, path, .015),
            "start_min_mm": before*1000, "end_min_mm": float(min(gaps))*1000,
            "joint_waypoints_rad": path, "steps":len(path),
            "scope":"sampled table and other-arm geometry; no self-contact, attachment or enable-drift validation",
            "motion_issued":False}


def workspace(backend, box, config):
    box = np.asarray(box, dtype=float)
    snap = backend.snapshot
    height, width = snap.rgb.shape[:2]
    x1,y1,x2,y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Expected a valid local region in RGB pixel coordinates")
    uv, xyz = snap.geometry.register(snap.raw_depth)
    inside = (uv[:,0]>=x1)&(uv[:,0]<x2)&(uv[:,1]>=y1)&(uv[:,1]<y2)
    t = backend.extrinsics
    p = xyz[inside] @ t[:3,:3].T + t[:3,3]
    up = np.asarray(config["up_axis_mount"])
    h = config["clearance"]["plane_height_m"]
    q = p[np.abs(p@up-h) < .012]
    for _ in range(3):
        if len(q) < 100:
            raise TableNotVisible("Insufficient tabletop returns near the calibrated height")
        c = q.mean(0)
        _,singular,axes = np.linalg.svd(q-c, full_matrices=False)
        if singular[1] / np.sqrt(len(q)) < .01:
            raise TableNotVisible("Table samples lack two-dimensional support")
        n = axes[-1] if axes[-1]@up > 0 else -axes[-1]
        if n@up < .995:
            raise ValueError("Observed plane differs from the calibrated tabletop normal")
        q = q[np.abs((q-c)@n)<.004]
    if len(q)<100:
        raise TableNotVisible("Insufficient planar inliers")
    # Project only observed returns into a 1 cm grid. Holes are deliberately unknown.
    cells = np.unique(np.floor(q[:,:2]/.01).astype(int),axis=0)
    raised = p[(p-c)@n > .008]
    occupied = np.unique(np.floor(raised[:,:2]/.01).astype(int),axis=0)
    hull = ConvexHull(q[:,:2])
    return {"frame_id":backend.frame_id,"source":"registered RGB-D","frame":"dual_arm_mount",
            "local_roi_xyxy":box.tolist(),"plane_normal":n.tolist(),"plane_height_m":float(c@n),
            "rms_mm":float(np.sqrt(np.mean(((q-c)@n)**2))*1000),"inliers":len(q),
            "observed_xy_bounds_m":[q[:,:2].min(0).tolist(),q[:,:2].max(0).tolist()],
            "observed_hull_m2":float(hull.volume),"grid_size_m":.01,
            "table_cells_xy":cells.tolist(),"raised_cells_xy":occupied.tolist(),
            "scope":"surface support and raised returns; missing/occluded cells unknown; no verified 3D motion volume"}


def tcp_report(geometry, arm, hand, side, config):
    geometry.set_pose(arm, hand)
    ids=[geometry.model.geom(f"{side}_{digit}_elastomer_visual").id for digit in ("thumb","index")]
    pair=np.zeros(6)
    aperture=mj.mj_geomDistance(geometry.model,geometry.data,*ids,.5,pair)
    if not 0 < aperture < .5:
        raise ValueError("No distinct modeled opposing pad points")
    mid=pair.reshape(2,3).mean(axis=0)
    tcp=geometry.data.site(f"{side}_tcp").xpos
    return {"frame":"dual_arm_mount","modeled_pad_mid_m":mid.tolist(),
            "modeled_pad_points_m":pair.reshape(2,3).tolist(),
            "modeled_tcp_minus_mid_m":(tcp-mid).tolist(),"modeled_pad_gap_mm":float(aperture*1000),
            "configured_grasp_offset_m":config.get("tcp_offset_m"),
            "measurement":"requires an independently observed grip center and target surface at aligned grasp; model-only vector is not a measured grasp offset"}


def pad_measurement(snapshot, boxes, geometry, arm, hand, side):
    """Compare two image-selected depth patches to the nearest modeled pad surfaces."""
    if len(boxes)!=8:
        raise ValueError("Supply thumb and index boxes in x1,y1,x2,y2 order")
    geometry.set_pose(arm,hand)
    measurements=[]
    for digit, box in zip(("thumb","index"),(boxes[:4],boxes[4:])):
        point=snapshot.locate(snapshot.frame_id,digit,box)
        if not point.get("found") or point.get("inlier_samples",0)<8:
            raise ValueError(f"No reliable depth for {digit}")
        p=np.array(point["point_mount_m"])
        i=geometry.model.geom(f"{side}_{digit}_elastomer_visual").id
        v=geometry.meshes[i]@geometry.data.geom_xmat[i].reshape(3,3).T+geometry.data.geom_xpos[i]
        measurements.append({"digit":digit,"point_m":p.tolist(),"samples":point["inlier_samples"],
                             "depth_mad_m":point["depth_mad_m"],"nearest_mesh_vertex_m":float(np.linalg.norm(v-p,axis=1).min())})
    middle=np.mean([r["point_m"] for r in measurements],axis=0)
    tcp=geometry.data.site(side+"_tcp").xpos
    return {"frame_id":snapshot.frame_id,"patches":measurements,"visible_midpoint_m":middle.tolist(),
            "tcp_minus_visible_midpoint_m":(tcp-middle).tolist(),"grasp_offset_verified":False,
            "scope":"open hand visible surfaces, one view; the contact faces and target-to-TCP offset at grasp require aligned contact evidence"}


def preview_approach(snapshot, box, geometry, arm, hand, side, config):
    from hardware import preview_reach
    surface=snapshot.locate(snapshot.frame_id,"target",box)
    if not surface.get("found"):
        raise ValueError("No usable target depth")
    tcp=tcp_report(geometry,arm,hand,side,config)
    modeled_offset=np.array(tcp["modeled_tcp_minus_mid_m"])
    dest=np.array(surface["point_mount_m"])+modeled_offset+.065*np.array(config["up_axis_mount"])
    start=geometry.data.site(side+"_tcp").xpos.copy()
    q=np.array(arm)
    n=max(1,int(np.ceil(np.linalg.norm(start-dest)/.0025)))
    checked=0
    reason=None
    for p in np.linspace(start,dest,n+1)[1:]:
        try:
            geometry.pose(q,hand)
            ik=preview_reach(side,p,q)
            nxt=np.array(ik["target_joint_deg"])
            if np.max(abs(nxt-q))>1.5 or ik["position_residual_m"]>.0002:
                raise ValueError("IK discontinuity or residual")
            for sample in np.linspace(q,nxt,max(2,int(np.ceil(max(abs(nxt-q))/.25))+1))[1:]:
                geometry.pose(sample,hand)
        except (ValueError,RuntimeError) as exc:
            reason=str(exc)
            break
        q=nxt
        checked+=1
    return {"frame_id":snapshot.frame_id,"model_pregrasp_m":dest.tolist(),"checked_segments":checked,
            "segments":n,"model_path_passed":checked==n,"reason":reason,"physically_verified":False,
            "scope":"fixed wrist orientation; geometry and IK only, pending measured grasp offset and 3D corridor"}
