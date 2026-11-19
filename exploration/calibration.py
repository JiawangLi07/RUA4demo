"""One bounded, unloaded finger calibration experiment on the right-hand rig."""
from contextlib import ExitStack
import fcntl
import json
import base64
from io import BytesIO
from PIL import Image
import time
from types import SimpleNamespace

import numpy as np

from hardware import ROOT, DEPLOYMENT, hardware_owners, open_state_sources, read_state
from rgbd import camera_stream
from robot import SnapshotBackend
from task_library import atomic_json
from .clearance import HandClearance
from .hand_enable import await_enable_state
from .station import validate_rig


def require(value, message):
    if not value:
        raise RuntimeError(message)


def index_trial(config, reference, out, cancelled, *, dialogue=None):
    """Preload measured angles, test a sub-degree finger excursion, disable and measure."""
    require(config["arm"] == "right", "Probe implemented for the right Sharpa only")
    validate_rig(config["camera_serial"], config["rig"])
    out.mkdir()
    report = {"status":"blocked","arm_commands":0,"step_sent":False,"loaded_contact_test":False}
    phase = "acquire"
    try:
        with ExitStack() as resources:
            for path in (ROOT/".capture.lock", DEPLOYMENT/"runs/.active.lock"):
                lock = resources.enter_context(path.open("a"))
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            require(not any(hardware_owners().values()), "Rig occupied")
            from tianji_command_adapter import ReliableTianjiRobot
            from hardware import arm_settings
            arm_config=arm_settings()
            arm_config.vel_ratio=arm_config.acc_ratio=5
            arm, hands = open_state_sources(resources, ReliableTianjiRobot(arm_config), tactile_sides=("right",))
            camera = resources.enter_context(camera_stream(config["camera_serial"]))
            snap = camera()
            state = read_state(arm,hands,snap.monotonic_ns,("right",))
            require(state["joints_aligned"], "RGB-D and joints are not aligned")
            dest = out/"before"
            dest.mkdir()
            SnapshotBackend(snap,state,reference.extrinsics,dest)
            q = np.array(state["hand_rad"]["right"])
            hand = hands["right"]
            sdk = hand._hand
            armed = False
            def get(call):
                err, value = call()
                hand._raise_if_error(err, call.__name__)
                return value
            def set_value(call,*args):
                hand._raise_if_error(call(*args), call.__name__)
            require(not get(sdk.get_enable_state), "Hand already enabled")
            geometry = HandClearance(SimpleNamespace(arm="right",up=np.array(config["up_axis_mount"]),
                                     clearance=config["clearance"]),state["arm_deg"]["left"],state["hand_rad"]["left"])
            start = state["arm_deg"]["right"]
            require(geometry.report(start,q)["passed"], "Start table or interarm clearance failed")
            require(np.max(abs(q-reference.joints["hand_rad"]["right"])) < .015, "Hand moved since capture")
            require(np.mean(np.max(abs(snap.rgb.astype(float)-reference.snapshot.rgb.astype(float)),axis=2)>35)<.015,
                    "Scene changed since agent observation")
            require(all(v["cur_state"]==0 and v["err_code"]==0 for v in state["arm_status"].values()), "Arm not idle")
            cutoff = time.monotonic()+12
            def guard():
                image=camera()
                changed=np.max(abs(image.rgb.astype(float)-snap.rgb.astype(float)),axis=2)>35
                require(changed.mean()<.015,"Scene changed during calibration")
                require(not cancelled() and time.monotonic()<cutoff, "Cancelled or timed out")
                for side,aid in (("left","A"),("right","B")):
                    st=arm.read_arm_state(aid,side)
                    require(st.err_code==0 and st.cur_state==0 and
                            np.max(abs(np.rad2deg(st.joint_pos)-state["arm_deg"][side]))<.15,"Arm moved or faulted")
            def release():
                set_value(sdk.set_enable_state,False)
                await_enable_state(lambda:get(sdk.get_enable_state), False, lambda:None)
                hand._enabled=False
                report["disabled_confirmed"]=True
            index = [j.name for j in hand.cfg.joints].index("right_index_MCP_FE")
            target=q.copy()
            target[index]-=np.deg2rad(.75)
            bounds=np.array([(j.min_rad,j.max_rad) for j in hand.cfg.joints])
            require(np.all((q>=bounds[:,0])&(q<=bounds[:,1])&(target>=bounds[:,0])&(target<=bounds[:,1])),
                    "Finger outside configured travel")
            seq=None
            def joints():
                nonlocal seq
                until=time.monotonic()+.25
                while time.monotonic()<until:
                    st=hand.read_state()
                    if st.sequence!=seq:
                        seq=st.sequence
                        v=np.asarray(st.joint_pos)
                        require(v.shape==(22,) and np.isfinite(v).all(),"Invalid hand readback")
                        return v
                    time.sleep(.008)
                raise RuntimeError("Stale hand feedback")
            frames={}
            def forces():
                rows=[]
                for ch in range(5):
                    until=time.monotonic()+.25
                    while True:
                        f=hand.read_tactile_frame(ch,.04)
                        if f is not None and f.frame_id!=frames.get(ch):
                            break
                        require(time.monotonic()<until,"Stale fingertip feedback")
                        time.sleep(.003)
                    frames[ch]=f.frame_id
                    rows.append(f.f6)
                v=np.asarray(rows)
                require(v.shape==(5,6) and np.isfinite(v).all(),"Invalid fingertip feedback")
                return v
            baseline=np.stack([forces() for _ in range(8)])
            center=np.median(baseline,axis=0)
            def monitor():
                guard()
                latest=forces()
                # Static sensor variation is evidence, not a calibrated contact-stop limit.
                report["tactile_observation"] = {
                    "baseline_f6": center.tolist(),
                    "measured_f6": latest.tolist(),
                    "delta_f6": (latest-center).tolist(),
                    "contact_stop_calibrated": False,
                }
                return joints()
            try:
                phase="protected_preload"
                guard()
                require(get(sdk.get_control_mode)==hand._symbols["ControlMode"].POSITION,"Hand not in position mode")
                # Own cleanup before transferring control, including partial setter failures.
                armed=True
                hand._enabled=True
                source=get(sdk.get_control_source)
                speed=float(get(sdk.get_speed_coeff))
                current=float(get(sdk.get_current_coeff))
                report["as_found"]={"source":str(source),"speed_coeff":speed,"current_coeff":current}
                require(source==hand._symbols["ControlSource"].SDK,"Hand ownership is not SDK")
                require(0 < speed <= .150001 and 0 < current <= .200001,
                        "Current speed/current settings exceed unloaded commissioning settings")
                if not sdk.is_collision_protection_enabled():
                    set_value(sdk.enable_collision_protection,True)
                require(sdk.is_collision_protection_enabled(),"Protection not confirmed")
                require(not get(sdk.get_enable_state),"Unexpected enable after protection configuration")
                phase="measured_pose_preload"
                set_value(sdk.set_joint_position,q.tolist(),True)
                require(not get(sdk.get_enable_state) and np.max(abs(monitor()-q))<.015,"Preload drift")
                phase="unloaded_path"
                for v in np.linspace(q,target,5): geometry.pose(start,v)
                phase="enable_readback"
                names=[j.name for j in hand.cfg.joints]
                log=resources.enter_context((out/"enable-feedback.jsonl").open("w",buffering=1))
                t0=time.monotonic()
                report["preload_rad"]=q.tolist()
                report["joint_names"]=names
                def held_sample():
                    actual=monitor()
                    delta=actual-q
                    i=int(np.argmax(abs(delta)))
                    log.write(json.dumps({"ms":1000*(time.monotonic()-t0),"sequence":seq,
                                          "joint_rad":actual.tolist(),"delta_rad":delta.tolist()})+"\n")
                    if abs(delta[i])>report.get("peak_drift_rad",-1):
                        report.update(peak_drift_rad=float(abs(delta[i])),peak_drift_joint=names[i])
                    require(np.all((actual>=bounds[:,0])&(actual<=bounds[:,1])), "Enabled hand outside joint bounds")
                    require(abs(delta[i])<=np.deg2rad(3), f"Enable excursion exceeds 3 degrees: {names[i]}")
                    geometry.pose(start,actual)
                    return actual
                set_value(sdk.set_enable_state,True)
                await_enable_state(lambda:get(sdk.get_enable_state),True,held_sample)
                stable=[]
                until=time.monotonic()+.8
                while True:
                    actual=held_sample()
                    stable.append((time.monotonic(),actual.copy()))
                    while len(stable)>1 and stable[-1][0]-stable[1][0]>=.12:
                        stable.pop(0)
                    if (len(stable)>=3 and stable[-1][0]-stable[0][0]>=.12 and
                            np.max(np.ptp(np.stack([v for _,v in stable]),axis=0))<=.003):
                        break
                    require(time.monotonic()<until,"Hand did not settle after enabling")
                    time.sleep(.01)
                # The deployed origin routine also takes its start measurement after acquisition.
                q=stable[-1][1]
                report["settled_rad"]=q.tolist()
                target=q.copy()
                target[index]-=np.deg2rad(.75)
                require(np.all((target>=bounds[:,0])&(target<=bounds[:,1])),"Settled probe outside finger bounds")
                for candidate in np.linspace(q,target,5): geometry.pose(start,candidate)
                hold_low=q-.015
                hold_high=q+.015
                evidence=[]
                report["steps"]=evidence
                motion=resources.enter_context((out/"step-feedback.jsonl").open("w",buffering=1))
                def check_hold():
                    actual=monitor()
                    require(bool(get(sdk.get_enable_state)),"Drive no longer enabled")
                    require(np.all((actual>=hold_low)&(actual<=hold_high)), "Hand moved outside held interval")
                    return actual

                def step(joint, delta_deg):
                    nonlocal hold_low,hold_high
                    require(joint in names and np.isfinite(delta_deg) and 0<abs(delta_deg)<=2,"Invalid session step")
                    idx=names.index(joint)
                    begin=check_hold()
                    target=begin.copy()
                    target[idx]+=np.deg2rad(delta_deg)
                    require(np.all((target>=bounds[:,0])&(target<=bounds[:,1])),"Step outside joint limits")
                    for pose in np.linspace(begin,target,max(5,int(np.ceil(abs(delta_deg)/.25))+1)): geometry.pose(start,pose)
                    require(max(abs(check_hold()-begin))<.006,"State changed while planning")
                    hold_low=np.minimum(target,begin)-.015
                    hold_high=np.maximum(target,begin)+.015
                    set_value(sdk.set_joint_position,target.tolist(),False)
                    report["step_sent"]=True
                    sent=time.monotonic()
                    row={"step":len(evidence)+1,"joint":joint,"requested_delta_rad":float(target[idx]-begin[idx])}
                    evidence.append(row)
                    while True:
                        actual=check_hold()
                        elapsed=time.monotonic()-sent
                        row.update(measured_delta_rad=float(actual[idx]-begin[idx]),
                                   target_residual_rad=float(actual[idx]-target[idx]),
                                   target_reached=bool(abs(actual[idx]-target[idx])<.004))
                        motion.write(json.dumps({"step":row["step"],"ms":elapsed*1000,"sequence":seq,
                                                 "joint_rad":actual.tolist()})+"\n")
                        if row["target_reached"] or elapsed>=1.2:break
                    atomic_json(out/"steps.json",evidence)
                    return row

                from async_policy import FastLeftIK
                from kinematics import URDFKinematics
                ik=None
                if dialogue:
                    ik=FastLeftIK(URDFKinematics(DEPLOYMENT/"assets/robot/marvin_dual_arm.urdf"),
                                  np.array([(j.min_deg,j.max_deg) for j in arm.cfg.joints]),"right")

                def arm_step(joint,delta_deg):
                    nonlocal start, snap
                    from cells_real import ArmTrajectoryBackend
                    require(joint in [f"Joint{i}_R" for i in range(1,8)] and np.isfinite(delta_deg) and 0<abs(delta_deg),"Invalid arm step")
                    begin=np.array(arm.read_joint_position_deg("B"))
                    goal=begin.copy()
                    goal[int(joint[5])-1]+=delta_deg
                    path=np.linspace(begin,goal,max(2,int(np.ceil(abs(delta_deg)/.25))+1))
                    return execute_arm_path(path)

                def cartesian_move(values,relative):
                    from .cartesian import cartesian_path
                    begin=arm.read_joint_position_deg("B")
                    try:
                        path,target_pose=cartesian_path(ik,begin,values,relative)
                        result=execute_arm_path(path)
                    except ValueError as exc:
                        return {"status":"path_rejected","motion_issued":False,"reason":str(exc)}
                    pose=ik.forward_jacobian(np.deg2rad(start))[0]
                    result.update(target_pose_mount=target_pose.tolist(),measured_pose_mount=pose.tolist())
                    return result

                def execute_arm_path(path):
                    nonlocal start,snap
                    from cells_real import ArmTrajectoryBackend
                    begin,goal=path[0],path[-1]
                    arm_bounds=np.array([(j.min_deg,j.max_deg) for j in arm.cfg.joints])
                    require(np.all((goal>=arm_bounds[:,0])&(goal<=arm_bounds[:,1])),"Arm goal outside native limits")
                    fingers=check_hold()
                    for pose in path:
                        geometry.pose(pose,fingers)
                        model=geometry.model
                        data=geometry.data
                        for gid in range(model.ngeom):
                            name=model.geom(gid).name or ""
                            if not name.startswith("Link") or not name.endswith("_R_visual"):
                                continue
                            mesh=model.geom(gid).dataid[0]
                            off=model.mesh_vertadr[mesh]
                            vertices=model.mesh_vert[off:off+model.mesh_vertnum[mesh]]
                            world=vertices@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
                            table=world[(abs(world[:,0])<.65)&(world[:,1]<-.20)&(world[:,1]>-1)]
                            if len(table):
                                require(min(table@geometry.normal)-geometry.height>geometry.margin,"Arm link too close to table")
                            require(geometry.other_arm_gap(gid)>geometry.separation,"Arms too close")
                    frame=camera()
                    observed=read_state(arm,hands,frame.monotonic_ns)
                    folder=out/f"arm-step-{report['arm_commands']}"
                    folder.mkdir()
                    backend=ArmTrajectoryBackend(frame,observed,reference.extrinsics,folder,arm,hands,camera,excursion_deg=None)
                    report["arm_commands"]+=1
                    result=backend.run_path("right","B",goal,planned_path=path)
                    require("error" not in result,str(result))
                    start=np.array(arm.read_joint_position_deg("B"))
                    state["arm_deg"]["right"]=start.tolist()
                    snap=camera()
                    report.setdefault("arm_steps",[]).append(result)
                    atomic_json(out/"arm-steps.json",report["arm_steps"])
                    return result

                phase="index_step"
                first_step=step(names[index],-.75)
                report.update(first_step)
                if dialogue:
                    report["dialogue_started"]=True
                    cutoff=float("inf")
                    def snapshot():
                        actual=check_hold()
                        picture=camera()
                        buf=BytesIO()
                        Image.fromarray(picture.rgb).save(buf,format="JPEG",quality=78)
                        return {"feedback":{"joint_rad":actual.tolist(),"sequence":seq,"enabled":True,
                                            "frames_checked":len(frames),"arm_commands":report["arm_commands"],
                                            "tcp_pose_mount":ik.forward_jacobian(np.deg2rad(start))[0].tolist(),
                                            "right_arm_deg":list(start),"arm_joint_names":[f"Joint{i}_R" for i in range(1,8)]},
                                "image_base64":base64.b64encode(buf.getvalue()).decode("ascii")}
                    def measure(box):
                        from .station_diagnostics import tcp_report
                        actual=check_hold()
                        picture=camera()
                        observed=read_state(arm,hands,picture.monotonic_ns)
                        require(observed["joints_aligned"],"Unaligned target observation")
                        folder=out/f"measurement-{len(list(out.glob('measurement-*')))}"
                        folder.mkdir()
                        backend=SnapshotBackend(picture,observed,reference.extrinsics,folder)
                        target=backend.locate(backend.frame_id,"shuttlecock",box)
                        pose=tcp_report(geometry,start,actual,"right",config)
                        return {"target_surface":target,"current_grip_geometry":pose,
                                "scope":"surface and modeled pads; contact offset unverified"}

                    phase="model_steps"
                    report["conclusion"]=dialogue.run(step,check_hold,snapshot,names,first_step,arm_move=arm_step,measure=measure,cartesian_move=cartesian_move)
                phase="unloaded_stop"
            finally:
                if armed:release()
            first=joints()
            time.sleep(.2)
            report["post_disable_change_rad"]=float(max(abs(joints()-first)))
            snap=camera()
            after=read_state(arm,hands,snap.monotonic_ns)
            dest=out/"after"
            dest.mkdir()
            SnapshotBackend(snap,after,reference.extrinsics,dest)
            report["status"]="session_complete" if dialogue else ("unloaded_step_measured" if report["target_reached"] else "tracking_shortfall")
    except Exception as exc:
        report["reason"]=str(exc)
    report["phase"]=phase
    atomic_json(out/"result.json",report)
    return report
