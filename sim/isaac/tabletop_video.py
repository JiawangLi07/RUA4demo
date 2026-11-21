"""Render synchronized external and head videos, preserving every recorded physics-time frame."""

import argparse
import itertools
import json
import subprocess
import tempfile
from pathlib import Path

TARGETS = {
    "ball": "小球",
    "bottle": "瓶子",
    "box": "方盒",
    "can": "圆罐",
    "cushion": "软垫",
    "driver": "螺丝刀",
    "marker": "马克笔",
    "metal": "金属件",
    "redtool": "红色工具",
    "shuttle": "横放羽毛球",
    "spray": "喷瓶",
    "tube": "球筒",
    "white": "白色小件",
    "wire": "线材",
    "egg": "椭球",
    "spool": "线轴",
    "pliers": "钳子",
    "clip": "夹件",
    "green": "绿色件",
    "red": "红色件",
    "yellow": "黄色件",
    "orange": "橙色件",
    "carrot": "胡萝卜形件",
    "block": "方块",
    "stepped_rod": "阶梯杆",
}
PHASES = {
    "reacquired_head_roi": "从头部画面重新锁定目标",
    "head_only_grasp": "头部 D455 网格直接求抓姿",
    "lightning_head_pregrasp": "从头部网格筛选预抓取和腕部观察位",
    "lightning_filter": "筛选 Lightning Grasp 候选",
    "hand_assignment": "按桌面观测分配左右手",
    "survey_head": "头部观测桌面",
    "survey_wrist": "腕部复查",
    "attempt_start": "选择目标",
    "registered_head": "头部配准",
    "coarse_head_roi": "估计观察位置",
    "wrist_search": "换视角",
    "registered_wrist": "腕部近景配准",
    "selected": "选定手形",
    "mesh_contact_family": "按网格几何选择接触部位",
    "aligned_finger_pads": "根据腕部网格调整指腹位置",
    "mesh_fitted_palm_and_joints": "腕部网格求解掌位和各指角度",
    "planned_mesh_closure": "执行网格求得的逐指闭合轨迹",
    "planned_carry": "按物体下侧形状收拢手指",
    "handle_surface_targets": "按手柄截面求各指接触点",
    "continuous_mesh_closure": "按关节回读逐指收拢",
    "mesh_closure_readback": "核对各指闭合",
    "seat_handle_encoders": "逐指贴合粗手柄",
    "handle_encoder_readback": "核对手柄两侧指位",
    "handle_reobserved": "复测闭合后的手柄",
    "mesh_reobserved": "复测手指间的物体位置",
    "handle_plan": "从腕部网格拟合多指对握",
    "synchronous_handle_approach": "拇指和对侧手指同步接近手柄",
    "approach": "接近物体",
    "close": "闭合手指",
    "test_lift": "试提约 2 厘米",
    "test_observation": "核对物体是否跟手",
    "test_observation_repeat": "复核试提",
    "measure_lift_clearance": "测量物体悬空高度",
    "correct_lift_clearance": "按物体实测高度补偿",
    "hold_start": "开始计时保持",
    "hold_end": "结束保持",
    "lower_and_release": "放低并松开",
    "local_withdraw": "局部撤手",
    "clear_previous_target": "离开上一目标",
    "attempt_end": "本次尝试结束",
    "target_end": "切换目标",
    "episode_end": "试验结束",
}


def clock(seconds):
    ms = round(seconds * 1000)
    hour, ms = divmod(ms, 3600000)
    minute, ms = divmod(ms, 60000)
    second, ms = divmod(ms, 1000)
    return f"{hour:02d}:{minute:02d}:{second:02d},{ms:03d}"


def render(run, out, speed, scene=None):
    frames = json.loads((run / "frames.json").read_bytes())
    if len(frames) < 2:
        raise ValueError("two or more recorded frames required")
    if [f["frame"] for f in frames] != list(range(len(frames))):
        raise ValueError("capture index is not continuous")
    fps = round(1 / (frames[1]["sim_time"] - frames[0]["sim_time"]))
    end = len(frames) / fps
    events = [
        json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()
    ]
    scene = scene or run / "scene.json"
    aliases = (
        json.loads(scene.read_bytes()).get("evaluation_aliases", {})
        if scene.is_file()
        else {}
    )
    lines = [(0.0, "双臂后撤，指尖向上，掌心相对")]
    for event in events:
        phase = event["phase"]
        name = aliases.get(event["name"], event["name"])
        msg = PHASES.get(phase, "视觉核对保持" if phase.startswith("hold_") else "")
        if not msg:
            continue
        target = (
            f"{TARGETS.get(name, name)} · 第 {event['attempt']} 次　"
            if name and event["attempt"]
            else ""
        )
        text = target + msg
        if phase in ("selected", "close"):
            text += "　" + event.get("posture", "").replace("_", " ")
        if event.get("vote") in ("on_table", "carried", "unknown"):
            text += (
                "　"
                + {
                    "on_table": "仍在桌上",
                    "carried": "观察到随手移动",
                    "unknown": "视觉证据不足",
                }[event["vote"]]
            )
        t = event["time"]
        if t == lines[-1][0]:
            lines[-1] = (t, text)
        elif text != lines[-1][1]:
            lines.append((t, text))
    lines.append((end, ""))
    inputs = []
    for folder, movie in (("observer_frames", "observer.mp4"), ("frames", "head.mp4")):
        if (run / folder).is_dir():
            inputs.extend(
                [
                    "-framerate",
                    str(fps),
                    "-i",
                    str((run / folder / "%06d.png").resolve()),
                ]
            )
        else:
            inputs.extend(["-i", str((run / movie).resolve())])
    with tempfile.TemporaryDirectory() as tmp:
        captions = Path(tmp) / "captions.srt"
        cards = []
        for i, ((t, text), (next_t, _)) in enumerate(itertools.pairwise(lines), 1):
            if next_t <= t:
                continue
            cards.append(f"{i}\n{clock(t)} --> {clock(next_t)}\n{text}\n")
        captions.write_text("\n".join(cards), encoding="utf8")
        filters = (
            "[0:v]scale=640:426,pad=640:480:0:27:color=0x17212a[left];"
            "[1:v]scale=640:480[right];[left][right]hstack,pad=iw:ih+62:0:0:color=0x17212a,"
            "subtitles=captions.srt:force_style='FontName=Noto Sans CJK SC,FontSize=15,Alignment=2,MarginV=9',"
            f"settb=AVTB,setpts=PTS/{speed}"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                *inputs,
                "-filter_complex",
                filters,
                "-r",
                str(fps * speed),
                "-frames:v",
                str(len(frames)),
                "-c:v",
                "libx264",
                "-threads",
                "3",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out.resolve()),
            ],
            cwd=tmp,
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene", type=Path, help="optional offline display names")
    parser.add_argument("--speed", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    render(args.run, args.out, args.speed, args.scene)
