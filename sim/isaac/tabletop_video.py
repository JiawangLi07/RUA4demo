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
}
PHASES = {
    "survey_head": "头部观测桌面",
    "survey_wrist": "腕部复查",
    "attempt_start": "选择目标",
    "registered_head": "头部配准",
    "coarse_head_roi": "估计观察位置",
    "wrist_search": "换视角",
    "registered_wrist": "腕部近景配准",
    "selected": "选定手形",
    "aligned_finger_pads": "根据腕部网格调整指腹位置",
    "approach": "接近物体",
    "close": "闭合手指",
    "test_lift": "试提约 2 厘米",
    "test_observation": "核对物体是否跟手",
    "test_observation_repeat": "复核试提",
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


def render(run, out, speed):
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
    lines = [(0.0, "双臂后撤，指尖向上，掌心相对")]
    for event in events:
        phase = event["phase"]
        name = event["name"]
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
    parser.add_argument("--speed", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    render(args.run, args.out, args.speed)
