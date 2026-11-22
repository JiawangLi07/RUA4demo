"""Full-speed native capture with a fixed close-up and time-aligned, offline physical audit."""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim.isaac.scene_meshes import load_mesh_bundle
from sim.isaac.tabletop_score import path_minimum
from sim.isaac.tabletop_video import PHASES, TARGETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--crop", nargs=3, type=int, required=True, metavar=("X", "Y", "SIZE")
    )
    args = parser.parse_args()
    cv2.setNumThreads(1)
    run = args.run
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scene = json.loads((run / "scene.json").read_bytes())
    target = next(
        o
        for o in scene["added_objects"]
        if (
            o["name"] == args.target
            or scene.get("evaluation_aliases", {}).get(
                Path(o["mesh"]["file"]).stem, Path(o["mesh"]["file"]).stem
            )
            == args.target
        )
    )
    bundle = load_mesh_bundle(target["mesh"])
    points = np.concatenate([v for v, _ in bundle["collisions"]])
    with np.load(run / "physics_audit.npz") as z:
        body = z["object_names"].tolist().index(target["name"])
        decisions = json.loads((run / "decisions.json").read_bytes())
        code = next(
            key
            for key, label in scene["evaluation_aliases"].items()
            if label == args.target
        )
        decision = decisions[code]
        hands = decision.get("attempt_hands") or [decision.get("hand", "right")]
        loads = {}
        table_loads = {}
        for side in set(hands):
            link = z[side + "_link_names"].tolist().index(side + "_hand_C_MC")
            loads[side] = np.column_stack(
                (
                    z[side + "_link_load_N"][:, link, body],
                    z[side + "_digit_load_N"][:, :, body],
                )
            )
            if side + "_link_table_load_N" in z:
                table_loads[side] = z[side + "_link_table_load_N"].sum(axis=1)
        lower = 100 * (
            path_minimum(z["pose_wxyz"][:, body], points) - float(z["table_z"])
        )
        dt = float(z["dt"])
    labels = ["中央掌面", "拇指", "食指", "中指", "无名指", "小指"]
    frames = json.loads((run / "frames.json").read_bytes())
    events = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
    changes = [(0.0, hands[0])]
    for event in events:
        if event["name"] == code and event["phase"] == "attempt_start":
            changes.append((event["time"], hands[event["attempt"] - 1]))
    hand_index = 0
    phases = [(0.0, "双臂竖掌，开始观察桌面")]
    for e in events:
        name = scene["evaluation_aliases"].get(e["name"], "")
        if name and e["phase"] in PHASES:
            phases.append(
                (
                    e["time"],
                    f"{'左手' if e.get('hand') == 'left' else '右手'} · {TARGETS.get(name, name)}　尝试 {e['attempt']}　·　{PHASES[e['phase']]}",
                )
            )
    capture = cv2.VideoCapture(str(run / "observer.mp4"))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError("unreadable native camera recording")
    stage_file = args.out.with_name("." + args.out.name)
    width, height = 1280, 628
    writer = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "3",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "17",
            "-movflags",
            "+faststart",
            str(stage_file),
        ],
        stdin=subprocess.PIPE,
    )
    filepath = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    small, font, bold = [ImageFont.truetype(filepath, size) for size in (16, 19, 22)]
    x, y, size = args.crop
    phase = 0
    for index, frame in enumerate(frames):
        ok, original = capture.read()
        assert ok and original.shape[:2] == (512, 768)
        image = Image.new("RGB", (width, height), (248, 248, 246))
        rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        image.paste(Image.fromarray(rgb), (0, 0))
        zoom = cv2.resize(
            rgb[y : y + size, x : x + size],
            (512, 512),
            interpolation=cv2.INTER_LANCZOS4,
        )
        image.paste(Image.fromarray(zoom), (768, 0))
        d = ImageDraw.Draw(image)
        time = frame["sim_time"]
        while (
            hand_index + 1 < len(changes) and changes[hand_index + 1][0] <= time + 1e-6
        ):
            hand_index += 1
        side = changes[hand_index][1]
        load = loads[side]
        title = (
            ("左手" if side == "left" else "右手")
            + " · "
            + TARGETS.get(args.target, args.target)
        )
        step = np.clip(round(time / dt) - 1, 0, len(load) - 1)
        while phase + 1 < len(phases) and phases[phase + 1][0] <= time + 1e-6:
            phase += 1
        d.text(
            (16, 519),
            f"{time:05.2f} s　{phases[phase][1]}",
            font=bold,
            fill=(34, 53, 57),
        )
        d.text(
            (1262, 521),
            f"{title}最低点离桌　{lower[step]:.2f} cm",
            font=font,
            fill=(24, 80, 84),
            anchor="ra",
        )
        d.text((16, 555), "离线接触载荷", font=font, fill=(34, 53, 57))
        detail = f"手与桌面 {table_loads[side][step]:.1f} N" if side in table_loads else title
        d.text((16, 583), detail, font=small, fill=(87, 99, 101))
        for i, (label, force) in enumerate(zip(labels, load[step])):
            left = 174 + 182 * i
            color = (33, 104, 94) if force > 0.25 else (102, 110, 112)
            d.text((left, 552), label, font=font, fill=color)
            d.text((left, 578), f"{force:.2f} N", font=small, fill=color)
            d.rounded_rectangle((left, 611, left + 146, 616), 2, fill=(218, 222, 219))
            active = int(146 * min(1, np.log1p(max(0, force)) / np.log(21)))
            if active >= 2:
                d.rounded_rectangle((left, 611, left + active, 616), 2, fill=color)
        writer.stdin.write(image.tobytes())
    more, _ = capture.read()
    assert not more
    capture.release()
    writer.stdin.close()
    assert writer.wait() == 0
    count = int(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "default=nw=1:nk=1",
                str(stage_file),
            ]
        )
    )
    assert count == len(frames)
    stage_file.replace(args.out)
    print(args.out, count)


if __name__ == "__main__":
    main()
