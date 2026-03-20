"""
Raw npz → GR00T-compatible LeRobot v2 format 변환
GR00T와 SmolVLA 모두 이 데이터를 사용.

사용법:
  python utils/convert_to_lerobot.py --input-dir ./data/raw --output-dir ./data/lerobot_dataset
  python utils/convert_to_lerobot.py --verify --data-dir ./data/lerobot_dataset
"""

import os, sys, json, argparse
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.doosan_e0509_config import *


def convert(input_dir, output_dir, task, fps=10):
    inp = Path(input_dir)
    out = Path(output_dir)
    episodes = sorted(inp.glob("episode_*"))
    if not episodes:
        print(f"❌ No episodes in {input_dir}")
        return

    print(f"Converting {len(episodes)} episodes...")
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "videos" / "chunk-000" / "observation.images.front").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    all_states, all_actions, ep_metas = [], [], []
    total_frames = 0

    for idx, ep_dir in enumerate(episodes):
        data = np.load(str(ep_dir / "data.npz"))
        states, actions = data["states"], data["actions"]
        n = len(states)

        # Images → MP4
        img_file = ep_dir / "images.npz"
        if img_file.exists():
            imgs = np.load(str(img_file))["images"]
            vid_path = out / "videos" / "chunk-000" / "observation.images.front" / f"episode_{idx:06d}.mp4"
            _save_mp4(imgs, str(vid_path), fps)

        # Parquet
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pa.table({
                "observation.state": [states[i].tolist() for i in range(n)],
                "action": [actions[i].tolist() for i in range(n)],
                "timestamp": [float(i) / fps for i in range(n)],
                "episode_index": [idx] * n,
                "frame_index": list(range(total_frames, total_frames + n)),
                "index": list(range(total_frames, total_frames + n)),
                "task_index": [0] * n,
            })
            pq.write_table(table, str(out / "data" / "chunk-000" / f"episode_{idx:06d}.parquet"))
        except ImportError:
            print("⚠️ pyarrow not installed. Saving as npz fallback.")
            np.savez(str(out / "data" / "chunk-000" / f"episode_{idx:06d}.npz"),
                     states=states, actions=actions)

        all_states.append(states)
        all_actions.append(actions)
        ep_metas.append({"episode_index": idx, "tasks": [task], "length": n})
        total_frames += n
        print(f"  ✅ episode_{idx:04d}: {n} frames")

    # Stats
    s = np.concatenate(all_states)
    a = np.concatenate(all_actions)
    stats = {
        "observation.state": {"mean": s.mean(0).tolist(), "std": s.std(0).clip(1e-6).tolist(),
                              "min": s.min(0).tolist(), "max": s.max(0).tolist()},
        "action": {"mean": a.mean(0).tolist(), "std": a.std(0).clip(1e-6).tolist(),
                   "min": a.min(0).tolist(), "max": a.max(0).tolist()},
    }

    # Meta files
    info = {
        "codebase_version": "v2.1", "robot_type": "doosan_e0509",
        "total_episodes": len(episodes), "total_frames": total_frames, "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "features": {
            "observation.state": {"dtype": "float32", "shape": [ACTION_DIM]},
            "action": {"dtype": "float32", "shape": [ACTION_DIM]},
            "observation.images.front": {
                "dtype": "video", "shape": [CAMERA_HEIGHT, CAMERA_WIDTH, 3],
                "video.fps": fps, "video.codec": "avc1", "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False, "has_audio": False, "video.channels": 3,
            },
        },
    }
    _write_json(out / "meta" / "info.json", info)
    _write_json(out / "meta" / "stats.json", stats)
    with open(str(out / "meta" / "episodes.jsonl"), "w") as f:
        for ep in ep_metas:
            f.write(json.dumps(ep) + "\n")
    with open(str(out / "meta" / "tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task}) + "\n")

    print(f"\n✅ {len(episodes)} episodes, {total_frames} frames → {out}")


def _write_json(path, data):
    with open(str(path), "w") as f:
        json.dump(data, f, indent=2)


def _save_mp4(images, path, fps=10):
    try:
        import cv2
        h, w = images.shape[1], images.shape[2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
        for img in images:
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        writer.release()
    except Exception:
        try:
            import subprocess, tempfile
            from PIL import Image
            with tempfile.TemporaryDirectory() as tmp:
                for i, img in enumerate(images):
                    Image.fromarray(img).save(f"{tmp}/f_{i:06d}.png")
                subprocess.run(["ffmpeg", "-y", "-framerate", str(fps),
                                "-i", f"{tmp}/f_%06d.png", "-c:v", "libx264",
                                "-pix_fmt", "yuv420p", path], capture_output=True)
        except Exception as e:
            print(f"  ⚠️ Video encoding failed: {e}")


def verify(data_dir):
    info_path = Path(data_dir) / "meta" / "info.json"
    if info_path.exists():
        with open(str(info_path)) as f:
            info = json.load(f)
        print(f"✅ {info['total_episodes']} episodes, {info['total_frames']} frames, "
              f"fps={info['fps']}, robot={info['robot_type']}")
        stats_path = Path(data_dir) / "meta" / "stats.json"
        if stats_path.exists():
            with open(str(stats_path)) as f:
                stats = json.load(f)
            print(f"   action mean: {[f'{x:.4f}' for x in stats['action']['mean']]}")
            print(f"   action std:  {[f'{x:.4f}' for x in stats['action']['std']]}")
    else:
        print(f"❌ No info.json in {data_dir}/meta/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default=RAW_DATA_DIR)
    p.add_argument("--output-dir", default=LEROBOT_DATA_DIR)
    p.add_argument("--task", default="Pick up the blue bottle and place it on the tray")
    p.add_argument("--fps", type=int, default=CONTROL_HZ)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--data-dir", default=None)
    args = p.parse_args()
    if args.verify:
        verify(args.data_dir or args.output_dir)
    else:
        convert(args.input_dir, args.output_dir, args.task, args.fps)
