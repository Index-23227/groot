"""
GR00T N1.6 Modality Config for Doosan E0509.
Isaac-GR00T의 examples/SO100/so100_config.py 참고.

사용:
  --modality-config-path configs/groot_modality_config.py
  python configs/groot_modality_config.py --output ./data/lerobot_dataset/meta/modality.json
"""

import json, argparse, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.doosan_e0509_config import NUM_JOINTS

MODALITY_CONFIG = {
    "state": {
        "joint_positions": {
            "original_key": "observation.state",
            "dim": list(range(NUM_JOINTS)),
            "transform": "normalize",
        },
        "gripper": {
            "original_key": "observation.state",
            "dim": [NUM_JOINTS],
            "transform": "normalize",
        },
    },
    "action": {
        "joint_positions": {
            "original_key": "action",
            "dim": list(range(NUM_JOINTS)),
            "transform": "normalize",
        },
        "gripper": {
            "original_key": "action",
            "dim": [NUM_JOINTS],
            "transform": "normalize",
        },
    },
    "video": {
        "front": {"original_key": "observation.images.front"},
    },
    "annotation": {
        "task": {"original_key": "annotation.human.task"},
    },
}

def generate_modality_json(output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(MODALITY_CONFIG, f, indent=2)
    print(f"Generated: {output_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="./data/lerobot_dataset/meta/modality.json")
    generate_modality_json(p.parse_args().output)
