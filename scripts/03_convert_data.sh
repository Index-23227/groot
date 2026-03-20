#!/bin/bash
# P3 담당: Raw → LeRobot v2 변환 + modality.json
set -e
TASK=${1:-"Pick up the blue bottle and place it on the tray"}
echo "=== Data Conversion ==="
python utils/convert_to_lerobot.py --task "${TASK}"
python configs/groot_modality_config.py --output ./data/lerobot_dataset/meta/modality.json
python utils/convert_to_lerobot.py --verify
echo "✅ Data ready! → bash scripts/04_train_groot.sh & bash scripts/04_train_smolvla.sh"
