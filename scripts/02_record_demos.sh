#!/bin/bash
# P1(조작)+P2(운영): 데모 녹화
set -e
TASK=${1:-"Pick up the blue bottle and place it on the tray"}
NUM=${2:-50}
echo "=== Demo Recording: ${NUM} episodes ==="
echo "P1: hand-guiding으로 동작 / P2: Enter/Ctrl+C로 녹화 제어"
python utils/doosan_recorder.py --task "${TASK}" --num-episodes ${NUM}
