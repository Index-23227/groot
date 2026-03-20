#!/bin/bash
# P3 담당: SmolVLA 환경 셋업 (~30분)
set -e
echo "=== [P3] SmolVLA Setup ==="

if [ ! -d "lerobot" ]; then
    echo "[1/2] Cloning LeRobot..."
    git clone https://github.com/huggingface/lerobot.git
fi

echo "[2/2] Installing SmolVLA..."
cd lerobot && pip install -e ".[smolvla]" && cd ..

python -c "
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
policy = SmolVLAPolicy.from_pretrained('lerobot/smolvla_base')
print('✅ SmolVLA loaded')
"

echo "✅ SmolVLA ready (30분 완료). 다음: 데이터 변환 준비 / 데모 수집 합류"
