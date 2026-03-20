#!/bin/bash
# P2 담당: GR00T N1.6 환경 셋업 (~2h)
set -e
echo "=== [P2] GR00T N1.6 Setup ==="

if [ ! -d "Isaac-GR00T" ]; then
    echo "[1/5] Cloning Isaac-GR00T..."
    git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git
fi

echo "[2/5] Installing dependencies..."
cd Isaac-GR00T && pip install -e .[base]

echo "[3/5] Installing flash-attn..."
pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️ flash-attn failed"

echo "[4/5] Downloading checkpoint..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/GR00T-N1.6-3B')
print('Done')
" 2>/dev/null || echo "⚠️ Will download during training"

echo "[5/5] Sanity check: 1 step..."
python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path ./demo_data/cube_to_bowl_5 \
  --embodiment-tag NEW_EMBODIMENT \
  --max-steps 1 --batch-size 4 --num-gpus 1 2>&1 | tail -3
cd ..

echo "✅ GR00T ready. 다음: 데모 수집 합류"
