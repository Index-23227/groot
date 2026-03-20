#!/bin/bash
# P2 담당: GR00T N1.6 fine-tuning (A안, GPU 4장)
set -e
echo "=== [A안] GR00T N1.6 Fine-tuning ==="

source configs/doosan_e0509_config.py 2>/dev/null || true

cd Isaac-GR00T
export CUDA_VISIBLE_DEVICES=0,1,2,3

python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path ../data/lerobot_dataset \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path ../configs/groot_modality_config.py \
  --num-gpus 4 \
  --output-dir ../checkpoints/groot \
  --global-batch-size 64 \
  --max-steps 10000 \
  --save-steps 2000 \
  --save-total-limit 5 \
  --use-wandb \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader-num-workers 8

cd ..
echo "✅ GR00T training done! Checkpoints: ./checkpoints/groot/"
