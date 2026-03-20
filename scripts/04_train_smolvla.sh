#!/bin/bash
# P3 담당: SmolVLA fine-tuning (Plan B, GPU 1장)
set -e
echo "=== [Plan B] SmolVLA Fine-tuning ==="

export CUDA_VISIBLE_DEVICES=4

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/doosan_e0509_pickplace \
  --dataset.root=./data/lerobot_dataset \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=./checkpoints/smolvla \
  --job_name=doosan_smolvla \
  --policy.device=cuda \
  --save_checkpoint=true \
  --save_freq=5000

echo "✅ SmolVLA done! Checkpoints: ./checkpoints/smolvla/"
