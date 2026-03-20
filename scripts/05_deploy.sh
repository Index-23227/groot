#!/bin/bash
# 배포: groot / smolvla / planc
MODEL=${1:-"groot"}
INSTRUCTION=${2:-"Pick up the blue bottle and place it on the tray"}

echo "=== Deploy: ${MODEL} ==="

if [ "$MODEL" = "groot" ]; then
    echo "▶ Terminal 1 (inference server):"
    echo "  cd Isaac-GR00T && python gr00t/eval/run_gr00t_server.py \\"
    echo "    --model-path ../checkpoints/groot/checkpoint-10000 \\"
    echo "    --embodiment-tag NEW_EMBODIMENT"
    echo ""
    echo "▶ Terminal 2 (robot controller):"
    echo "  python utils/doosan_vla_controller.py \\"
    echo "    --vla-url http://localhost:5555 \\"
    echo "    --instruction \"${INSTRUCTION}\""

elif [ "$MODEL" = "smolvla" ]; then
    echo "▶ Terminal 1:"
    echo "  lerobot-eval --policy.path=./checkpoints/smolvla/last/pretrained_model"
    echo ""
    echo "▶ Terminal 2:"
    echo "  python utils/doosan_vla_controller.py \\"
    echo "    --vla-url http://localhost:8000 \\"
    echo "    --instruction \"${INSTRUCTION}\""

elif [ "$MODEL" = "planc" ]; then
    echo "▶ Single terminal:"
    echo "  python utils/plan_c_classical.py --instruction \"${INSTRUCTION}\""
fi

echo ""
echo "각 터미널에서 위 명령어를 실행하세요."
