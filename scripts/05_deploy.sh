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

elif [ "$MODEL" = "track2" ]; then
    echo "▶ Track 2: Gemini-ER + cuRobo + RGBD (Zero-shot, VLA 학습 불필요)"
    echo ""
    echo "▶ Step 1: 카메라 테스트"
    echo "  python track2/camera_test.py"
    echo ""
    echo "▶ Step 2: 캘리브레이션 (최초 1회)"
    echo "  python track2/calibration.py"
    echo ""
    echo "▶ Step 3: 단일 태스크 실행"
    echo "  GEMINI_API_KEY=your-key python track2/track2_main.py \\"
    echo "    --instruction \"${INSTRUCTION}\""
    echo ""
    echo "▶ Step 4: 복합 태스크 (Gemini-ER 자동 분해)"
    echo "  GEMINI_API_KEY=your-key python track2/track2_main.py \\"
    echo "    --instruction \"${INSTRUCTION}\" --mode multi"
    echo ""
    echo "▶ Step 5: Dual-Brain (Gemini-ER + VLA 통합)"
    echo "  GEMINI_API_KEY=your-key python track2/dual_brain.py \\"
    echo "    --instruction \"${INSTRUCTION}\" --vla-url http://localhost:5555"
fi

echo ""
echo "각 터미널에서 위 명령어를 실행하세요."
