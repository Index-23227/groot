"""
Gemini Embodied Reasoning 전체 테스트 스위트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

카테고리별 20개 모드:
  [A] 지각          object_detection, pose_estimation, depth_estimation,
                    scene_graph, occlusion_detection
  [B] 조작 계획     affordance, pregrasp_pose, sequencing,
                    collision_check, placement_check
  [C] 언어 기반     object_grounding, instruction_parse,
                    ambiguity_resolve, task_decompose
  [D] 안전/오류     grasp_success, placement_verify,
                    anomaly_detect, medication_verify
  [E] 연속 추론     state_change, progress_monitor, replan, human_intent

사용법:
  python tests/test_gemini_embodied.py                         # 인터랙티브
  python tests/test_gemini_embodied.py --mode object_detection
  python tests/test_gemini_embodied.py --mode benchmark        # 전체 실행
  python tests/test_gemini_embodied.py --image scene.jpg --mode affordance
  python tests/test_gemini_embodied.py --list                  # 모드 목록 + 세팅 가이드
"""

import os, sys, time, json, argparse, io, re, copy
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
from configs.doosan_e0509_config import CAMERA_WIDTH, CAMERA_HEIGHT
from utils.doosan_recorder import CameraCapture

# ══════════════════════════════════════════════════════════════════════════════
# 세팅 가이드 — 각 모드별 물리적 준비 사항
# ══════════════════════════════════════════════════════════════════════════════

SETUP_GUIDE = {
    # ── [A] 지각 ──────────────────────────────────────────────────────────────
    "object_detection": {
        "name": "물체 감지",
        "setup": [
            "작업대 위에 약병(색상 다양하게), 트레이, 컵 등 3~5개 물체 배치",
            "카메라가 작업 공간 전체를 포함하도록 위치 조정",
            "조명: 물체 그림자 최소화 (균일한 밝기)",
        ],
        "tip": "물체 간격 5cm 이상 유지하면 bbox가 더 정확합니다",
    },
    "pose_estimation": {
        "name": "6D 자세 추정",
        "setup": [
            "단일 물체(약병 하나)만 작업대 중앙에 배치",
            "다양한 방향으로 기울어진 물체도 테스트 (세워두기, 눕히기, 45° 기울이기)",
            "카메라 위치: 물체 위 30~50cm 비스듬히 (물체 측면도 보이도록)",
        ],
        "tip": "같은 물체를 3가지 방향으로 놓고 각각 테스트하면 좋습니다",
    },
    "depth_estimation": {
        "name": "깊이 추정",
        "setup": [
            "물체들을 카메라로부터 가까이(20cm), 중간(40cm), 멀리(60cm) 세 위치에 배치",
            "같은 크기 물체를 거리만 달리해서 배치하면 비교 테스트 가능",
            "배경에 체커보드 패턴 있으면 더 정확",
        ],
        "tip": "RGB 카메라 한 대로는 추정 오차가 크므로 상대적 거리 비교에 집중",
    },
    "scene_graph": {
        "name": "씬 그래프 (공간 관계)",
        "setup": [
            "약병(left), 트레이(center), 컵(right) — 가로로 3개 나란히 배치",
            "물체 하나를 다른 물체 위에 올려두기 (on_top_of 관계 테스트)",
            "물체 하나를 다른 물체 앞/뒤에 배치 (in_front_of 테스트)",
        ],
        "tip": "관계어 'left_of / right_of / in_front_of / behind / on_top_of / next_to' 전부 나오도록",
    },
    "occlusion_detection": {
        "name": "가림 감지",
        "setup": [
            "약병을 트레이 뒤에 절반만 보이도록 배치",
            "큰 물체 뒤에 작은 물체를 숨기기",
            "손으로 물체 일부를 가리고 테스트",
        ],
        "tip": "완전히 가린 것(invisible)과 부분 가린 것(partially_visible) 둘 다 테스트",
    },

    # ── [B] 조작 계획 ─────────────────────────────────────────────────────────
    "affordance": {
        "name": "그라스프 어포던스",
        "setup": [
            "원통형 약병 + 납작한 약봉투 + 뚜껑 있는 통 — 서로 다른 형태 3종 배치",
            "같은 약병을 세운 것 / 누운 것 / 기울어진 것으로 각각 테스트",
            "그리퍼가 접근 가능한 방향으로 물체 주변 공간 확보",
        ],
        "tip": "Doosan 그리퍼 최대 벌림 120mm 기준, 50mm 이상 물체는 side grasp만 가능",
    },
    "pregrasp_pose": {
        "name": "사전 자세 계획",
        "setup": [
            "물체 주변에 장애물(다른 약병 등) 배치해서 접근 경로 제약",
            "물체가 벽/가장자리 가까이 있어 한쪽만 접근 가능한 상황 연출",
            "로봇 기준 물체 방향: 정면, 측면, 대각선 각각 배치해서 테스트",
        ],
        "tip": "approach_direction 출력이 'from_top / from_left / from_right / from_front' 중 어느 게 나오는지 확인",
    },
    "sequencing": {
        "name": "다물체 순서 계획",
        "setup": [
            "트레이 하나 + 약병 3개(빨강/파랑/흰색) 배치",
            "약병마다 우선순위가 다른 상황 설정 (예: '빨간 약이 제일 급해')",
            "트레이 용량이 2개뿐인 상황 연출 (공간 계획 필요)",
        ],
        "tip": "instruction에 우선순위 정보 포함: '빨간 약부터 옮겨줘'",
    },
    "collision_check": {
        "name": "충돌 경로 검사",
        "setup": [
            "목표 물체(약병) 주변에 장애물(컵, 다른 병) 밀집 배치",
            "로봇이 지나가야 할 경로 위에 물체가 있는 상황",
            "높이 차이가 있는 장애물 배치 (낮은 장애물 위로 지나갈 수 있는지)",
        ],
        "tip": "obstacles 목록과 safe_approach_path 출력 확인",
    },
    "placement_check": {
        "name": "놓을 위치 안전 검사",
        "setup": [
            "트레이 위에 이미 물체가 있어서 공간이 좁은 상황",
            "트레이가 기울어진 상황 (안정적으로 놓을 수 없음)",
            "트레이가 비어있는 상황 (정상 케이스)",
        ],
        "tip": "safe=false 케이스도 반드시 테스트 — 거절 이유가 나와야 정상",
    },

    # ── [C] 언어 기반 ─────────────────────────────────────────────────────────
    "object_grounding": {
        "name": "언어 → 물체 위치",
        "setup": [
            "파란 약병, 빨간 약병, 흰색 약병 — 색상별 3개 배치",
            "크기가 다른 같은 색 약병 2개 배치 (큰 것/작은 것)",
            "'두 번째 약병', '왼쪽 것' 등 상대적 표현도 테스트",
        ],
        "tip": "instruction 변수를 바꿔가며 여러 번 테스트: '파란 것', '가장 큰 것', '트레이 옆에 있는 것'",
    },
    "instruction_parse": {
        "name": "복잡한 명령 파싱",
        "setup": [
            "약병 3개 + 트레이 2개(A구역/B구역) + 레이블 배치",
            "복잡한 명령: '빨간 약을 A트레이에, 파란 약은 B트레이에 옮겨줘'",
            "조건부 명령: '트레이가 비어있으면 파란 약병을 옮겨줘'",
        ],
        "tip": "output의 sub_tasks 리스트가 명령을 올바르게 분해했는지 확인",
    },
    "ambiguity_resolve": {
        "name": "모호한 명령 해결",
        "setup": [
            "똑같이 생긴 흰 약병 3개를 다른 위치에 배치",
            "비슷한 색의 약병 2개 (연파랑 vs 파랑) 배치",
            "'그 약병 가져와' 같은 지시대명사 포함 명령 사용",
        ],
        "tip": "candidates 목록과 disambiguation_question 출력 확인 — 모델이 되물어야 좋은 신호",
    },
    "task_decompose": {
        "name": "복잡한 태스크 분해",
        "setup": [
            "약병 여러 개 + 처방전 종이(레이블) + 트레이 배치",
            "고수준 명령: '오늘 처방 조제해줘' (여러 단계 필요)",
            "의존성 있는 태스크: 'A를 한 다음 B를 해줘'",
        ],
        "tip": "task_graph의 dependencies 필드로 순서 의존성 확인",
    },

    # ── [D] 안전/오류 감지 ────────────────────────────────────────────────────
    "grasp_success": {
        "name": "잡기 성공 여부 판단",
        "setup": [
            "성공 케이스: 그리퍼가 약병을 제대로 감싸고 있는 사진",
            "실패 케이스: 그리퍼가 약병 끝만 살짝 걸린 사진",
            "실패 케이스: 약병이 기울어진 채로 잡힌 사진",
            "→ 스마트폰으로 각 상황 사진 찍어서 --image로 테스트",
        ],
        "tip": "그리퍼 없이 손으로 약병 잡고 흉내내도 테스트 가능",
    },
    "placement_verify": {
        "name": "배치 완료 검증",
        "setup": [
            "정상 배치: 트레이 안에 약병이 세워진 상태",
            "오배치 1: 약병이 트레이 밖에 나와 있는 상태",
            "오배치 2: 약병이 트레이 안에서 넘어진 상태",
            "오배치 3: 약병이 다른 약병에 부딪혀 있는 상태",
        ],
        "tip": "correct=false일 때 correction_needed 출력이 나와야 정상",
    },
    "anomaly_detect": {
        "name": "이상 상황 감지",
        "setup": [
            "정상 씬: 약병들이 트레이에 정렬된 상태",
            "이상 1: 약병이 쏟아진 상태 (여러 개 엎어짐)",
            "이상 2: 작업 공간에 모르는 물체 침입 (음료수 캔 등)",
            "이상 3: 약병 레이블이 뒤집힌 상태",
        ],
        "tip": "anomaly_type 별로 robot_action='stop / alert / continue' 출력 확인",
    },
    "medication_verify": {
        "name": "약 종류 검증 (약국 특화)",
        "setup": [
            "색상 다른 약병 3개 + 처방전 종이(손으로 쓴 것도 OK) 배치",
            "처방전: '파란 약 2개, 빨간 약 1개' 같이 명시",
            "오조제 케이스: 처방전에는 파란 약인데 빨간 약이 트레이에 있는 상황",
        ],
        "tip": "match=false일 때 mismatch_detail이 구체적으로 나와야 좋은 모델",
    },

    # ── [E] 연속 추론 ─────────────────────────────────────────────────────────
    "state_change": {
        "name": "상태 변화 감지 (2프레임 비교)",
        "setup": [
            "프레임 A: 약병 3개가 원래 위치에 있는 상태로 캡처",
            "프레임 B: 약병 1개를 옮긴 후 캡처",
            "→ --image-before / --image-after 옵션으로 두 이미지 비교",
        ],
        "tip": "moved_objects, added_objects, removed_objects 출력 확인",
    },
    "progress_monitor": {
        "name": "태스크 진행 상황 추적",
        "setup": [
            "3개 약병을 트레이로 옮기는 태스크 중간 상태 촬영",
            "0개 옮긴 초기 상태, 1개 옮긴 상태, 2개 옮긴 상태, 완료 상태",
            "→ 각 단계를 사진 찍어두고 --image로 순서대로 테스트",
        ],
        "tip": "completed_steps / remaining_steps / progress_percent 출력 확인",
    },
    "replan": {
        "name": "실패 후 재계획",
        "setup": [
            "목표 약병을 잡으려 했는데 다른 물체가 옮겨진 상황",
            "트레이가 가득 찬 상황 (원래 목적지 사용 불가)",
            "목표 물체가 넘어진 상황",
            "→ instruction에 '이전에 시도했지만 실패했어' 맥락 포함",
        ],
        "tip": "failure_reason 분석 후 alternative_plan이 나와야 정상",
    },
    "human_intent": {
        "name": "사람 의도 인식",
        "setup": [
            "손을 뻗어 약병을 가리키는 제스처 (pointing)",
            "손을 뻗어 약병을 집으려는 제스처 (reaching)",
            "손바닥을 펴서 로봇을 멈추는 제스처 (stop)",
            "트레이를 가리키는 제스처 (indicating destination)",
        ],
        "tip": "robot_action='handover / wait / stop / assist' 출력 확인",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# 프롬프트
# ══════════════════════════════════════════════════════════════════════════════

PROMPTS = {

    # ── [A] 지각 ──────────────────────────────────────────────────────────────

    "object_detection": """
약국 조제 씬입니다. 로봇이 조작 가능한 모든 물체를 탐지하세요.
JSON만 답하세요 (코드블록 없이):
{
  "objects": [
    {
      "label": "blue_bottle",
      "display_name": "파란 약병",
      "bbox_norm": [cx, cy, w, h],
      "graspable": true,
      "color": "blue",
      "object_type": "bottle",
      "confidence": 0.95
    }
  ],
  "scene_description": "한 문장 요약",
  "num_graspable": 2
}
bbox_norm: 0~1 정규화, [cx, cy, w, h]
""",

    "pose_estimation": """
이미지에서 각 물체의 6D 자세를 추정하세요.
로봇 좌표계 기준: x=오른쪽, y=앞쪽, z=위쪽.
JSON만 답하세요:
{
  "objects": [
    {
      "label": "blue_bottle",
      "bbox_norm": [cx, cy, w, h],
      "orientation": "upright",
      "tilt_deg": 0,
      "tilt_axis": "none",
      "roll_deg": 0,
      "grasp_difficulty": "easy/medium/hard",
      "reason": "세워진 원통형, 위쪽 접근 용이"
    }
  ]
}
orientation: upright / tilted / lying_flat / inverted
""",

    "depth_estimation": """
이미지에서 각 물체까지의 상대적 거리를 추정하세요.
카메라 기준 원근감, 물체 크기 단서를 사용하세요.
JSON만 답하세요:
{
  "objects": [
    {
      "label": "blue_bottle",
      "bbox_norm": [cx, cy, w, h],
      "depth_category": "near/mid/far",
      "estimated_distance_cm": 30,
      "confidence": 0.7,
      "depth_cues_used": ["size", "position", "overlap"]
    }
  ],
  "depth_order": ["blue_bottle", "tray", "red_bottle"],
  "note": "단안 카메라 추정 한계 기술"
}
""",

    "scene_graph": """
이미지의 물체들 간 공간 관계를 그래프로 표현하세요.
JSON만 답하세요:
{
  "nodes": [
    {"id": "blue_bottle", "bbox_norm": [cx, cy, w, h], "type": "object"}
  ],
  "edges": [
    {
      "subject": "blue_bottle",
      "relation": "left_of",
      "object": "tray",
      "confidence": 0.9
    }
  ],
  "reachability": {
    "robot_can_reach": ["blue_bottle", "tray"],
    "blocked": []
  }
}
relation 종류: left_of / right_of / in_front_of / behind / on_top_of / inside / next_to / touching
""",

    "occlusion_detection": """
이미지에서 가려진 물체와 가시성을 분석하세요.
JSON만 답하세요:
{
  "objects": [
    {
      "label": "blue_bottle",
      "bbox_norm": [cx, cy, w, h],
      "visibility": "full/partial/occluded",
      "visible_percent": 80,
      "occluded_by": null,
      "graspable_despite_occlusion": true,
      "recommended_approach": "from_left"
    }
  ],
  "occlusion_summary": "전체 씬 가림 상황 요약"
}
""",

    # ── [B] 조작 계획 ─────────────────────────────────────────────────────────

    "affordance": """
Doosan E0509 로봇 (평행 그리퍼, 최대 벌림 120mm) 기준으로
각 물체의 그라스프 어포던스를 분석하세요.
JSON만 답하세요:
{
  "affordances": [
    {
      "object": "blue_bottle",
      "bbox_norm": [cx, cy, w, h],
      "graspable": true,
      "grasp_type": "cylindrical_side/top_pinch/lateral/envelope",
      "approach_direction": "from_top/from_left/from_right/from_front",
      "grasp_point_norm": [cx, cy],
      "estimated_width_mm": 50,
      "required_gripper_opening_mm": 60,
      "within_gripper_limit": true,
      "stability_score": 0.85,
      "reason": "원통형, 측면 파지 권장"
    }
  ]
}
""",

    "pregrasp_pose": """
각 물체를 잡기 위한 사전 자세(pre-grasp) 계획을 세우세요.
장애물과 접근 공간을 고려하세요.
JSON만 답하세요:
{
  "targets": [
    {
      "object": "blue_bottle",
      "bbox_norm": [cx, cy, w, h],
      "approach_direction": "from_top",
      "pregrasp_offset_mm": {"x": 0, "y": 0, "z": 100},
      "approach_angle_deg": 0,
      "clearance_available": true,
      "obstacles_in_path": [],
      "alternative_approach": "from_right",
      "notes": "위쪽 공간 충분, 직접 하강 가능"
    }
  ]
}
""",

    "sequencing": """
이미지에 여러 물체가 있습니다.
'모든 약병을 트레이에 옮겨줘' 명령을 위한 최적 순서를 계획하세요.
JSON만 답하세요:
{
  "task": "multi_pick_place",
  "objects_to_move": [
    {"order": 1, "label": "red_bottle", "bbox_norm": [cx, cy, w, h],
     "reason": "가장 접근하기 쉬운 위치, 다른 물체 방해 없음"}
  ],
  "destination": {"label": "tray", "bbox_norm": [cx, cy, w, h]},
  "strategy": "nearest_first/furthest_first/left_to_right/custom",
  "estimated_steps": 14,
  "risk_factors": ["crowded_workspace"]
}
""",

    "collision_check": """
로봇이 목표 물체에 접근할 때 충돌 위험을 분석하세요.
JSON만 답하세요:
{
  "target": {"label": "blue_bottle", "bbox_norm": [cx, cy, w, h]},
  "obstacles": [
    {
      "label": "red_bottle",
      "bbox_norm": [cx, cy, w, h],
      "collision_risk": "high/medium/low",
      "risk_direction": "from_top",
      "clearance_mm": 20
    }
  ],
  "safe_approach_directions": ["from_left", "from_top"],
  "blocked_directions": ["from_right"],
  "overall_risk": "low/medium/high",
  "recommendation": "왼쪽에서 비스듬히 접근 권장"
}
""",

    "placement_check": """
트레이(목적지)에 물체를 안전하게 놓을 수 있는지 검사하세요.
JSON만 답하세요:
{
  "destination": {"label": "tray", "bbox_norm": [cx, cy, w, h]},
  "placement_safe": true,
  "available_space": "large/medium/small/none",
  "occupied_positions": [],
  "recommended_spot_norm": [cx, cy],
  "risks": [],
  "tray_stable": true,
  "max_additional_items": 3,
  "notes": "트레이 왼쪽 공간 충분"
}
""",

    # ── [C] 언어 기반 ─────────────────────────────────────────────────────────

    "object_grounding": """
명령: "파란 약병을 가져와"

이미지에서 해당 명령의 목표 물체를 찾아 bbox를 반환하세요.
여러 후보가 있으면 모두 열거하고 최선 선택 이유를 설명하세요.
JSON만 답하세요:
{
  "instruction": "파란 약병을 가져와",
  "target_found": true,
  "best_match": {
    "label": "blue_bottle",
    "bbox_norm": [cx, cy, w, h],
    "match_score": 0.95,
    "match_reason": "색상 파랑 일치, 약병 형태 일치"
  },
  "other_candidates": [],
  "ambiguous": false
}
""",

    "instruction_parse": """
명령: "빨간 약은 A트레이에, 파란 약은 B트레이에 옮겨줘. 흰 약은 그냥 둬."

이 명령을 분석하고 구조화된 태스크로 분해하세요.
JSON만 답하세요:
{
  "original_instruction": "...",
  "sub_tasks": [
    {
      "id": 1,
      "action": "pick_and_place",
      "object": {"label": "red_bottle", "bbox_norm": [cx, cy, w, h]},
      "destination": {"label": "tray_A", "bbox_norm": [cx, cy, w, h]},
      "condition": null
    }
  ],
  "excluded_objects": [{"label": "white_bottle", "reason": "명시적으로 제외됨"}],
  "parse_confidence": 0.9,
  "ambiguities": []
}
""",

    "ambiguity_resolve": """
명령: "그 약병 가져와"

이미지에서 모호한 대상을 찾고 해결 방법을 제시하세요.
JSON만 답하세요:
{
  "instruction": "그 약병 가져와",
  "is_ambiguous": true,
  "candidates": [
    {"label": "bottle_1", "bbox_norm": [cx, cy, w, h], "description": "파란 약병 왼쪽"},
    {"label": "bottle_2", "bbox_norm": [cx, cy, w, h], "description": "파란 약병 오른쪽"}
  ],
  "most_likely": {"label": "bottle_1", "probability": 0.6, "reason": "최근 대화 맥락 없음, 크기 더 큼"},
  "disambiguation_question": "파란 약병이 두 개 있어요. 왼쪽 것인가요, 오른쪽 것인가요?",
  "can_proceed_without_confirmation": false
}
""",

    "task_decompose": """
명령: "오늘 처방 조제해줘. 파란 약 2개, 빨간 약 1개를 트레이에 담고 확인해줘."

이 고수준 태스크를 로봇이 실행 가능한 원자 동작으로 분해하세요.
JSON만 답하세요:
{
  "high_level_task": "pharmacy_dispensing",
  "atomic_steps": [
    {
      "step": 1,
      "action": "detect_objects",
      "description": "파란 약병 2개, 빨간 약병 1개 위치 파악",
      "preconditions": [],
      "estimated_duration_s": 2
    },
    {
      "step": 2,
      "action": "pick_and_place",
      "object": "blue_bottle_1",
      "destination": "tray",
      "preconditions": ["step_1_complete"],
      "estimated_duration_s": 5
    }
  ],
  "task_graph": {
    "dependencies": {"2": ["1"], "3": ["1"], "4": ["2", "3"]}
  },
  "total_estimated_duration_s": 30,
  "risks": ["blue_bottle 2개 식별 필요"]
}
""",

    # ── [D] 안전/오류 감지 ────────────────────────────────────────────────────

    "grasp_success": """
로봇이 물체를 잡은 상태의 이미지입니다.
그라스프 성공 여부를 판단하세요.
JSON만 답하세요:
{
  "grasp_attempt": {
    "object": "blue_bottle",
    "success": true,
    "confidence": 0.9
  },
  "grasp_quality": "secure/marginal/failed",
  "issues": [],
  "object_orientation": "upright",
  "slip_risk": "low/medium/high",
  "recommendation": "proceed/regrasp/abort",
  "reason": "그리퍼가 약병 중앙을 안정적으로 감싸고 있음"
}
issues 예시: "edge_grasp", "tilted_object", "partial_contact", "unstable"
""",

    "placement_verify": """
로봇이 물체를 트레이에 놓은 후의 이미지입니다.
배치가 올바른지 검증하세요.
JSON만 답하세요:
{
  "placed_object": {"label": "blue_bottle", "bbox_norm": [cx, cy, w, h]},
  "destination": {"label": "tray", "bbox_norm": [cx, cy, w, h]},
  "placement_correct": true,
  "object_stable": true,
  "inside_target": true,
  "issues": [],
  "correction_needed": null,
  "task_complete": true
}
issues 예시: "outside_target", "tipping_over", "blocking_other", "wrong_orientation"
correction_needed 예시: "push_left_5cm", "stand_upright"
""",

    "anomaly_detect": """
이미지에서 로봇 작업에 위험하거나 비정상적인 상황을 감지하세요.
JSON만 답하세요:
{
  "anomaly_detected": false,
  "anomalies": [
    {
      "type": "spilled_object",
      "description": "약병 3개가 넘어져 있음",
      "severity": "high/medium/low",
      "bbox_norm": [cx, cy, w, h],
      "robot_action": "stop/alert/avoid/continue"
    }
  ],
  "scene_status": "normal/warning/danger",
  "recommended_action": "proceed",
  "human_notification_needed": false
}
anomaly types: spilled_object / unknown_object / human_intrusion /
               damaged_item / wrong_label / unstable_stack / empty_workspace
""",

    "medication_verify": """
약국 조제 검증 씬입니다.
처방전(이미지 내 텍스트나 레이블)과 실제 약병을 대조하세요.
처방전이 안 보이면 "파란 약 2개, 빨간 약 1개" 기준으로 확인하세요.
JSON만 답하세요:
{
  "prescription": {
    "items": [
      {"medication": "blue_bottle", "quantity": 2},
      {"medication": "red_bottle", "quantity": 1}
    ]
  },
  "tray_contents": [
    {"label": "blue_bottle", "bbox_norm": [cx, cy, w, h], "count": 2}
  ],
  "verification_result": "pass/fail/partial",
  "mismatches": [],
  "missing_items": [],
  "extra_items": [],
  "dispensing_safe": true,
  "error_summary": null
}
""",

    # ── [E] 연속 추론 ─────────────────────────────────────────────────────────

    "state_change": """
이미지는 로봇 작업 후의 현재 씬입니다.
이전 상태(약병 3개가 작업대에 있었음)와 비교하여 변화를 감지하세요.
JSON만 답하세요:
{
  "current_state": {
    "objects": [
      {"label": "blue_bottle", "bbox_norm": [cx, cy, w, h], "location": "tray"}
    ]
  },
  "changes_detected": {
    "moved_objects": [
      {"label": "blue_bottle", "from": "workspace_left", "to": "tray"}
    ],
    "added_objects": [],
    "removed_objects": [],
    "state_changes": []
  },
  "task_progress": "1_of_3_bottles_moved",
  "next_recommended_action": "pick red_bottle"
}
""",

    "progress_monitor": """
현재 이미지는 '약병 3개를 트레이에 옮기기' 태스크 진행 중 상태입니다.
진행 상황을 분석하세요.
JSON만 답하세요:
{
  "task": "move_3_bottles_to_tray",
  "total_steps": 3,
  "completed_steps": 1,
  "progress_percent": 33,
  "completed_items": [
    {"label": "blue_bottle", "bbox_norm": [cx, cy, w, h]}
  ],
  "remaining_items": [
    {"label": "red_bottle", "bbox_norm": [cx, cy, w, h], "priority": 1}
  ],
  "on_track": true,
  "estimated_remaining_steps": 2,
  "issues_detected": []
}
""",

    "replan": """
상황: 로봇이 파란 약병을 잡으려 했으나 실패했습니다.
현재 씬을 보고 새로운 계획을 수립하세요.
JSON만 답하세요:
{
  "failure_analysis": {
    "original_target": "blue_bottle",
    "failure_reason": "object_moved/grasp_slip/collision/target_occluded",
    "current_scene_description": "씬 현재 상태"
  },
  "alternative_plan": {
    "strategy": "retry_different_approach/pick_different_object/request_human",
    "new_target": {"label": "blue_bottle", "bbox_norm": [cx, cy, w, h]},
    "new_approach_direction": "from_left",
    "steps": [
      {"step": 1, "action": "reposition_arm", "description": "왼쪽으로 재접근"}
    ]
  },
  "confidence": 0.8,
  "human_assistance_needed": false
}
""",

    "human_intent": """
이미지에 사람의 손이나 몸이 포함되어 있습니다.
사람의 의도를 파악하고 로봇의 적절한 반응을 결정하세요.
JSON만 답하세요:
{
  "human_detected": true,
  "human_body_parts": ["right_hand"],
  "gesture_type": "pointing/reaching/stop/handing_over/indicating",
  "gesture_confidence": 0.85,
  "indicated_object": {"label": "blue_bottle", "bbox_norm": [cx, cy, w, h]},
  "human_intent": "wants_robot_to_pick_blue_bottle",
  "robot_action": "handover/assist_pick/wait/stop/ask_confirm",
  "safety_concern": false,
  "response_message": "파란 약병을 드릴까요?"
}
""",
}

# 카테고리 정의
CATEGORIES = {
    "A_perception":     ["object_detection", "pose_estimation", "depth_estimation",
                         "scene_graph", "occlusion_detection"],
    "B_manipulation":   ["affordance", "pregrasp_pose", "sequencing",
                         "collision_check", "placement_check"],
    "C_language":       ["object_grounding", "instruction_parse",
                         "ambiguity_resolve", "task_decompose"],
    "D_safety":         ["grasp_success", "placement_verify",
                         "anomaly_detect", "medication_verify"],
    "E_sequential":     ["state_change", "progress_monitor", "replan", "human_intent"],
}

CATEGORY_LABELS = {
    "A_perception":   "[A] 지각",
    "B_manipulation": "[B] 조작 계획",
    "C_language":     "[C] 언어 기반",
    "D_safety":       "[D] 안전/오류",
    "E_sequential":   "[E] 연속 추론",
}

# ══════════════════════════════════════════════════════════════════════════════
# Gemini 클라이언트
# ══════════════════════════════════════════════════════════════════════════════

def get_client():
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        token_path = Path(__file__).parent.parent / "token"
        if token_path.exists():
            api_key = token_path.read_text().strip()
            os.environ["GOOGLE_API_KEY"] = api_key
        else:
            raise EnvironmentError("groot/token 파일 또는 GOOGLE_API_KEY 환경변수 필요")
    from google import genai
    return genai.Client(api_key=api_key)


def call_gemini(client, model: str, contents: list) -> tuple[str, float]:
    """이미지+텍스트 전송, (raw_text, latency_s) 반환"""
    t0 = time.time()
    resp = client.models.generate_content(model=model, contents=contents)
    return resp.text.strip(), round(time.time() - t0, 3)


def parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text).strip()
    try:
        s, e = text.find("{"), text.rfind("}") + 1
        return json.loads(text[s:e])
    except Exception:
        return {"raw": text, "parse_error": True}


def run_mode(client, model: str, img_rgb: np.ndarray, mode: str,
             extra_images: list = None) -> dict:
    """단일 모드 실행. extra_images: state_change 등 다중 이미지 모드용"""
    pil = Image.fromarray(img_rgb.astype(np.uint8))
    contents = [PROMPTS[mode], pil]
    if extra_images:
        contents += [Image.fromarray(img.astype(np.uint8)) for img in extra_images]

    print(f"\n{'─'*60}")
    print(f"  MODE : {mode}  ({SETUP_GUIDE[mode]['name']})")
    print(f"  MODEL: {model}")
    print(f"{'─'*60}")

    raw, latency = call_gemini(client, model, contents)
    result = parse_json(raw)
    result["_latency_s"] = latency
    result["_model"] = model
    result["_mode"] = mode

    if result.get("parse_error"):
        print(f"  [!] JSON 파싱 실패 — raw 출력:")
        print(f"  {raw[:300]}")
    else:
        print(f"  Latency: {latency}s")
        print("  Result:")
        print(json.dumps({k: v for k, v in result.items() if not k.startswith("_")},
                         ensure_ascii=False, indent=2))
    return result

# ══════════════════════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = [
    (0, 255, 0), (255, 165, 0), (0, 200, 255), (255, 0, 255),
    (200, 200, 0), (0, 255, 200), (255, 100, 100), (100, 100, 255),
]

def _tbg(img, text, org, scale=0.45, color=(255,255,255), bg=(0,0,0)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x-2, y-th-3), (x+tw+2, y+3), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

def _n2p(cx, cy, bw, bh, W, H):
    return (int((cx-bw/2)*W), int((cy-bh/2)*H),
            int((cx+bw/2)*W), int((cy+bh/2)*H))

def draw_result(img_bgr: np.ndarray, result: dict) -> np.ndarray:
    vis = img_bgr.copy()
    H, W = vis.shape[:2]
    mode = result.get("_mode", "")

    def bbox(obj_or_list, key="bbox_norm", idx=0):
        b = (obj_or_list if isinstance(obj_or_list, list) else obj_or_list.get(key, []))
        return b if len(b) == 4 else []

    # ── 공통: objects 배열 있으면 bbox 그리기
    objects = result.get("objects") or result.get("affordances") or result.get("nodes") or []
    for i, obj in enumerate(objects):
        b = obj.get("bbox_norm") or obj.get("grasp_point_norm", [])
        color = PALETTE[i % len(PALETTE)]
        if len(b) == 4:
            x1,y1,x2,y2 = _n2p(*b, W, H)
            cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
            label = obj.get("label") or obj.get("object") or obj.get("id") or "?"
            extra = ""
            if "confidence" in obj: extra = f" {obj['confidence']:.2f}"
            elif "grasp_type" in obj: extra = f" {obj['grasp_type']}"
            elif "visibility" in obj: extra = f" {obj['visibility']}"
            elif "depth_category" in obj: extra = f" {obj['depth_category']}"
            _tbg(vis, label+extra, (x1, max(y1-5,15)), color=color)

    # ── 모드별 추가 오버레이
    if mode in ("action_plan", "sequencing", "instruction_parse", "task_decompose"):
        tgt = result.get("target_object") or (result.get("objects_to_move") or [None])[0]
        dst = result.get("destination")
        if tgt and bbox(tgt):
            x1,y1,x2,y2 = _n2p(*tgt["bbox_norm"], W, H)
            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),3)
            _tbg(vis,"PICK: "+tgt.get("label","?"), (x1,max(y1-5,15)), color=(0,255,0))
        if dst and isinstance(dst, dict) and dst.get("bbox_norm") and len(dst["bbox_norm"])==4:
            x1,y1,x2,y2 = _n2p(*dst["bbox_norm"], W, H)
            cv2.rectangle(vis,(x1,y1),(x2,y2),(255,165,0),3)
            _tbg(vis,"PLACE: "+dst.get("label","?"), (x1,max(y1-5,15)), color=(255,165,0))
            if tgt and tgt.get("bbox_norm") and len(tgt["bbox_norm"])==4:
                p1=(int(tgt["bbox_norm"][0]*W), int(tgt["bbox_norm"][1]*H))
                p2=(int(dst["bbox_norm"][0]*W), int(dst["bbox_norm"][1]*H))
                cv2.arrowedLine(vis,p1,p2,(255,255,0),2,tipLength=0.2)
        steps = result.get("action_steps") or result.get("atomic_steps") or []
        for i,s in enumerate(steps[:6]):
            desc = s.get("description") or s.get("action","")
            _tbg(vis, f"{s.get('step',i+1)}. {desc[:40]}", (8, 22+i*20), scale=0.42)

    elif mode == "affordance":
        for aff in result.get("affordances", []):
            gp = aff.get("grasp_point_norm", [])
            if len(gp) == 2:
                cx_px, cy_px = int(gp[0]*W), int(gp[1]*H)
                ok = aff.get("graspable", True)
                c = (0,255,0) if ok else (0,0,255)
                cv2.drawMarker(vis,(cx_px,cy_px),c,cv2.MARKER_CROSS,24,2)
                _tbg(vis, aff.get("grasp_type","?"), (cx_px+10,cy_px), color=c)

    elif mode == "scene_graph":
        for e in result.get("edges", []):
            _tbg(vis, f"{e['subject']} {e['relation']} {e['object']}", (8, H-30), scale=0.4)

    elif mode in ("grasp_success", "placement_verify"):
        quality = result.get("grasp_quality") or ("OK" if result.get("placement_correct") else "FAIL")
        color = (0,255,0) if quality in ("secure","OK") else (0,0,255)
        _tbg(vis, f"Result: {quality}", (W//2-60, H//2), scale=0.9, color=color)
        rec = result.get("recommendation") or result.get("correction_needed","")
        if rec: _tbg(vis, str(rec), (8, H-30), scale=0.45, color=(200,200,255))

    elif mode == "anomaly_detect":
        status = result.get("scene_status","normal")
        sc = {"normal":(0,255,0),"warning":(0,165,255),"danger":(0,0,255)}.get(status,(200,200,200))
        _tbg(vis, f"STATUS: {status.upper()}", (8, 25), scale=0.7, color=sc)
        for a in result.get("anomalies",[]):
            b = a.get("bbox_norm",[])
            if len(b)==4:
                x1,y1,x2,y2=_n2p(*b,W,H)
                cv2.rectangle(vis,(x1,y1),(x2,y2),(0,0,255),3)
                _tbg(vis,a.get("type","anomaly"),(x1,max(y1-5,15)),color=(0,0,255))

    elif mode == "medication_verify":
        vr = result.get("verification_result","?")
        vc = {"pass":(0,255,0),"fail":(0,0,255),"partial":(0,165,255)}.get(vr,(200,200,200))
        _tbg(vis, f"VERIFY: {vr.upper()}", (8, 28), scale=0.8, color=vc)

    elif mode == "human_intent":
        intent = result.get("human_intent","?")
        action = result.get("robot_action","?")
        _tbg(vis, f"Intent: {intent}", (8, 25), scale=0.5, color=(255,200,100))
        _tbg(vis, f"Robot: {action}", (8, 48), scale=0.5, color=(100,255,200))

    elif mode == "progress_monitor":
        pct = result.get("progress_percent",0)
        bar_w = int((W-20) * pct / 100)
        cv2.rectangle(vis,(10,H-25),(W-10,H-10),(50,50,50),-1)
        cv2.rectangle(vis,(10,H-25),(10+bar_w,H-10),(0,255,100),-1)
        _tbg(vis,f"Progress: {pct}%",(12,H-12),scale=0.45,color=(0,0,0),bg=(0,255,100))

    # 헤더
    latency = result.get("_latency_s",0)
    guide_name = SETUP_GUIDE.get(mode,{}).get("name","")
    _tbg(vis, f"[{mode}] {guide_name} | {latency:.2f}s", (8, H-8), scale=0.4, color=(200,200,255))
    if result.get("parse_error"):
        _tbg(vis,"JSON PARSE ERROR",(8,48),scale=0.6,color=(0,0,255))

    return vis

# ══════════════════════════════════════════════════════════════════════════════
# 세팅 가이드 출력
# ══════════════════════════════════════════════════════════════════════════════

def print_setup_guide(mode: str = None):
    modes = [mode] if mode else list(SETUP_GUIDE.keys())
    for cat, mlist in CATEGORIES.items():
        cat_modes = [m for m in mlist if m in modes]
        if not cat_modes:
            continue
        print(f"\n{'━'*65}")
        print(f"  {CATEGORY_LABELS[cat]}")
        print(f"{'━'*65}")
        for m in cat_modes:
            g = SETUP_GUIDE[m]
            print(f"\n  [{m}]  {g['name']}")
            print(f"  {'─'*55}")
            print(f"  세팅:")
            for i, s in enumerate(g["setup"], 1):
                print(f"    {i}. {s}")
            print(f"  Tip: {g['tip']}")

# ══════════════════════════════════════════════════════════════════════════════
# 인터랙티브 루프
# ══════════════════════════════════════════════════════════════════════════════

def interactive_loop(client, model: str):
    camera = CameraCapture()
    ALL_MODES = list(PROMPTS.keys())
    mode_idx = 0
    last_vis = None
    last_result = None

    print(f"\n{'━'*65}")
    print(f"  Gemini Embodied Reasoning Interactive")
    print(f"  Model: {model}")
    print(f"{'━'*65}")
    print("  키 조작:")
    print("  [c]      현재 프레임 분석")
    print("  [Tab]    다음 모드")
    print("  [0~9/a~j] 모드 직접 선택")
    print("  [s]      결과 저장 (PNG + JSON)")
    print("  [h]      현재 모드 세팅 가이드 출력")
    print("  [b]      전체 카테고리 목록 출력")
    print("  [q]      종료")

    # 모드 번호 매핑 출력
    print(f"\n  {'─'*55}")
    for cat, mlist in CATEGORIES.items():
        print(f"  {CATEGORY_LABELS[cat]}")
        for i, m in enumerate(ALL_MODES):
            if m in mlist:
                key = str(i) if i < 10 else chr(ord('a') + i - 10)
                print(f"    [{key}] {m}  —  {SETUP_GUIDE[m]['name']}")
    print(f"  {'─'*55}\n")

    KEY_MAP = {}
    for i, m in enumerate(ALL_MODES):
        k = str(i) if i < 10 else chr(ord('a') + i - 10)
        KEY_MAP[ord(k)] = i

    out_dir = Path("./results/gemini_embodied")
    out_dir.mkdir(parents=True, exist_ok=True)

    while True:
        frame_rgb = camera.read()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        mode = ALL_MODES[mode_idx]

        display = frame_bgr.copy()
        H, W = display.shape[:2]

        # 상단 상태바
        _tbg(display, f"Mode [{mode_idx}]: {mode}  ({SETUP_GUIDE[mode]['name']})",
             (8, 18), scale=0.5, color=(0,255,255))
        _tbg(display, "c:분석  Tab:다음  h:세팅  s:저장  q:종료",
             (8, 38), scale=0.4, color=(200,200,200))

        # 마지막 결과 미니어처 (우하단)
        if last_vis is not None:
            mw, mh = W//3, H//3
            mini = cv2.resize(last_vis, (mw, mh))
            display[H-mh:, W-mw:] = mini

        cv2.imshow("Gemini Embodied Reasoning", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == 9:  # Tab
            mode_idx = (mode_idx + 1) % len(ALL_MODES)
        elif key in KEY_MAP:
            mode_idx = KEY_MAP[key]
            print(f"\n[모드 변경] → {ALL_MODES[mode_idx]}")
        elif key == ord('h'):
            print_setup_guide(mode)
        elif key == ord('b'):
            print_setup_guide()
        elif key == ord('c'):
            print(f"\n[캡처] {mode} 분석 중...")
            result = run_mode(client, model, frame_rgb, mode)
            last_result = result
            last_vis = draw_result(frame_bgr.copy(), result)
            cv2.imshow("Gemini Embodied Reasoning", last_vis)
        elif key == ord('s') and last_result is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            m = last_result.get("_mode", mode)
            cv2.imwrite(str(out_dir / f"{ts}_{m}.png"), last_vis)
            with open(out_dir / f"{ts}_{m}.json", "w") as f:
                json.dump(last_result, f, ensure_ascii=False, indent=2)
            print(f"[저장] {out_dir}/{ts}_{m}.*")

    camera.release()
    cv2.destroyAllWindows()

# ══════════════════════════════════════════════════════════════════════════════
# 벤치마크
# ══════════════════════════════════════════════════════════════════════════════

def benchmark(client, model: str, img_rgb: np.ndarray, category: str = None):
    if category:
        modes = CATEGORIES.get(category, [])
        if not modes:
            print(f"[Error] 카테고리 없음: {category}")
            print(f"  가능한 카테고리: {list(CATEGORIES.keys())}")
            return
    else:
        modes = list(PROMPTS.keys())

    print(f"\n{'━'*65}")
    print(f"  BENCHMARK  {'카테고리: '+category if category else '전체 모드'}")
    print(f"  Model: {model}  |  Modes: {len(modes)}개")
    print(f"{'━'*65}")

    summary = []
    for mode in modes:
        result = run_mode(client, model, img_rgb, mode)
        summary.append({
            "mode": mode,
            "name": SETUP_GUIDE[mode]["name"],
            "latency_s": result["_latency_s"],
            "parse_ok": not result.get("parse_error", False),
        })

    print(f"\n{'━'*65}")
    print("  RESULT SUMMARY")
    print(f"{'━'*65}")
    total = 0
    for cat, mlist in CATEGORIES.items():
        cat_results = [s for s in summary if s["mode"] in mlist]
        if not cat_results:
            continue
        print(f"\n  {CATEGORY_LABELS[cat]}")
        for s in cat_results:
            ok = "✅" if s["parse_ok"] else "❌"
            print(f"    {ok} {s['mode']:25s} {s['latency_s']:5.2f}s  {s['name']}")
            total += s["latency_s"]

    print(f"\n  {'─'*55}")
    ok_count = sum(1 for s in summary if s["parse_ok"])
    print(f"  성공: {ok_count}/{len(summary)}  |  총 시간: {total:.1f}s  |  평균: {total/len(summary):.2f}s")

    # 저장
    out_dir = Path("./results/gemini_embodied")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"benchmark_{ts}.json"
    with open(report_path, "w") as f:
        json.dump({"model": model, "summary": summary}, f, ensure_ascii=False, indent=2)
    print(f"\n  리포트 저장: {report_path}")

# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    all_modes = list(PROMPTS.keys())

    p = argparse.ArgumentParser(
        description="Gemini Embodied Reasoning 전체 테스트",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--model", default="gemini-robotics-er-1.5-preview")
    p.add_argument("--mode", default="interactive",
                   choices=["interactive", "benchmark"] + all_modes,
                   help="실행할 모드")
    p.add_argument("--category", default=None,
                   choices=list(CATEGORIES.keys()),
                   help="벤치마크 시 카테고리 지정 (기본: 전체)")
    p.add_argument("--image", default=None, help="카메라 대신 사용할 이미지 경로")
    p.add_argument("--image-before", default=None, help="state_change 모드: 이전 프레임")
    p.add_argument("--list", action="store_true", help="모드 목록 + 세팅 가이드 출력")
    args = p.parse_args()

    if args.list:
        print_setup_guide()
        return

    try:
        client = get_client()
    except EnvironmentError as e:
        print(f"[Error] {e}")
        sys.exit(1)

    def load_image(path):
        return np.array(Image.open(path).convert("RGB"))

    def get_camera_frame():
        cam = CameraCapture()
        frame = cam.read()
        cam.release()
        return frame

    if args.mode == "interactive":
        interactive_loop(client, args.model)

    elif args.mode == "benchmark":
        img = load_image(args.image) if args.image else get_camera_frame()
        benchmark(client, args.model, img, args.category)

    else:
        img = load_image(args.image) if args.image else get_camera_frame()
        extra = [load_image(args.image_before)] if args.image_before else None
        result = run_mode(client, args.model, img, args.mode, extra)
        vis = draw_result(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), result)
        cv2.imshow("Gemini Embodied Reasoning", vis)
        print("\n[아무 키나 누르면 종료]")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
