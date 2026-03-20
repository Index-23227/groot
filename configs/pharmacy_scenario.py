"""
약국 조제 보조 시나리오 정의

물체, 조제함 위치, instruction 셋, 시뮬레이션 waypoint 등
시나리오에 필요한 모든 상수를 한곳에 정의.
"""

import numpy as np

# =============================================================
# 약품 정의
# =============================================================

MEDICINES = {
    "blue_bottle": {
        "name_ko": "파란 약병",
        "name_en": "blue medicine bottle",
        "color": "blue",
        "sim_position": np.deg2rad([10, -20, -50, 0, 60, 0]),  # 시뮬레이션용 joint pose
    },
    "red_bottle": {
        "name_ko": "빨간 약병",
        "name_en": "red medicine bottle",
        "color": "red",
        "sim_position": np.deg2rad([20, -15, -55, 0, 65, 0]),
    },
    "white_bottle": {
        "name_ko": "하얀 약병",
        "name_en": "white medicine bottle",
        "color": "white",
        "sim_position": np.deg2rad([-5, -25, -45, 0, 55, 0]),
    },
    "yellow_box": {
        "name_ko": "노란 약상자",
        "name_en": "yellow medicine box",
        "color": "yellow",
        "sim_position": np.deg2rad([15, -10, -60, 0, 70, 0]),
    },
}

# =============================================================
# 조제함 슬롯 정의
# =============================================================

DISPENSING_SLOTS = {
    1: {
        "name_ko": "조제함 1번",
        "name_en": "dispensing slot 1",
        "sim_position": np.deg2rad([40, -30, -40, 10, 80, 0]),
    },
    2: {
        "name_ko": "조제함 2번",
        "name_en": "dispensing slot 2",
        "sim_position": np.deg2rad([50, -30, -40, 10, 80, 0]),
    },
    3: {
        "name_ko": "조제함 3번",
        "name_en": "dispensing slot 3",
        "sim_position": np.deg2rad([60, -30, -40, 10, 80, 0]),
    },
}

# =============================================================
# 시나리오 (instruction + 정답 매핑)
# =============================================================

SCENARIOS = [
    {
        "id": "basic_single",
        "description": "기본 — 약병 하나 옮기기",
        "instruction": "Pick up the blue medicine bottle and place it in dispensing slot 1",
        "instruction_ko": "파란 약병을 조제함 1번에 넣어줘",
        "pick": "blue_bottle",
        "place": 1,
    },
    {
        "id": "basic_red",
        "description": "기본 — 빨간 약병",
        "instruction": "Pick up the red medicine bottle and place it in dispensing slot 2",
        "instruction_ko": "빨간 약병을 조제함 2번에 넣어줘",
        "pick": "red_bottle",
        "place": 2,
    },
    {
        "id": "basic_white",
        "description": "기본 — 하얀 약병",
        "instruction": "Pick up the white medicine bottle and place it in dispensing slot 3",
        "instruction_ko": "하얀 약병을 조제함 3번에 넣어줘",
        "pick": "white_bottle",
        "place": 3,
    },
    {
        "id": "different_slot",
        "description": "같은 약 다른 슬롯",
        "instruction": "Pick up the blue medicine bottle and place it in dispensing slot 3",
        "instruction_ko": "파란 약병을 조제함 3번에 넣어줘",
        "pick": "blue_bottle",
        "place": 3,
    },
    {
        "id": "box_item",
        "description": "약상자 옮기기",
        "instruction": "Pick up the yellow medicine box and place it in dispensing slot 1",
        "instruction_ko": "노란 약상자를 조제함 1번에 넣어줘",
        "pick": "yellow_box",
        "place": 1,
    },
]

# =============================================================
# 시뮬레이션 파라미터
# =============================================================

SIM_HOME_POSITION = np.deg2rad([0, 0, -90, 0, 90, 0])
SIM_PRE_GRASP_OFFSET = np.deg2rad([0, 5, 5, 0, -5, 0])  # pick 위치 위로 살짝
SIM_LIFT_OFFSET = np.deg2rad([0, 10, 10, 0, -10, 0])     # 들어올림
SIM_GRASP_STEPS = 30    # grasp 접근에 필요한 스텝
SIM_TRANSFER_STEPS = 40 # pick → place 이동 스텝
SIM_PLACE_STEPS = 20    # place + release 스텝
SIM_NOISE_SCALE = 0.003 # action noise (rad)

# =============================================================
# 데모 수집 instruction 템플릿
# =============================================================

DEMO_INSTRUCTIONS = [
    "Pick up the {color} medicine bottle and place it in dispensing slot {slot}",
    "Move the {color} bottle to slot {slot}",
    "Place the {color} medicine in dispensing slot {slot}",
]


def get_scenario(scenario_id: str) -> dict:
    for s in SCENARIOS:
        if s["id"] == scenario_id:
            return s
    raise ValueError(f"Unknown scenario: {scenario_id}. Available: {[s['id'] for s in SCENARIOS]}")


def get_all_instructions() -> list:
    """데모 수집 시 사용할 전체 instruction 목록"""
    instructions = []
    for med_key, med in MEDICINES.items():
        for slot_id in DISPENSING_SLOTS:
            for tmpl in DEMO_INSTRUCTIONS:
                instructions.append(tmpl.format(color=med["color"], slot=slot_id))
    return instructions


if __name__ == "__main__":
    print("=== Pharmacy Scenario ===\n")
    print("Medicines:")
    for k, v in MEDICINES.items():
        print(f"  {k}: {v['name_ko']} ({v['name_en']})")
    print(f"\nDispensing Slots: {list(DISPENSING_SLOTS.keys())}")
    print(f"\nScenarios:")
    for s in SCENARIOS:
        print(f"  [{s['id']}] {s['description']}")
        print(f"    EN: {s['instruction']}")
        print(f"    KO: {s['instruction_ko']}")
    print(f"\nTotal demo instruction variants: {len(get_all_instructions())}")
