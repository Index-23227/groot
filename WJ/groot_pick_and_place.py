#!/usr/bin/env python3
"""
GR00T Pick and Place — SAM2 + GPT-4o CoT + 로봇 제어
자연어 명령 → SAM2 세그먼트 → GPT-4o 추론 → 캘리브레이션 → 로봇 실행

사용법:
  python3 ~/groot_pick_and_place.py
"""
import sys, io, base64, time, pickle, os, json
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

os.environ["QT_QPA_PLATFORM"] = "xcb"

import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from openai import OpenAI
import pyrealsense2 as rs
import subprocess

# === 경로 ===
ROOT = Path(__file__).parent / "groot"
CHECKPOINT = ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = ROOT / "results" / "sam2_cache_live"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# === 캘리브레이션 데이터 ===
camera_points = np.array([
    [207.3, -201.0, 710.0], [2.4, -111.9, 657.0],
    [-105.9, -241.9, 734.0], [-147.5, -153.5, 681.0],
    [127.3, -86.5, 644.0], [-198.6, -17.0, 605.0],
    [211.9, 18.5, 586.0], [129.1, -112.9, 598.0],
    [-73.5, -47.8, 563.0], [84.1, -269.8, 683.0],
    [86.2, -267.1, 685.0], [-99.3, -91.0, 586.0],
    [81.9, -261.9, 710.0], [-99.8, -80.0, 606.0],
    [84.2, -244.7, 731.0], [-100.8, -65.5, 630.0],
    [77.8, -230.3, 763.0], [-99.6, -49.3, 661.0],
    [77.8, -226.6, 775.0], [-100.0, -45.8, 671.0],
    [77.9, -217.2, 776.0], [-99.9, -38.9, 677.0],
    [78.3, -214.2, 792.0], [-99.4, -36.3, 688.0],
], dtype=np.float64)

robot_points = np.array([
    [400.0, 200.0, 200.0], [500.0, 0.0, 200.0],
    [350.0, -100.0, 200.0], [450.0, -150.0, 200.0],
    [530.0, 120.0, 200.0], [600.0, -200.0, 200.0],
    [650.0, 200.0, 200.0], [530.0, 120.0, 260.0],
    [600.0, -80.0, 260.0], [350.0, 80.0, 260.0],
    [350.0, 80.0, 260.0], [550.0, -100.0, 260.0],
    [350.0, 80.0, 220.0], [550.0, -100.0, 220.0],
    [350.0, 80.0, 200.0], [550.0, -100.0, 200.0],
    [350.0, 80.0, 180.0], [550.0, -100.0, 180.0],
    [350.0, 80.0, 170.0], [550.0, -100.0, 170.0],
    [350.0, 80.0, 160.0], [550.0, -100.0, 160.0],
    [350.0, 80.0, 150.0], [550.0, -100.0, 150.0],
], dtype=np.float64)

# === 설정 ===
CAN_Z_THRESHOLD = 180
GRIP_CAN = 500
GRIP_BLOCK = 550
SAFE_Z = 400
MIN_Z = 120
MAX_REACH = 800
current_pos = [453.0, 0.0, 400.0]


# === 캘리브레이션 ===
def compute_transform(cam_pts, rob_pts):
    n = cam_pts.shape[0]
    A = np.hstack([cam_pts, np.ones((n, 1))])
    T, _, _, _ = np.linalg.lstsq(A, rob_pts, rcond=None)
    return T

def cam_to_robot(cam_xyz, T):
    return np.append(cam_xyz, 1.0) @ T


# === 로봇 제어 ===
def move_line_cmd(x, y, z):
    if z < MIN_Z:
        z = MIN_Z
    cmd = (
        f'ros2 service call /dsr01/motion/move_line dsr_msgs2/srv/MoveLine '
        f'"{{pos: [{x:.1f}, {y:.1f}, {z:.1f}, 3.4, -180.0, 93.4], '
        f'vel: [80.0, 80.0], acc: [80.0, 80.0]}}"'
    )
    subprocess.run(cmd, shell=True, env=os.environ.copy(), capture_output=True)
    time.sleep(0.5)

def move_robot(x, y, z):
    if z < MIN_Z:
        z = MIN_Z
    print(f"    ↑ z상승 ({SAFE_Z}mm)")
    move_line_cmd(current_pos[0], current_pos[1], SAFE_Z)
    current_pos[2] = SAFE_Z

    print(f"    → xy이동 (x={x:.1f}, y={y:.1f})")
    move_line_cmd(x, y, SAFE_Z)
    current_pos[0] = x
    current_pos[1] = y

    print(f"    ↓ z하강 ({z:.1f}mm)")
    move_line_cmd(x, y, z)
    current_pos[2] = z

def gripper_control(position):
    position = max(0, min(700, position))
    cmd = (
        f'ros2 topic pub /dsr01/gripper/position_cmd std_msgs/msg/Int32 '
        f'"{{data: {position}}}" --once'
    )
    subprocess.run(cmd, shell=True, env=os.environ.copy(), capture_output=True)

def go_home():
    print("    홈 복귀...")
    cmd = (
        'ros2 service call /dsr01/motion/move_joint dsr_msgs2/srv/MoveJoint '
        '"{pos: [0.0, 0.0, 90.0, 0.0, 90.0, 90.0], vel: 30.0, acc: 30.0}"'
    )
    subprocess.run(cmd, shell=True, env=os.environ.copy(), capture_output=True)
    subprocess.run(cmd, shell=True, env=os.environ.copy(), capture_output=True)
    current_pos[0] = 453.0
    current_pos[1] = 0.0
    current_pos[2] = 400.0


# === SAM2 유틸 ===
def make_crop(image, mask_data, padding=12):
    mask = mask_data["segmentation"].astype(bool)
    h, w = image.shape[:2]
    x, y, bw, bh = [int(v) for v in mask_data["bbox"]]
    x1, y1 = max(0, x-padding), max(0, y-padding)
    x2, y2 = min(w, x+bw+padding), min(h, y+bh+padding)
    arr = np.array(Image.fromarray(image).convert("RGBA"))
    arr[~mask] = [255, 255, 255, 255]
    return Image.fromarray(arr).convert("RGB").crop((x1, y1, x2, y2))

def _to_b64(img, max_side=512):
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def compute_top_surface_ee(mask_data, depth_np):
    mask_seg = mask_data["segmentation"]
    ys, xs = np.where(mask_seg)
    if len(ys) == 0:
        return None, None
    depths = depth_np[ys, xs]
    valid = depths > 0
    if valid.sum() == 0:
        return None, None
    ys, xs, depths = ys[valid], xs[valid], depths[valid]

    d_min = depths.min()
    d_range = depths.max() - d_min
    near_th = d_min + max(d_range * 0.05, 3)
    near = depths <= near_th
    pt_a = (int(np.mean(xs[near])), int(np.mean(ys[near])))

    y_min = ys.min()
    y_th = y_min + max((ys.max() - y_min) * 0.05, 3)
    top_y = ys <= y_th
    pt_b = (int(np.mean(xs[top_y])), int(np.mean(ys[top_y])))

    ee_x = (pt_a[0] + pt_b[0]) // 2
    ee_y = (pt_a[1] + pt_b[1]) // 2
    depth_mm = float(depths[near].mean())

    return (ee_x, ee_y), depth_mm


# === GPT-4o CoT ===
def identify_cot_with_crops(client, crops, instruction):
    prompt = (
        f"지시: {instruction}\n\n"
        f"아래 {len(crops)}개 이미지는 각각 씬에서 분리된 개별 객체입니다.\n"
        "배경은 흰색으로 처리되어 있고, 객체만 보입니다.\n\n"
        "이 지시는 pick-and-place 태스크입니다.\n"
        "단계별로 추론하세요:\n\n"
        "Step 1. PICK 대상: 집어야 할 객체의 번호(0-based)\n"
        "Step 2. PICK 윗면: 해당 crop에서 윗면 중심점 (normalized 0~1)\n"
        "Step 3. PLACE 참조: 놓을 위치의 참조 객체 번호(0-based), 없으면 null\n"
        "Step 4. PLACE 위치: 참조 객체 crop에서 놓을 위치 (normalized 0~1)\n"
        "Step 5. ACTION: action sequence\n\n"
        "반드시 아래 JSON 형식으로만 답하세요:\n"
        '{"step1_pick_index": 0,'
        ' "step1_pick_description": "설명",'
        ' "step2_pick_top_center": {"u": 0.5, "v": 0.3},'
        ' "step3_place_ref_index": 1,'
        ' "step3_place_ref_description": "설명",'
        ' "step4_place_position": {"u": 0.5, "v": 0.5},'
        ' "step4_place_relation": "옆에",'
        ' "step5_actions": ["move_to_pick", "grasp", "lift", "move_to_place", "release"],'
        ' "confidence": 0.9}'
    )
    content = [{"type": "text", "text": prompt}]
    for i, c in enumerate(crops):
        content += [
            {"type": "text", "text": f"[{i}번]"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{_to_b64(c)}"}},
        ]

    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model="gpt-4o", temperature=0.1, max_tokens=800,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": content}],
            )
            raw = r.choices[0].message.content
            s, e = raw.find("{"), raw.rfind("}") + 1
            return json.loads(raw[s:e]) if s >= 0 and e > 0 else {}
        except Exception as e:
            wait = 15 if "429" in str(e) else 8
            print(f"    retry {attempt+1}/3 ({wait}s)")
            time.sleep(wait)
    return {}


def crop_to_original(mask_data, crop_pil, u_norm, v_norm, padding=12):
    bx, by, bw, bh = [int(v) for v in mask_data["bbox"]]
    cx1 = max(0, bx - padding)
    cy1 = max(0, by - padding)
    ox = cx1 + int(u_norm * crop_pil.width)
    oy = cy1 + int(v_norm * crop_pil.height)
    return (ox, oy)


# === 메인 ===
print("=" * 55)
print("  GR00T Pick and Place")
print("  SAM2 + GPT-4o CoT + Doosan E0509")
print("=" * 55)

# 캘리브레이션
T = compute_transform(camera_points, robot_points)
avg_err = np.mean([np.linalg.norm(cam_to_robot(camera_points[i], T) - robot_points[i])
                    for i in range(len(camera_points))])
print(f"캘리브레이션 평균 오차: {avg_err:.1f}mm")

# SAM2 로드
print(f"\nSAM2 로딩... (device={DEVICE})")
t_load = time.time()
sam2_model = build_sam2(SAM2_CFG, str(CHECKPOINT), device=DEVICE)
generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=16,
    pred_iou_thresh=0.80,
    stability_score_thresh=0.90,
    min_mask_region_area=800,
)
print(f"SAM2 로드 완료 ({time.time()-t_load:.1f}s)")

# OpenAI 클라이언트
client = OpenAI()

# 리얼센스 카메라
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

for _ in range(30):
    pipeline.wait_for_frames()

print("\n카메라 준비 완료!")
print("명령어를 입력하세요 (q=종료):")
print("  예: 파란색 캔을 초록색 캔 옆에 놓아라\n")

cycle_count = 0

try:
    while True:
        instruction = input("\n명령> ").strip()
        if instruction.lower() == 'q':
            break
        if not instruction:
            continue

        t_total = time.time()

        # 1. 카메라 캡처
        print("\n[1/6] 카메라 캡처...")
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            print("카메라 프레임 실패")
            continue

        rgb_np = np.asanyarray(color_frame.get_data())
        # BGR → RGB for SAM2
        rgb_for_sam = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
        depth_np = np.asanyarray(depth_frame.get_data())
        H, W = rgb_np.shape[:2]
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        # 현재 장면 저장
        cv2.imwrite(os.path.expanduser(f"~/groot_scene_{cycle_count}.jpg"), rgb_np)

        # 2. SAM2 세그먼테이션
        print("[2/6] SAM2 세그먼테이션...")
        t_sam = time.time()

        # ROI: 가운데 영역
        ROI_X1 = W // 4
        ROI_X2 = 3 * W // 4
        roi_rgb = rgb_for_sam[:, ROI_X1:ROI_X2]

        raw_masks = sorted(generator.generate(roi_rgb), key=lambda x: x["area"], reverse=True)

        # ROI → 원본 좌표 변환
        masks = []
        for m in raw_masks:
            seg_full = np.zeros((H, W), dtype=bool)
            seg_full[:, ROI_X1:ROI_X2] = m["segmentation"]
            bx, by, bw, bh = m["bbox"]
            masks.append({
                **m,
                "segmentation": seg_full,
                "bbox": (bx + ROI_X1, by, bw, bh),
            })

        print(f"  {len(masks)}개 객체 감지 ({time.time()-t_sam:.1f}s)")

        if len(masks) == 0:
            print("  객체 감지 실패!")
            continue

        # 3. Crops 생성
        crops = [make_crop(rgb_for_sam, m) for m in masks]

        # 4. GPT-4o CoT
        print(f"[3/6] GPT-4o CoT 추론 ({len(crops)}개 crops)...")
        t_cot = time.time()
        cot_res = identify_cot_with_crops(client, crops, instruction)
        print(f"  추론 완료 ({time.time()-t_cot:.1f}s)")

        if not cot_res:
            print("  GPT-4o 추론 실패!")
            continue

        pick_idx = min(int(cot_res.get("step1_pick_index", 0)), len(masks)-1)
        pick_mask = masks[pick_idx]
        pick_crop = crops[pick_idx]

        place_ref_idx = cot_res.get("step3_place_ref_index")
        place_mask = None
        place_crop = None
        if place_ref_idx is not None:
            place_ref_idx = min(int(place_ref_idx), len(masks)-1)
            place_mask = masks[place_ref_idx]
            place_crop = crops[place_ref_idx]

        print(f"  PICK:  [{pick_idx}번] {cot_res.get('step1_pick_description', '')}")
        print(f"  PLACE: [{place_ref_idx}번] {cot_res.get('step3_place_ref_description', '')} ({cot_res.get('step4_place_relation', '')})")

        # 5. EE 좌표 계산
        print("[4/6] EE 좌표 계산...")

        # Pick EE (depth 기반 윗면 중심)
        pick_ee_result = compute_top_surface_ee(pick_mask, depth_np)
        if pick_ee_result[0] is not None:
            pick_ee, pick_depth_mm = pick_ee_result
        else:
            # fallback: crop norm
            pick_tc = cot_res.get("step2_pick_top_center", {"u": 0.5, "v": 0.5})
            pick_ee = crop_to_original(pick_mask, pick_crop,
                                        float(pick_tc.get("u", 0.5)),
                                        float(pick_tc.get("v", 0.5)))
            pick_depth_mm = depth_np[pick_ee[1], pick_ee[0]]

        # Pick: 픽셀 → 카메라 3D → 로봇 좌표
        pick_depth_m = pick_depth_mm / 1000.0 if pick_depth_mm > 100 else \
                       depth_frame.get_distance(pick_ee[0], pick_ee[1])
        pick_3d = rs.rs2_deproject_pixel_to_point(intrinsics, list(pick_ee), pick_depth_m)
        pick_cam = np.array([pick_3d[0]*1000, pick_3d[1]*1000, pick_3d[2]*1000])
        pick_robot = cam_to_robot(pick_cam, T)

        print(f"  Pick EE: pixel={pick_ee}, depth={pick_depth_mm:.0f}mm")
        print(f"  Pick 로봇: x={pick_robot[0]:.1f}, y={pick_robot[1]:.1f}, z={pick_robot[2]:.1f}")

        # Place EE
        place_robot = None
        if place_mask is not None and place_crop is not None:
            place_tc = cot_res.get("step4_place_position", {"u": 0.5, "v": 0.5})
            place_ee = crop_to_original(place_mask, place_crop,
                                         float(place_tc.get("u", 0.5)),
                                         float(place_tc.get("v", 0.5)))
            place_ee_x = max(0, min(W-1, place_ee[0]))
            place_ee_y = max(0, min(H-1, place_ee[1]))
            place_depth_m = depth_frame.get_distance(place_ee_x, place_ee_y)
            if place_depth_m > 0:
                place_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [place_ee_x, place_ee_y], place_depth_m)
                place_cam = np.array([place_3d[0]*1000, place_3d[1]*1000, place_3d[2]*1000])
                place_robot = cam_to_robot(place_cam, T)
                print(f"  Place 로봇: x={place_robot[0]:.1f}, y={place_robot[1]:.1f}, z={place_robot[2]:.1f}")

        # 6. 도달 가능 확인
        pick_dist = np.sqrt(pick_robot[0]**2 + pick_robot[1]**2)
        if pick_dist > MAX_REACH:
            print(f"  Pick 도달 불가! 거리={pick_dist:.0f}mm")
            continue

        if pick_robot[2] < MIN_Z:
            pick_robot[2] = MIN_Z

        # 물체 타입
        if pick_robot[2] >= CAN_Z_THRESHOLD:
            obj_type, grip_val = "캔", GRIP_CAN
        else:
            obj_type, grip_val = "블럭", GRIP_BLOCK

        print(f"  물체: {obj_type} → 그리퍼: {grip_val}")

        # === 로봇 실행 ===
        print(f"\n[5/6] 로봇 실행!")

        # Pick
        print(f"  [Pick] 이동...")
        move_robot(pick_robot[0], pick_robot[1], pick_robot[2])

        print(f"  [Grip] 잡기 ({grip_val})...")
        gripper_control(grip_val)
        time.sleep(1)
        gripper_control(grip_val)
        time.sleep(1)

        # Place
        if place_robot is not None:
            pick_z = pick_robot[2]
            place_z = place_robot[2]
            place_z = place_z + (pick_z - MIN_Z) if place_z > MIN_Z else pick_z
            place_z += 20
            if obj_type == "블럭":
                place_z += 10
            if place_z < MIN_Z:
                place_z = MIN_Z

            place_dist = np.sqrt(place_robot[0]**2 + place_robot[1]**2)
            if place_dist > MAX_REACH:
                print(f"  Place 도달 불가 → 기본 위치 사용")
                place_robot = np.array([450.0, 0.0, 200.0])
                place_z = 200.0

            print(f"  [Place] 이동 (x={place_robot[0]:.1f}, y={place_robot[1]:.1f}, z={place_z:.1f})...")
            move_robot(place_robot[0], place_robot[1], place_z)
        else:
            print(f"  [Place] 기본 위치...")
            move_robot(450.0, 0.0, 200.0)

        print(f"  [Release] 그리퍼 열기...")
        gripper_control(0)
        time.sleep(1)
        gripper_control(0)
        time.sleep(1)

        # 홈 복귀
        print(f"\n[6/6] 홈 복귀")
        go_home()

        cycle_count += 1
        t_total = time.time() - t_total
        print(f"\n{'='*55}")
        print(f"  사이클 {cycle_count} 완료! ({t_total:.1f}s)")
        print(f"  {cot_res.get('step1_pick_description', '')} → {cot_res.get('step4_place_relation', '')} {cot_res.get('step3_place_ref_description', '')}")
        print(f"{'='*55}")

finally:
    pipeline.stop()
    print(f"\n총 {cycle_count}개 사이클 완료")
