"""
Gemini ER 1.5 bounding box 시각화
graspability_test.json의 can_bbox_norm / gripper_bbox_norm을 base 이미지에 그려서 저장
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# 경로 설정
root     = Path(__file__).parent.parent
img_dir  = root / "data" / "base_images"
json_path = root / "results" / "auto_eval" / "graspability_test.json"
out_dir  = root / "results" / "bbox_vis"
out_dir.mkdir(parents=True, exist_ok=True)

results = json.loads(json_path.read_text())

# Ground truth (실제 파지 성공 여부)
GT = {
    "base1": True,  "base2": False, "base3": False,
    "base4": True,  "base5": False, "base6": False,
    "base7": True,  "base8": False, "base9": False,
}

# 색상 정의
COLOR_CAN     = (0, 200, 80)    # 초록: 캔
COLOR_GRIPPER = (30, 144, 255)  # 파랑: 그리퍼
COLOR_FP      = (255, 80, 80)   # 빨강: FP (Gemini가 틀린 경우)
COLOR_TEXT    = (255, 255, 255)
COLOR_BG_TP   = (0, 180, 60)
COLOR_BG_TN   = (50, 50, 200)
COLOR_BG_FP   = (200, 40, 40)
COLOR_BG_FN   = (200, 160, 0)

def draw_bbox_norm(draw, bbox_norm, img_w, img_h, color, label, line_width=3):
    """normalized bbox [cx, cy, w, h] → 이미지에 사각형 그리기"""
    cx, cy, bw, bh = bbox_norm
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)

    # 박스
    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

    # 라벨 배경
    font_size = max(14, img_h // 30)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox_text = draw.textbbox((x1, y1 - font_size - 4), label, font=font)
    draw.rectangle(bbox_text, fill=color)
    draw.text((x1, y1 - font_size - 4), label, fill=COLOR_TEXT, font=font)

    return (x1, y1, x2, y2)

def draw_center_cross(draw, bbox_norm, img_w, img_h, color, size=10):
    """bbox 중심에 십자 마커"""
    cx, cy = bbox_norm[0], bbox_norm[1]
    px, py = int(cx * img_w), int(cy * img_h)
    draw.line([(px - size, py), (px + size, py)], fill=color, width=2)
    draw.line([(px, py - size), (px, py + size)], fill=color, width=2)

def verdict_info(gemini_graspable, gt):
    if gemini_graspable is True  and gt is True:  return "TP", COLOR_BG_TP
    if gemini_graspable is False and gt is False: return "TN", COLOR_BG_TN
    if gemini_graspable is True  and gt is False: return "FP", COLOR_BG_FP
    return "FN", COLOR_BG_FN

print("Bounding box 시각화 시작...")

for r in results:
    img_name = r["image"]
    path = img_dir / f"{img_name}.jpg"
    if not path.exists():
        print(f"  {img_name}: 이미지 없음")
        continue

    img = Image.open(path).convert("RGB")
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)

    can_bbox     = r.get("can_bbox_norm")
    gripper_bbox = r.get("gripper_bbox_norm")
    graspable    = r.get("graspable")
    confidence   = r.get("confidence", 0)
    gt_val       = GT[img_name]
    verdict, verdict_color = verdict_info(graspable, gt_val)

    # 그리퍼 색상: FP이면 빨강, 아니면 파랑
    gripper_color = COLOR_FP if verdict == "FP" else COLOR_GRIPPER

    # 캔 bbox
    if can_bbox:
        draw_bbox_norm(draw, can_bbox, img_w, img_h, COLOR_CAN, "CAN", line_width=3)
        draw_center_cross(draw, can_bbox, img_w, img_h, COLOR_CAN)

    # 그리퍼 bbox
    if gripper_bbox:
        draw_bbox_norm(draw, gripper_bbox, img_w, img_h, gripper_color, "GRIPPER", line_width=3)
        draw_center_cross(draw, gripper_bbox, img_w, img_h, gripper_color)

    # 중심 정렬선 (수직)
    if can_bbox and gripper_bbox:
        can_cx = int(can_bbox[0] * img_w)
        grip_cx = int(gripper_bbox[0] * img_w)
        can_cy  = int(can_bbox[1] * img_h)
        grip_cy = int(gripper_bbox[1] * img_h)
        # 수평 오프셋 선
        draw.line([(grip_cx, grip_cy), (can_cx, can_cy)], fill=(255, 215, 0), width=2)
        # X축 기준선
        draw.line([(can_cx, 0), (can_cx, img_h)], fill=(100, 100, 100), width=1)

    # 상단 정보 패널
    try:
        font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font_lg = ImageFont.load_default()
        font_sm = font_lg

    panel_h = 90
    panel = Image.new("RGB", (img_w, panel_h), (30, 30, 30))
    pdraw = ImageDraw.Draw(panel)

    graspable_str = "Graspable ✓" if graspable else "Not Graspable ✗"
    gt_str = "GT: ✓" if gt_val else "GT: ✗"

    # verdict 배경
    pdraw.rectangle([0, 0, 110, panel_h], fill=verdict_color)
    pdraw.text((8, 10), verdict, fill=COLOR_TEXT, font=font_lg)
    pdraw.text((8, 45), gt_str, fill=COLOR_TEXT, font=font_sm)
    pdraw.text((8, 65), f"conf:{confidence}", fill=COLOR_TEXT, font=font_sm)

    # Gemini 판단
    gcolor = (80, 220, 80) if graspable else (220, 80, 80)
    pdraw.text((120, 10), img_name.upper(), fill=(220, 220, 220), font=font_lg)
    pdraw.text((120, 45), graspable_str, fill=gcolor, font=font_sm)

    # Alignment / reason
    alignment = r.get("alignment", "")
    reason    = r.get("reason", "")[:60]
    pdraw.text((120, 65), f"Align: {alignment}  |  {reason}", fill=(180, 180, 180), font=font_sm)

    # X 오프셋 표시
    if can_bbox and gripper_bbox:
        x_err = abs(can_bbox[0] - gripper_bbox[0])
        y_diff = abs(can_bbox[1] - gripper_bbox[1])
        pdraw.text((img_w - 230, 10), f"X-err: {x_err*100:.1f}%", fill=(255, 215, 0), font=font_sm)
        pdraw.text((img_w - 230, 35), f"Y-diff: {y_diff:.2f}", fill=(255, 215, 0), font=font_sm)
        threshold_color = (80, 220, 80) if x_err <= 0.05 else (220, 80, 80)
        pdraw.text((img_w - 230, 60), f"thresh: 5% / 0.10", fill=threshold_color, font=font_sm)

    # 패널 + 원본 이미지 합치기
    combined = Image.new("RGB", (img_w, img_h + panel_h))
    combined.paste(panel, (0, 0))
    combined.paste(img, (0, panel_h))

    out_path = out_dir / f"{img_name}_bbox.jpg"
    combined.save(out_path, quality=92)
    print(f"  {img_name}: [{verdict}] graspable={graspable} gt={gt_val} → {out_path.name}")

print(f"\n저장 완료: {out_dir}")
print(f"총 {len(results)}장 처리")
