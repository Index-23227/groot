"""
Gemini ER 1.5 object detection 결과 시각화
gripper 제외, 객체 bbox만 base 이미지에 그려서 저장
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

root     = Path(__file__).parent.parent
img_dir  = root / "data" / "base_images"
json_path = root / "results" / "auto_eval" / "object_detection.json"
out_dir  = root / "results" / "object_bbox_vis"
out_dir.mkdir(parents=True, exist_ok=True)

results = json.loads(json_path.read_text())

CAN_COLOR     = (0, 220, 80)   # 초록: 캔
NOT_VIS_COLOR = (120, 120, 120)

def draw_bbox(draw, bbox_norm, img_w, img_h, color, label, conf, font):
    cx, cy, bw, bh = bbox_norm
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)

    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    tag = f"{label} {conf:.0%}"
    tb = draw.textbbox((x1, y1), tag, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    ty = y1 - th - 4 if y1 - th - 4 >= 0 else y1 + 2
    draw.rectangle([x1, ty - 2, x1 + tw + 6, ty + th + 2], fill=color)
    draw.text((x1 + 3, ty), tag, fill=(0, 0, 0), font=font)

try:
    font    = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
except Exception:
    font = ImageFont.load_default()
    font_lg = font

print("캔 bbox 시각화 시작...")

for entry in results:
    img_name    = entry["image"]
    obj         = entry.get("object")
    latency     = entry.get("latency_s", 0)
    instruction = entry.get("instruction", "")

    path = img_dir / f"{img_name}.jpg"
    if not path.exists():
        print(f"  {img_name}: 이미지 없음")
        continue

    img = Image.open(path).convert("RGB")
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)

    visible = obj.get("visible", False) if obj else False
    bbox    = obj.get("bbox_norm") if obj else None
    conf    = obj.get("confidence", 0) if obj else 0
    label   = obj.get("label", "sprite_can") if obj else "sprite_can"

    if visible and bbox and any(v > 0 for v in bbox):
        draw_bbox(draw, bbox, img_w, img_h, CAN_COLOR, label, conf, font)

    # 상단 패널
    panel_h = 55
    panel = Image.new("RGB", (img_w, panel_h), (20, 20, 20))
    pdraw = ImageDraw.Draw(panel)

    status_color = (0, 220, 80) if visible else (200, 60, 60)
    status_text  = "DETECTED" if visible else "NOT FOUND"
    pdraw.text((10, 8),   img_name.upper(),  fill=(220, 220, 220), font=font_lg)
    pdraw.text((110, 8),  status_text,        fill=status_color,    font=font_lg)
    pdraw.text((10, 33),  f"지시: {instruction}", fill=(160, 160, 160), font=font)
    pdraw.text((img_w - 120, 8), f"{latency}s", fill=(180, 180, 180), font=font_lg)

    combined = Image.new("RGB", (img_w, img_h + panel_h))
    combined.paste(panel, (0, 0))
    combined.paste(img,   (0, panel_h))

    out_path = out_dir / f"{img_name}_objects.jpg"
    combined.save(out_path, quality=92)
    print(f"  {img_name}: {'✅' if visible else '❌'} conf={conf} → {out_path.name}")

print(f"\n저장 완료: {out_dir}")
