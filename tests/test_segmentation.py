"""
Gemini ER 1.5 — Segmentation Test
instruction 기반 타겟 객체(스프라이트 캔)만 polygon segmentation
"""
import sys, json, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

token = Path(__file__).parent.parent / "token"
google_key = next(l.strip() for l in token.read_text().splitlines() if l.strip().startswith("AIza"))

from google import genai
from google.genai import types

client = genai.Client(api_key=google_key)

INSTRUCTION = "스프라이트 캔을 집어라"

PROMPT = (
    f"지시: {INSTRUCTION}\n\n"
    "이미지에서 스프라이트 캔만 segmentation하세요. 로봇 팔과 그리퍼는 무시하세요.\n\n"
    "캔의 외곽선을 polygon 좌표(정규화 0~1)로 최대한 정밀하게 표현하세요.\n"
    "JSON 형식으로만 답하세요:\n"
    "{\n"
    '  "label": "sprite_can",\n'
    '  "visible": true,\n'
    '  "confidence": 0.95,\n'
    '  "polygon_norm": [[x1,y1],[x2,y2],...],\n'
    '  "bbox_norm": [cx, cy, w, h]\n'
    "}"
)

img_dir = Path(__file__).parent.parent / "data" / "base_images"
out_dir = Path(__file__).parent.parent / "results" / "segmentation_vis"
out_dir.mkdir(parents=True, exist_ok=True)

try:
    font    = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
except Exception:
    font = ImageFont.load_default()
    font_lg = font

results = []

print("=" * 60)
print(f"  Gemini ER 1.5 — Segmentation (sprite can)")
print("=" * 60)

for i in range(1, 10):
    path = img_dir / f"base{i}.jpg"
    if not path.exists():
        print(f"base{i}: 파일 없음")
        continue

    pil = Image.open(path)
    t0  = time.time()

    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model="gemini-robotics-er-1.5-preview",
                contents=[PROMPT, pil],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            break
        except Exception as e:
            wait = 15 if "429" in str(e) else 10
            print(f"  재시도 {attempt+1}/3 ({wait}s 대기)...")
            time.sleep(wait)

    elapsed = time.time() - t0
    text = resp.text.strip()
    start, end = text.find("{"), text.rfind("}") + 1
    try:
        obj = json.loads(text[start:end])
    except Exception:
        obj = None

    entry = {"image": f"base{i}", "latency_s": round(elapsed, 1), "result": obj}
    results.append(entry)

    # --- 시각화 ---
    img = pil.convert("RGB")
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    visible = obj.get("visible", False) if obj else False
    polygon = obj.get("polygon_norm") if obj else None
    bbox    = obj.get("bbox_norm") if obj else None
    conf    = obj.get("confidence", 0) if obj else 0

    if visible and polygon and len(polygon) >= 3:
        # polygon 픽셀 좌표 변환
        pts = [(int(x * img_w), int(y * img_h)) for x, y in polygon]

        # 반투명 마스크
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        odraw.polygon(pts, fill=(0, 220, 80, 80))   # 초록 반투명
        odraw.polygon(pts, outline=(0, 255, 80, 255))  # 초록 외곽선
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

        # 외곽선 두껍게
        for j in range(len(pts)):
            p1 = pts[j]
            p2 = pts[(j + 1) % len(pts)]
            draw.line([p1, p2], fill=(0, 255, 80), width=3)

        # 중심점
        if bbox:
            cx_px = int(bbox[0] * img_w)
            cy_px = int(bbox[1] * img_h)
            r = 6
            draw.ellipse([cx_px-r, cy_px-r, cx_px+r, cy_px+r], fill=(255, 255, 0))

        print(f"base{i}: ✅ {len(polygon)}점 polygon  conf={conf}  ({elapsed:.1f}s)")
    else:
        print(f"base{i}: ❌ 캔 없음  ({elapsed:.1f}s)")

    # 상단 패널
    panel_h = 55
    panel = Image.new("RGB", (img_w, panel_h), (20, 20, 20))
    pdraw = ImageDraw.Draw(panel)

    status_color = (0, 220, 80) if visible else (200, 60, 60)
    status_text  = f"SEGMENTED ({len(polygon)}pts)" if (visible and polygon) else "NOT FOUND"
    pdraw.text((10, 8),  f"base{i}".upper(),   fill=(220, 220, 220), font=font_lg)
    pdraw.text((100, 8), status_text,           fill=status_color,    font=font_lg)
    pdraw.text((10, 33), f"지시: {INSTRUCTION}", fill=(160, 160, 160), font=font)
    pdraw.text((img_w - 110, 8), f"{elapsed:.1f}s", fill=(180, 180, 180), font=font_lg)

    combined = Image.new("RGB", (img_w, img_h + panel_h))
    combined.paste(panel, (0, 0))
    combined.paste(img,   (0, panel_h))

    out_path = out_dir / f"base{i}_seg.jpg"
    combined.save(out_path, quality=92)

# 결과 저장
out_json = Path(__file__).parent.parent / "results" / "auto_eval" / "segmentation_test.json"
out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(f"\n저장 완료: {out_dir}")
print(f"JSON: {out_json}")
