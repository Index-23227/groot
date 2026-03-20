"""
테이블 ROI 감지 + SAM2 비교 테스트

Step 0: GPT-4o로 회색 테이블 bbox 자동 감지
Step A: SAM2 전체 이미지 세그멘테이션
Step B: SAM2 테이블 ROI만 세그멘테이션
→ mask 수 / 시간 / 결과를 HTML로 비교
"""
import sys, io, base64, json, time, random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

sys.path.append(str(Path(__file__).parent.parent))
from utils.object_localizer import ObjectLocalizer, _pil_to_b64

ROOT     = Path(__file__).parent.parent
IMG_PATH = ROOT / "data" / "base_images" / "base12.jpg"
OUT_DIR  = ROOT / "results" / "table_roi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_KEY = next(l.strip() for l in (ROOT/"token").read_text().splitlines() if l.strip().startswith("sk-"))
client = OpenAI(api_key=OPENAI_KEY)

try:
    FONT_SM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    FONT_MD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
except:
    FONT_SM = FONT_MD = ImageFont.load_default()

# ── 유틸 ─────────────────────────────────────────────────────
def pil_to_b64_html(img: Image.Image, max_side: int = 900) -> str:
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def draw_roi_box(image_np, roi, color=(0, 200, 255), label="TABLE ROI"):
    img = Image.fromarray(image_np).copy()
    d = ImageDraw.Draw(img)
    x1, y1, x2, y2 = roi
    d.rectangle([x1, y1, x2, y2], outline=color, width=4)
    d.rectangle([x1, y1-28, x1+len(label)*11, y1], fill=color)
    d.text((x1+4, y1-26), label, fill=(0,0,0), font=FONT_MD)
    return img

def make_mask_overlay(image_np, masks, highlight_idx=None):
    random.seed(42)
    img_pil = Image.fromarray(image_np).convert("RGBA")
    H, W = image_np.shape[:2]
    for i, mask_data in enumerate(masks):
        color = (random.randint(60,230), random.randint(60,230), random.randint(60,230))
        if highlight_idx is not None and i == highlight_idx:
            color = (0, 220, 80)
        seg = mask_data["segmentation"].astype(bool)
        lyr = np.zeros((H, W, 4), dtype=np.uint8)
        lyr[seg] = [*color, 110]
        img_pil = Image.alpha_composite(img_pil, Image.fromarray(lyr))
        x, y, bw, bh = [int(v) for v in mask_data["bbox"]]
        d = ImageDraw.Draw(img_pil)
        d.rectangle([x, y, x+bw, y+bh], outline=(*color, 255), width=2)
        d.text((x+3, y+2), str(i), fill=(255,255,255,255), font=FONT_SM)
    return img_pil.convert("RGB")

# ── Step 0: GPT-4o로 테이블 bbox 감지 ───────────────────────
print("=" * 55)
print("  Step 0: GPT-4o 테이블 감지")
print("=" * 55)

image_np = np.array(Image.open(IMG_PATH).convert("RGB"))
H, W = image_np.shape[:2]
print(f"  이미지 크기: {W}x{H}")

# GPT-4o에 전체 이미지 전달 → 테이블 bbox 요청
img_b64 = _pil_to_b64(Image.fromarray(image_np), max_side=1024)
t0 = time.time()
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "이 이미지에서 테이블 위에 놓인 모든 조작 가능한 객체들(캔, 블록, 원기둥 등)을 "
                    "tight하게 감싸는 bounding box를 구해주세요.\n"
                    "로봇 팔, 테이블 표면 자체, 배경은 제외하고 "
                    "오직 집을 수 있는 물체들만 포함하세요.\n"
                    "물체들을 모두 포함할 수 있도록 약간의 여백(padding)을 추가하세요.\n"
                    "normalized 좌표 (0~1)로 답해주세요.\n"
                    'JSON으로만: {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, '
                    '"confidence": 0.9, "note": "설명"}'
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            },
        ],
    }],
    max_tokens=200,
    temperature=0.1,
    response_format={"type": "json_object"},
)
t_detect = time.time() - t0
raw = resp.choices[0].message.content
obj = json.loads(raw)
print(f"  GPT-4o 응답: {obj}  ({t_detect:.1f}s)")

# normalized → pixel
x1n, y1n, x2n, y2n = obj["x1"], obj["y1"], obj["x2"], obj["y2"]
TABLE_ROI = (int(x1n*W), int(y1n*H), int(x2n*W), int(y2n*H))
print(f"  테이블 ROI (pixel): {TABLE_ROI}")

img_roi_vis = draw_roi_box(image_np, TABLE_ROI,
                            label=f"TABLE ({obj.get('confidence','')} conf)")

# ── SAM2 로딩 ────────────────────────────────────────────────
localizer = ObjectLocalizer()

# ── Step A: 전체 이미지 SAM2 ─────────────────────────────────
print("\n[Step A] 전체 이미지 SAM2...")
tA = time.time()
masks_full = localizer.segment_all(image_np, table_region=None)
tA = time.time() - tA
print(f"  {len(masks_full)}개 mask  ({tA:.1f}s)")
img_full = make_mask_overlay(image_np, masks_full)

# ── Step B: 테이블 ROI SAM2 ──────────────────────────────────
print("\n[Step B] 테이블 ROI SAM2...")
tB = time.time()
masks_roi = localizer.segment_all(image_np, table_region=TABLE_ROI)
tB = time.time() - tB
print(f"  {len(masks_roi)}개 mask  ({tB:.1f}s)")

# ROI 박스를 overlay에 추가
img_roi = make_mask_overlay(image_np, masks_roi)
img_roi_with_box = img_roi.copy()
d = ImageDraw.Draw(img_roi_with_box)
x1, y1, x2, y2 = TABLE_ROI
d.rectangle([x1, y1, x2, y2], outline=(0, 200, 255), width=4)

# ── 비교 수치 ────────────────────────────────────────────────
speedup = tA / tB if tB > 0 else 0
mask_reduction = 1 - len(masks_roi) / len(masks_full) if masks_full else 0
print(f"\n비교:")
print(f"  전체: {len(masks_full)}개 / {tA:.1f}s")
print(f"  ROI:  {len(masks_roi)}개 / {tB:.1f}s")
print(f"  속도: {speedup:.1f}x  |  mask 감소: {mask_reduction*100:.0f}%")

# ── HTML 생성 ─────────────────────────────────────────────────
def card(title, img_pil, badges=None, rows=None, color="#2c3e50", max_side=900):
    b64 = pil_to_b64_html(img_pil, max_side=max_side)
    badge_html = ""
    if badges:
        for label, val, bg in badges:
            badge_html += f'<span class="badge" style="background:{bg}">{label}: <b>{val}</b></span> '
    row_html = ""
    if rows:
        row_html = "<table>"
        for k, v in rows:
            row_html += f"<tr><td class='key'>{k}</td><td class='val'>{v}</td></tr>"
        row_html += "</table>"
    return f"""
    <div class="card">
      <div class="card-header" style="background:{color}">{title}</div>
      <div class="card-body">
        <img src="{b64}" />
        <div class="meta">{badge_html}{row_html}</div>
      </div>
    </div>"""

win_color_A = "#1a5276" if tA <= tB else "#555"
win_color_B = "#1a5276" if tB < tA  else "#555"

html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Table ROI vs Full — SAM2 비교</title>
<style>
  body {{ font-family:'Segoe UI',sans-serif; background:#0f0f0f; color:#e0e0e0; margin:0; padding:20px; }}
  h1   {{ text-align:center; color:#f0f0f0; letter-spacing:2px; margin-bottom:4px; }}
  .subtitle {{ text-align:center; color:#888; margin-bottom:30px; font-size:14px; }}
  .pipeline {{ display:flex; flex-wrap:wrap; gap:20px; justify-content:center; }}
  .card {{ background:#1c1c1c; border-radius:10px; overflow:hidden;
           width:560px; box-shadow:0 4px 16px rgba(0,0,0,0.5); }}
  .card-header {{ padding:10px 16px; font-weight:bold; font-size:15px; color:#fff; }}
  .card-body {{ padding:14px; }}
  .card-body img {{ width:100%; border-radius:6px; display:block; }}
  .meta {{ margin-top:10px; }}
  .badge {{ display:inline-block; padding:3px 10px; border-radius:12px;
            font-size:13px; color:#fff; margin:3px 3px 3px 0; }}
  table {{ width:100%; border-collapse:collapse; margin-top:8px; font-size:13px; }}
  td {{ padding:5px 8px; border-bottom:1px solid #2a2a2a; vertical-align:top; }}
  td.key {{ color:#aaa; width:36%; font-weight:bold; }}
  td.val {{ color:#e0e0e0; }}
  .summary {{ margin:28px auto; max-width:760px; background:#1c1c1c;
              border-radius:12px; padding:24px 32px; text-align:center; }}
  .summary h2 {{ margin:0 0 16px; color:#f0f0f0; }}
  .stat-row {{ display:flex; justify-content:center; gap:40px; flex-wrap:wrap; }}
  .stat {{ text-align:center; }}
  .stat .num {{ font-size:36px; font-weight:900; }}
  .stat .lbl {{ font-size:13px; color:#888; margin-top:4px; }}
  .green {{ color:#2ecc71; }}
  .blue  {{ color:#3498db; }}
  .gold  {{ color:#f1c40f; }}
</style>
</head>
<body>
<h1>Table ROI Detection + SAM2 비교</h1>
<div class="subtitle">
  Image: <b>{IMG_PATH.name}</b> ({W}×{H}) &nbsp;|&nbsp;
  GPT-4o 테이블 감지 → SAM2 전체 vs ROI 비교
</div>

<div class="pipeline">

{card(
    "Step 0 · GPT-4o — 테이블 영역 감지",
    img_roi_vis,
    badges=[
        ("Conf", obj.get("confidence",""), "#7d6608"),
        ("Time", f"{t_detect:.1f}s", "#555"),
    ],
    rows=[
        ("bbox (norm)", f"x1={x1n:.3f}, y1={y1n:.3f}, x2={x2n:.3f}, y2={y2n:.3f}"),
        ("bbox (pixel)", f"({TABLE_ROI[0]}, {TABLE_ROI[1]}, {TABLE_ROI[2]}, {TABLE_ROI[3]})"),
        ("Note", obj.get("note","—")),
        ("Model", "gpt-4o"),
    ],
    color="#7d6608"
)}

{card(
    f"Step A · SAM2 전체 이미지 — {len(masks_full)}개 mask",
    img_full,
    badges=[
        ("Masks", len(masks_full), "#1a5276"),
        ("Time", f"{tA:.1f}s", win_color_A),
    ],
    rows=[
        ("입력 크기", f"{W}×{H}"),
        ("points_per_side", "16"),
        ("mask 수", len(masks_full)),
        ("소요 시간", f"{tA:.2f}s"),
    ],
    color="#1a5276"
)}

{card(
    f"Step B · SAM2 테이블 ROI — {len(masks_roi)}개 mask",
    img_roi_with_box,
    badges=[
        ("Masks", len(masks_roi), "#145a32"),
        ("Time", f"{tB:.1f}s", win_color_B),
        ("Speedup", f"{speedup:.1f}×", "#1a7a1a"),
    ],
    rows=[
        ("ROI 크기", f"{TABLE_ROI[2]-TABLE_ROI[0]}×{TABLE_ROI[3]-TABLE_ROI[1]}"),
        ("points_per_side", "16"),
        ("mask 수", f"{len(masks_roi)}  (−{len(masks_full)-len(masks_roi)}개)"),
        ("소요 시간", f"{tB:.2f}s"),
    ],
    color="#145a32"
)}

</div>

<div class="summary">
  <h2>비교 요약</h2>
  <div class="stat-row">
    <div class="stat">
      <div class="num blue">{len(masks_full)} → {len(masks_roi)}</div>
      <div class="lbl">Mask 수<br>(전체 → ROI)</div>
    </div>
    <div class="stat">
      <div class="num green">{speedup:.1f}×</div>
      <div class="lbl">SAM2 속도 향상</div>
    </div>
    <div class="stat">
      <div class="num gold">{mask_reduction*100:.0f}%</div>
      <div class="lbl">불필요 mask 감소</div>
    </div>
    <div class="stat">
      <div class="num" style="color:#e67e22">{tA:.1f}s → {tB:.1f}s</div>
      <div class="lbl">SAM2 시간<br>(전체 → ROI)</div>
    </div>
  </div>
  <p style="color:#aaa;margin-top:16px;font-size:13px;">
    테이블 ROI 감지: GPT-4o ({t_detect:.1f}s) — 최초 1회만 필요 (캐시 가능)
  </p>
</div>

</body>
</html>"""

out_path = OUT_DIR / f"{IMG_PATH.stem}_table_roi.html"
out_path.write_text(html, encoding="utf-8")
print(f"\n저장: {out_path}")
