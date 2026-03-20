"""
파이프라인 단계별 HTML 시각화
Step0: GPT-4o ROI 감지 (1회 고정)
Step1: SAM2 (ROI 내 segmentation)
Step2: Top-5 filter + GPT-4o 타겟 식별
Step3: 타겟 mask overlay
Step4: GPT-4o graspability reasoning
Step5: GPT-4o calibration reasoning
Final: Action output
"""
import sys, io, base64, json, time, random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).parent.parent))
from utils.object_localizer import ObjectLocalizer, _pil_to_b64

ROOT     = Path(__file__).parent.parent
IMG_PATH = ROOT / "data" / "base_images" / "base12.jpg"
OUT_DIR  = ROOT / "results" / "pipeline_html"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INSTRUCTION = "노란 원기둥을 집어라"

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

def draw_roi(image_np, roi, color=(0, 200, 255)):
    img = Image.fromarray(image_np).convert("RGB").copy()
    d = ImageDraw.Draw(img)
    x1, y1, x2, y2 = roi
    d.rectangle([x1, y1, x2, y2], outline=color, width=5)
    d.rectangle([x1, y1, x1+180, y1+30], fill=color)
    d.text((x1+6, y1+6), "TASK WORKSPACE ROI", fill=(0,0,0), font=FONT_SM)
    return img

def make_sam2_overlay(image_np, masks, roi=None):
    random.seed(42)
    img_pil = Image.fromarray(image_np).convert("RGBA")
    H, W = image_np.shape[:2]
    for i, mask_data in enumerate(masks):
        color = (random.randint(60,230), random.randint(60,230), random.randint(60,230))
        seg = mask_data["segmentation"].astype(bool)
        lyr = np.zeros((H, W, 4), dtype=np.uint8)
        lyr[seg] = [*color, 110]
        img_pil = Image.alpha_composite(img_pil, Image.fromarray(lyr))
        x, y, bw, bh = [int(v) for v in mask_data["bbox"]]
        d = ImageDraw.Draw(img_pil)
        d.rectangle([x, y, x+bw, y+bh], outline=(*color, 255), width=2)
        d.text((x+3, y+2), str(i), fill=(255,255,255,255), font=FONT_SM)
    if roi:
        d = ImageDraw.Draw(img_pil)
        x1, y1, x2, y2 = roi
        d.rectangle([x1, y1, x2, y2], outline=(0, 200, 255, 255), width=4)
    return img_pil.convert("RGB")

def make_crop_grid(filtered, target_pos):
    THUMB = 160
    n = len(filtered)
    grid = Image.new("RGB", (n * (THUMB + 8), THUMB + 44), (40, 40, 40))
    for pos, (orig_i, m, cr, score) in enumerate(filtered):
        thumb = cr.copy(); thumb.thumbnail((THUMB, THUMB))
        xo = pos * (THUMB + 8) + (THUMB - thumb.width) // 2
        grid.paste(thumb, (xo, 4))
        d = ImageDraw.Draw(grid)
        is_target = (pos == target_pos)
        border_col = (0, 220, 80) if is_target else (120, 120, 120)
        d.rectangle([pos*(THUMB+8), 0, pos*(THUMB+8)+THUMB+6, THUMB+8],
                    outline=border_col, width=4 if is_target else 1)
        label = f"[{pos}] {score:.2f}" + (" ✓ TARGET" if is_target else "")
        d.text((pos*(THUMB+8)+4, THUMB+14), label, fill=border_col, font=FONT_SM)
    return grid

def make_target_overlay(image_np, target_mask, center_norm=None, roi=None):
    img_pil = Image.fromarray(image_np).convert("RGBA")
    H, W = image_np.shape[:2]
    seg = target_mask["segmentation"].astype(bool)
    lyr = np.zeros((H, W, 4), dtype=np.uint8)
    lyr[seg] = [0, 220, 80, 160]
    img_pil = Image.alpha_composite(img_pil, Image.fromarray(lyr))
    x, y, bw, bh = [int(v) for v in target_mask["bbox"]]
    d = ImageDraw.Draw(img_pil)
    d.rectangle([x, y, x+bw, y+bh], outline=(0, 255, 80, 255), width=4)
    if center_norm:
        cx = int(center_norm[0] * W)
        cy = int(center_norm[1] * H)
        r = 14
        d.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(255, 60, 0, 255), width=4)
        d.line([(cx-24, cy), (cx+24, cy)], fill=(255, 60, 0, 255), width=3)
        d.line([(cx, cy-24), (cx, cy+24)], fill=(255, 60, 0, 255), width=3)
    if roi:
        x1, y1, x2, y2 = roi
        d.rectangle([x1, y1, x2, y2], outline=(0, 200, 255, 200), width=3)
    return img_pil.convert("RGB")

def html_card(title, img_pil, badges=None, rows=None, color="#2c3e50"):
    b64 = pil_to_b64_html(img_pil)
    badge_html = "".join(
        f'<span class="badge" style="background:{bg}">{lbl}: <b>{val}</b></span> '
        for lbl, val, bg in (badges or [])
    )
    row_html = ""
    if rows:
        row_html = "<table>" + "".join(
            f"<tr><td class='key'>{k}</td><td class='val'>{v}</td></tr>"
            for k, v in rows
        ) + "</table>"
    return f"""
    <div class="card">
      <div class="card-header" style="background:{color}">{title}</div>
      <div class="card-body">
        <img src="{b64}" />
        <div class="meta">{badge_html}{row_html}</div>
      </div>
    </div>"""

# ══════════════════════════════════════════════════════════════
# Step 0: ROI 감지 (1회 고정)
# ══════════════════════════════════════════════════════════════
print("=" * 55)
print(f"  Pipeline HTML — {IMG_PATH.name}")
print(f"  지시: {INSTRUCTION}")
print("=" * 55)

t_total = time.time()
image_np = np.array(Image.open(IMG_PATH).convert("RGB"))
H, W = image_np.shape[:2]

print("\n[Step 0] GPT-4o — task workspace ROI 감지 (1회)...")
t0 = time.time()
# SAM2 로딩 전에 GPT-4o로 ROI만 먼저 감지
localizer = ObjectLocalizer(table_region=(0,0,W,H))  # SAM2 로딩 포함
roi = localizer.detect_table_region(image_np)
localizer._table_region = roi   # ROI 고정
t_roi = time.time() - t0
print(f"  ROI: {roi}  ({t_roi:.1f}s) → 이후 고정 사용")

img_roi_vis = draw_roi(image_np, roi)

# Step 1: SAM2
print(f"\n[Step 1] SAM2 — ROI 내 segmentation {roi}...")
t1 = time.time()
masks = localizer.segment_all(image_np, table_region=roi)
crops = [localizer.make_crop(image_np, m) for m in masks]
t1 = time.time() - t1
print(f"  {len(masks)}개 mask  ({t1:.1f}s)")
img_step1 = make_sam2_overlay(image_np, masks, roi=roi)

# Step 2: top-5 filter + GPT-4o 식별
print(f"\n[Step 2] Top-5 filter + GPT-4o 타겟 식별...")
t2 = time.time()
filtered = localizer.filter_top_crops(masks, crops, top_n=5)
filtered_crops = [c for _, _, c, _ in filtered]
pos = localizer.identify_target(filtered_crops, INSTRUCTION)
pos = min(pos, len(filtered) - 1)
orig_idx   = filtered[pos][0]
target_mask = masks[orig_idx]
target_crop = crops[orig_idx]
t2 = time.time() - t2
print(f"  완료  ({t2:.1f}s)")
img_step2 = make_crop_grid(filtered, pos)

# Step 3: 타겟 overlay
img_step3 = make_target_overlay(image_np, target_mask, roi=roi)

# Step 4: graspability
print(f"\n[Step 4] GPT-4o — graspability...")
t4 = time.time()
grasp = localizer.reason(target_crop, INSTRUCTION, "graspability")
t4 = time.time() - t4
print(f"  {grasp}  ({t4:.1f}s)")

# Step 5: calibration
print(f"\n[Step 5] GPT-4o — calibration...")
t5 = time.time()
calib = localizer.reason(target_crop, INSTRUCTION, "calibration")
t5 = time.time() - t5
print(f"  {calib}  ({t5:.1f}s)")

# Final overlay
center_norm = calib.get("center_norm")
img_final   = make_target_overlay(image_np, target_mask, center_norm=center_norm, roi=roi)

t_total = time.time() - t_total

# ── 액션 결정 ─────────────────────────────────────────────────
graspable    = grasp.get("graspable", False)
grasp_conf   = grasp.get("confidence", 0)
grasp_align  = grasp.get("alignment", "")
grasp_reason = grasp.get("reason", "")
calib_conf   = calib.get("confidence", 0)
orientation  = calib.get("orientation", "")
calib_notes  = calib.get("notes", "")

if center_norm:
    px_x = int(center_norm[0] * W)
    px_y = int(center_norm[1] * H)
else:
    x, y, bw, bh = [int(v) for v in target_mask["bbox"]]
    px_x, px_y = x + bw // 2, y + bh // 2

action_go    = graspable and grasp_conf >= 0.7
action_color = "#1a7a1a" if action_go else "#a01010"
action_label = "GO — GRASP" if action_go else "HOLD — NOT GRASPABLE"

# ── HTML ──────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Pipeline — {IMG_PATH.name}</title>
<style>
  body  {{ font-family:'Segoe UI',sans-serif; background:#0f0f0f; color:#e0e0e0; margin:0; padding:20px; }}
  h1    {{ text-align:center; color:#f0f0f0; letter-spacing:2px; margin-bottom:4px; }}
  .sub  {{ text-align:center; color:#888; margin-bottom:30px; font-size:14px; }}
  .pipe {{ display:flex; flex-wrap:wrap; gap:20px; justify-content:center; }}
  .card {{ background:#1c1c1c; border-radius:10px; overflow:hidden;
           width:500px; box-shadow:0 4px 16px rgba(0,0,0,0.5); }}
  .card-header {{ padding:10px 16px; font-weight:bold; font-size:15px; color:#fff; }}
  .card-body {{ padding:14px; }}
  .card-body img {{ width:100%; border-radius:6px; display:block; }}
  .meta {{ margin-top:10px; }}
  .badge {{ display:inline-block; padding:3px 10px; border-radius:12px;
            font-size:13px; color:#fff; margin:3px 3px 3px 0; }}
  table {{ width:100%; border-collapse:collapse; margin-top:8px; font-size:13px; }}
  td    {{ padding:5px 8px; border-bottom:1px solid #2a2a2a; vertical-align:top; }}
  td.key {{ color:#aaa; width:36%; font-weight:bold; }}
  td.val {{ color:#e0e0e0; word-break:break-word; }}
  .action {{ margin:30px auto; max-width:720px; border-radius:12px;
             padding:24px 32px; background:{action_color};
             text-align:center; box-shadow:0 6px 24px rgba(0,0,0,0.6); }}
  .action-label {{ font-size:34px; font-weight:900; letter-spacing:3px; color:#fff; }}
  .action-detail {{ margin-top:12px; font-size:14px; color:rgba(255,255,255,0.85); line-height:1.9; }}
  .timing {{ text-align:center; margin-top:18px; color:#666; font-size:13px; }}
  .step-arrow {{ text-align:center; font-size:28px; color:#555; margin:4px 0; width:100%; }}
</style>
</head>
<body>
<h1>🤖 Pipeline Visualization</h1>
<div class="sub">
  <b>{IMG_PATH.name}</b> ({W}×{H}) &nbsp;|&nbsp;
  지시: <b>"{INSTRUCTION}"</b> &nbsp;|&nbsp;
  Total: <b>{t_total:.1f}s</b>
</div>

<div class="pipe">

{html_card(
    "Step 0 · GPT-4o — Task Workspace ROI 감지 (1회 고정)",
    img_roi_vis,
    badges=[
        ("ROI", str(roi), "#7d6608"),
        ("Time", f"{t_roi:.1f}s", "#555"),
    ],
    rows=[
        ("ROI (pixel)", f"({roi[0]}, {roi[1]}) → ({roi[2]}, {roi[3]})"),
        ("ROI 크기", f"{roi[2]-roi[0]}×{roi[3]-roi[1]} px"),
        ("전략", "최초 1회만 감지, 이후 고정 재사용"),
        ("Model", "gpt-4o"),
    ],
    color="#7d6608"
)}

<div class="step-arrow">↓</div>

{html_card(
    f"Step 1 · SAM2 — ROI 내 Blind Segmentation ({len(masks)}개)",
    img_step1,
    badges=[
        ("Masks", len(masks), "#1a5276"),
        ("Time", f"{t1:.1f}s", "#555"),
    ],
    rows=[
        ("입력", f"ROI crop {roi[2]-roi[0]}×{roi[3]-roi[1]}"),
        ("mask 수", f"{len(masks)}개 (전체 이미지 대비 노이즈 감소)"),
        ("Config", "points_per_side=16, iou=0.80"),
        ("Note", "Instruction 없음 — blind segmentation"),
    ],
    color="#1a5276"
)}

<div class="step-arrow">↓</div>

{html_card(
    "Step 2 · Top-5 Filter + GPT-4o 타겟 식별",
    img_step2,
    badges=[
        ("Selected", f"pos[{pos}] → mask[{orig_idx}]", "#1a7a1a"),
        ("Time", f"{t2:.1f}s", "#555"),
    ],
    rows=[
        ("Filter", "area_score + yellow_HSV×3.0 상위 5개"),
        ("Instruction", f'"{INSTRUCTION}"'),
        ("Model", "gpt-4o (vision)"),
    ],
    color="#145a32"
)}

<div class="step-arrow">↓</div>

{html_card(
    "Step 3 · 타겟 Mask Overlay",
    img_step3,
    badges=[
        ("Target mask", f"[{orig_idx}]", "#1a7a1a"),
    ],
    rows=[
        ("Bbox", str(tuple(int(v) for v in target_mask["bbox"]))),
        ("Area", f"{target_mask['area']} px²"),
        ("ROI", str(roi)),
    ],
    color="#145a32"
)}

<div class="step-arrow">↓</div>

{html_card(
    "Step 4 · GPT-4o — Graspability Reasoning",
    target_crop,
    badges=[
        ("Graspable", str(graspable), "#1a7a1a" if graspable else "#a01010"),
        ("Conf", grasp_conf, "#7d6608"),
        ("Time", f"{t4:.1f}s", "#555"),
    ],
    rows=[
        ("Alignment", grasp_align),
        ("Reason", grasp_reason),
        ("Model", "gpt-4o (vision)"),
    ],
    color="#6e2f1a"
)}

<div class="step-arrow">↓</div>

{html_card(
    "Step 5 · GPT-4o — Calibration Reasoning",
    target_crop,
    badges=[
        ("center_norm", str(center_norm), "#5b2c6f"),
        ("Conf", calib_conf, "#7d6608"),
        ("Time", f"{t5:.1f}s", "#555"),
    ],
    rows=[
        ("Pixel center", f"({px_x}, {px_y})"),
        ("Orientation", orientation),
        ("Notes", calib_notes),
        ("Model", "gpt-4o (vision)"),
    ],
    color="#5b2c6f"
)}

<div class="step-arrow">↓</div>

{html_card(
    "Final · Action Output",
    img_final,
    badges=[
        ("Action", action_label, action_color),
    ],
    rows=[
        ("Pixel target (u,v)", f"({px_x}, {px_y})"),
        ("center_norm", str(center_norm)),
        ("Graspable", f"{graspable} (conf={grasp_conf})"),
        ("Alignment", grasp_align),
        ("Orientation", orientation),
    ],
    color=action_color
)}

</div>

<div class="action">
  <div class="action-label">{"✅" if action_go else "🛑"} {action_label}</div>
  <div class="action-detail">
    Pixel target &nbsp;→&nbsp; <b>u={px_x}, v={px_y}</b><br>
    center_norm &nbsp;→&nbsp; <b>{center_norm}</b><br>
    Graspable: <b>{graspable}</b> (conf={grasp_conf}) &nbsp;|&nbsp; Alignment: <b>{grasp_align}</b><br>
    {grasp_reason}
  </div>
</div>

<div class="timing">
  ROI 감지: {t_roi:.1f}s (1회) &nbsp;|&nbsp;
  SAM2: {t1:.1f}s &nbsp;|&nbsp;
  식별: {t2:.1f}s &nbsp;|&nbsp;
  Graspability: {t4:.1f}s &nbsp;|&nbsp;
  Calibration: {t5:.1f}s &nbsp;|&nbsp;
  <b>Total: {t_total:.1f}s</b>
</div>

</body>
</html>"""

out_path = OUT_DIR / f"{IMG_PATH.stem}_pipeline.html"
out_path.write_text(html, encoding="utf-8")
print(f"\n저장: {out_path}")
print(f"전체 소요: {t_total:.1f}s")
