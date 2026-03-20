"""
SAM2 없는 RGBD 파이프라인 HTML 시각화

Depth 있을 시: RealSense depth threshold → 객체 분리
Depth 없을 시: RGB fallback (엣지 + contour)

base12.jpg로 RGB fallback 모드 테스트.
"""
import sys, io, base64, json, time, random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).parent.parent))
from utils.rgbd_localizer import RGBDLocalizer, _pil_to_b64

ROOT     = Path(__file__).parent.parent
IMG_PATH = ROOT / "data" / "base_images" / "base12.jpg"
OUT_DIR  = ROOT / "results" / "rgbd_pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INSTRUCTION = "노란 원기둥을 집어라"
ROI = (192, 180, 1088, 540)   # 이전에 GPT-4o로 감지한 workspace ROI (고정)

try:
    FONT_SM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    FONT_MD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
except:
    FONT_SM = FONT_MD = ImageFont.load_default()

# ── 시각화 유틸 ───────────────────────────────────────────────
def pil_to_b64_html(img: Image.Image, max_side: int = 900) -> str:
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def draw_detections(rgb_np, objects, target_idx=None, roi=None):
    """감지된 객체 bbox + centroid 시각화"""
    img = Image.fromarray(rgb_np).convert("RGBA")
    H, W = rgb_np.shape[:2]
    random.seed(7)
    for i, obj in enumerate(objects):
        is_target = (i == target_idx)
        color = (0, 220, 80) if is_target else (
            random.randint(100,220), random.randint(100,220), random.randint(100,220)
        )
        # mask overlay
        lyr = np.zeros((H, W, 4), dtype=np.uint8)
        lyr[obj["mask"]] = [*color, 120]
        img = Image.alpha_composite(img, Image.fromarray(lyr))

        x, y, bw, bh = obj["bbox"]
        cx, cy = obj["centroid_px"]
        d = ImageDraw.Draw(img)
        width = 4 if is_target else 2
        d.rectangle([x, y, x+bw, y+bh], outline=(*color, 255), width=width)
        # centroid 십자
        r = 8 if is_target else 5
        d.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(*color, 255), width=3)
        d.line([(cx-14, cy), (cx+14, cy)], fill=(*color, 255), width=2)
        d.line([(cx, cy-14), (cx, cy+14)], fill=(*color, 255), width=2)
        label = f"[{i}]" + (" TARGET" if is_target else "")
        d.text((x+3, y+3), label, fill=(255,255,255,255), font=FONT_SM)

    # ROI 박스
    if roi:
        d = ImageDraw.Draw(img)
        x1, y1, x2, y2 = roi
        d.rectangle([x1, y1, x2, y2], outline=(0, 200, 255, 200), width=3)

    return img.convert("RGB")

def draw_target_final(rgb_np, target_obj, center_norm=None, roi=None):
    img = Image.fromarray(rgb_np).convert("RGBA")
    H, W = rgb_np.shape[:2]
    lyr = np.zeros((H, W, 4), dtype=np.uint8)
    lyr[target_obj["mask"]] = [0, 220, 80, 160]
    img = Image.alpha_composite(img, Image.fromarray(lyr))
    x, y, bw, bh = target_obj["bbox"]
    d = ImageDraw.Draw(img)
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
    return img.convert("RGB")

def make_crop_grid(objects, crops, target_idx):
    THUMB = 150
    n = len(crops)
    grid = Image.new("RGB", (n * (THUMB + 8), THUMB + 50), (30, 30, 30))
    random.seed(7)
    for i, (obj, crop) in enumerate(zip(objects, crops)):
        thumb = crop.copy(); thumb.thumbnail((THUMB, THUMB))
        xo = i * (THUMB + 8) + (THUMB - thumb.width) // 2
        grid.paste(thumb, (xo, 4))
        d = ImageDraw.Draw(grid)
        is_t = (i == target_idx)
        col = (0, 220, 80) if is_t else (140, 140, 140)
        d.rectangle([i*(THUMB+8), 0, i*(THUMB+8)+THUMB+6, THUMB+8],
                    outline=col, width=4 if is_t else 1)
        label = f"[{i}] {obj['area']}px" + (" ✓" if is_t else "")
        d.text((i*(THUMB+8)+4, THUMB+14), label, fill=col, font=FONT_SM)
        if obj["depth_mm"]:
            d.text((i*(THUMB+8)+4, THUMB+30), f"{obj['depth_mm']:.0f}mm", fill=(180,180,180), font=FONT_SM)
    return grid

def html_card(title, img_pil, badges=None, rows=None, color="#2c3e50"):
    b64 = pil_to_b64_html(img_pil)
    badges_html = "".join(
        f'<span class="badge" style="background:{bg}">{lbl}: <b>{val}</b></span> '
        for lbl, val, bg in (badges or [])
    )
    rows_html = ""
    if rows:
        rows_html = "<table>" + "".join(
            f"<tr><td class='key'>{k}</td><td class='val'>{v}</td></tr>"
            for k, v in rows
        ) + "</table>"
    return f"""
    <div class="card">
      <div class="card-header" style="background:{color}">{title}</div>
      <div class="card-body">
        <img src="{b64}"/>
        <div class="meta">{badges_html}{rows_html}</div>
      </div>
    </div>"""

# ══════════════════════════════════════════════════════════════
# 파이프라인 실행
# ══════════════════════════════════════════════════════════════
print("=" * 55)
print(f"  RGBD Pipeline (SAM2-free) — {IMG_PATH.name}")
print(f"  지시: {INSTRUCTION}")
print(f"  ROI: {ROI}")
print("=" * 55)

t_total = time.time()
rgb_np  = np.array(Image.open(IMG_PATH).convert("RGB"))
H, W    = rgb_np.shape[:2]

# RealSense 사용 가능 시 depth 캡처 (없으면 None → RGB fallback)
depth_np = None
try:
    import pyrealsense2 as rs
    print("\n[RealSense] depth 캡처 중...")
    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16,  30)
    profile = pipe.start(cfg)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()
    align = rs.align(rs.stream.color)
    for _ in range(15): pipe.wait_for_frames()
    frames   = align.process(pipe.wait_for_frames())
    rgb_np   = np.asanyarray(frames.get_color_frame().get_data())[:,:,::-1]
    depth_np = np.asanyarray(frames.get_depth_frame().get_data())  # mm (uint16)
    pipe.stop()
    print(f"  depth 캡처 완료 (scale={depth_scale})")
except Exception as e:
    print(f"\n[RealSense 없음 → RGB fallback] ({e})")
    depth_np = None

localizer = RGBDLocalizer()

# Step 1: 객체 분리
print(f"\n[Step 1] 객체 분리 ({'depth' if depth_np is not None else 'RGB fallback'})...")
t1 = time.time()
objects = localizer.segment_objects(rgb_np, depth=depth_np, roi=ROI)
crops   = [localizer.make_crop(rgb_np, o) for o in objects]
t1 = time.time() - t1
method  = objects[0]["method"] if objects else "none"
print(f"  {len(objects)}개 객체  ({t1:.2f}s)  method={method}")

img_step1 = draw_detections(rgb_np, objects, roi=ROI)

# Step 2: GPT-4o 타겟 식별
print(f"\n[Step 2] GPT-4o 타겟 식별 ({len(objects)}개)...")
t2 = time.time()
idx = localizer.identify_target(crops, INSTRUCTION)
idx = min(idx, len(objects) - 1)
target_obj  = objects[idx]
target_crop = crops[idx]
t2 = time.time() - t2
print(f"  완료  ({t2:.1f}s)")

img_step2 = make_crop_grid(objects, crops, idx)
img_step3 = draw_detections(rgb_np, objects, target_idx=idx, roi=ROI)

# Step 3: Graspability
print(f"\n[Step 3] GPT-4o graspability...")
t3 = time.time()
grasp = localizer.reason(target_crop, INSTRUCTION, "graspability")
t3 = time.time() - t3
print(f"  {grasp}  ({t3:.1f}s)")

# Step 4: Calibration
print(f"\n[Step 4] GPT-4o calibration...")
t4 = time.time()
calib = localizer.reason(target_crop, INSTRUCTION, "calibration")
t4 = time.time() - t4
print(f"  {calib}  ({t4:.1f}s)")

t_total = time.time() - t_total

# 결과 정리
center_norm  = calib.get("center_norm")
graspable    = grasp.get("graspable", False)
grasp_conf   = grasp.get("confidence", 0)
grasp_align  = grasp.get("alignment", "")
grasp_reason = grasp.get("reason", "")
orientation  = calib.get("orientation", "")
calib_notes  = calib.get("notes", "")

cx_px, cy_px = target_obj["centroid_px"]
if center_norm:
    cx_px = int(center_norm[0] * W)
    cy_px = int(center_norm[1] * H)

depth_str = f"{target_obj['depth_mm']:.0f}mm" if target_obj["depth_mm"] else "N/A"
action_go    = graspable and grasp_conf >= 0.7
action_color = "#1a7a1a" if action_go else "#a01010"
action_label = "GO — GRASP" if action_go else "HOLD — NOT GRASPABLE"

img_final = draw_target_final(rgb_np, target_obj, center_norm=center_norm, roi=ROI)

# ── HTML ──────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>RGBD Pipeline (SAM2-free) — {IMG_PATH.name}</title>
<style>
  body  {{ font-family:'Segoe UI',sans-serif; background:#0f0f0f; color:#e0e0e0; margin:0; padding:20px; }}
  h1    {{ text-align:center; color:#f0f0f0; letter-spacing:2px; margin-bottom:4px; }}
  .sub  {{ text-align:center; color:#888; margin-bottom:30px; font-size:14px; }}
  .pipe {{ display:flex; flex-wrap:wrap; gap:20px; justify-content:center; }}
  .card {{ background:#1c1c1c; border-radius:10px; overflow:hidden;
           width:520px; box-shadow:0 4px 16px rgba(0,0,0,0.5); }}
  .card-header {{ padding:10px 16px; font-weight:bold; font-size:15px; color:#fff; }}
  .card-body {{ padding:14px; }}
  .card-body img {{ width:100%; border-radius:6px; display:block; }}
  .meta  {{ margin-top:10px; }}
  .badge {{ display:inline-block; padding:3px 10px; border-radius:12px;
            font-size:13px; color:#fff; margin:3px 3px 3px 0; }}
  table  {{ width:100%; border-collapse:collapse; margin-top:8px; font-size:13px; }}
  td     {{ padding:5px 8px; border-bottom:1px solid #2a2a2a; vertical-align:top; }}
  td.key {{ color:#aaa; width:36%; font-weight:bold; }}
  td.val {{ color:#e0e0e0; word-break:break-word; }}
  .action {{ margin:28px auto; max-width:720px; border-radius:12px; padding:24px 32px;
             background:{action_color}; text-align:center; box-shadow:0 6px 24px rgba(0,0,0,0.6); }}
  .action-label  {{ font-size:32px; font-weight:900; letter-spacing:3px; color:#fff; }}
  .action-detail {{ margin-top:12px; font-size:14px; color:rgba(255,255,255,0.85); line-height:1.9; }}
  .timing {{ text-align:center; margin-top:16px; color:#666; font-size:13px; }}
  .arrow  {{ text-align:center; font-size:26px; color:#444; margin:4px 0; width:100%; }}
  .tag-sam2free {{ display:inline-block; background:#c0392b; color:#fff;
                   padding:2px 10px; border-radius:6px; font-size:12px;
                   font-weight:bold; margin-left:8px; vertical-align:middle; }}
</style>
</head>
<body>
<h1>🤖 RGBD Pipeline <span class="tag-sam2free">SAM2-FREE</span></h1>
<div class="sub">
  <b>{IMG_PATH.name}</b> ({W}×{H}) &nbsp;|&nbsp;
  지시: <b>"{INSTRUCTION}"</b> &nbsp;|&nbsp;
  Mode: <b>{method.upper()}</b> &nbsp;|&nbsp;
  ROI: <b>{ROI}</b> &nbsp;|&nbsp;
  Total: <b>{t_total:.1f}s</b>
</div>

<div class="pipe">

{html_card(
    "Step 1 · Depth/RGB 객체 분리",
    img_step1,
    badges=[
        ("Method", method.upper(), "#1a5276" if method=="depth" else "#7d6608"),
        ("Objects", len(objects), "#1a5276"),
        ("Time", f"{t1:.2f}s", "#2e4057"),
    ],
    rows=[
        ("Method", "RealSense depth threshold" if method=="depth" else "RGB 엣지+contour fallback"),
        ("ROI", str(ROI)),
        ("감지 객체 수", f"{len(objects)}개"),
        ("SAM2", "❌ 사용 안 함"),
    ],
    color="#1a3a5c"
)}

<div class="arrow">↓</div>

{html_card(
    "Step 2 · Crops & GPT-4o 타겟 식별",
    img_step2,
    badges=[
        ("Selected", f"[{idx}]", "#1a7a1a"),
        ("Time", f"{t2:.1f}s", "#555"),
    ],
    rows=[
        ("Instruction", f'"{INSTRUCTION}"'),
        ("Crops 수", len(objects)),
        ("Model", "gpt-4o (vision)"),
    ],
    color="#145a32"
)}

<div class="arrow">↓</div>

{html_card(
    "Step 3 · 타겟 확인",
    img_step3,
    badges=[
        ("Target", f"[{idx}]", "#1a7a1a"),
        ("Area", f"{target_obj['area']}px²", "#555"),
        ("Depth", depth_str, "#5b2c6f"),
    ],
    rows=[
        ("Centroid (px)", f"({cx_px}, {cy_px})"),
        ("Depth", depth_str),
        ("Bbox", str(target_obj["bbox"])),
    ],
    color="#145a32"
)}

<div class="arrow">↓</div>

{html_card(
    "Step 4 · GPT-4o Graspability",
    target_crop,
    badges=[
        ("Graspable", str(graspable), "#1a7a1a" if graspable else "#a01010"),
        ("Conf", grasp_conf, "#7d6608"),
        ("Time", f"{t3:.1f}s", "#555"),
    ],
    rows=[
        ("Alignment", grasp_align),
        ("Reason", grasp_reason),
        ("View", "버드뷰 기준"),
    ],
    color="#6e2f1a"
)}

<div class="arrow">↓</div>

{html_card(
    "Step 5 · GPT-4o Calibration",
    target_crop,
    badges=[
        ("center_norm", str(center_norm), "#5b2c6f"),
        ("Time", f"{t4:.1f}s", "#555"),
    ],
    rows=[
        ("Pixel (u,v)", f"({cx_px}, {cy_px})"),
        ("center_norm", str(center_norm)),
        ("Orientation", orientation),
        ("Notes", calib_notes),
    ],
    color="#5b2c6f"
)}

<div class="arrow">↓</div>

{html_card(
    "Final · Action Output",
    img_final,
    badges=[
        ("Action", action_label, action_color),
    ],
    rows=[
        ("Pixel target (u,v)", f"({cx_px}, {cy_px})"),
        ("Depth", depth_str),
        ("center_norm", str(center_norm)),
        ("Graspable", f"{graspable} (conf={grasp_conf})"),
        ("Alignment", grasp_align),
    ],
    color=action_color
)}

</div>

<div class="action">
  <div class="action-label">{"✅" if action_go else "🛑"} {action_label}</div>
  <div class="action-detail">
    Pixel target &nbsp;→&nbsp; <b>u={cx_px}, v={cy_px}</b><br>
    Depth &nbsp;→&nbsp; <b>{depth_str}</b><br>
    Graspable: <b>{graspable}</b> (conf={grasp_conf}) &nbsp;|&nbsp; Alignment: <b>{grasp_align}</b><br>
    {grasp_reason}
  </div>
</div>

<div class="timing">
  객체분리: {t1:.2f}s &nbsp;|&nbsp;
  GPT-4o 식별: {t2:.1f}s &nbsp;|&nbsp;
  Graspability: {t3:.1f}s &nbsp;|&nbsp;
  Calibration: {t4:.1f}s &nbsp;|&nbsp;
  <b>Total: {t_total:.1f}s</b>
  &nbsp;&nbsp;(SAM2 없음 — 기존 대비 ~12s 절약)
</div>

</body>
</html>"""

out_path = OUT_DIR / f"{IMG_PATH.stem}_rgbd_pipeline.html"
out_path.write_text(html, encoding="utf-8")
print(f"\n저장: {out_path}")
print(f"전체 소요: {t_total:.1f}s  (SAM2 없음)")
