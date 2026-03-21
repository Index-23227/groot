"""
base12.jpg 기준 파이프라인 단계별 HTML 시각화

Step 0: 원본 이미지 + ROI
Step 1: Depth/RGB clustering → 전체 객체 마스크
Step 2: VLM에 전달하는 crops
Step 3: VLM 타겟 식별 결과
Step 4: 최종 EE 좌표
"""
import sys, io, base64, time, random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).parent.parent))
from utils.pipeline_a import DepthClusterPipeline, _to_b64

ROOT     = Path(__file__).parent.parent
IMG_PATH = ROOT / "data" / "base_images" / "base12.jpg"
OUT_DIR  = ROOT / "results" / "vis_base12"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INSTRUCTION = "노란 원기둥을 집어라"
ROI = (192, 180, 1088, 540)   # 고정 ROI

try:
    FONT_SM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    FONT_MD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
except:
    FONT_SM = FONT_MD = ImageFont.load_default()

# ── 유틸 ─────────────────────────────────────────────────────
def p2b(img, max_side=900):
    img = img.copy(); img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def card(title, img_pil, badges=None, rows=None, color="#1e2a3a", note=None):
    b64 = p2b(img_pil)
    bh = "".join(
        f'<span class="badge" style="background:{bg}">{l}: <b>{v}</b></span> '
        for l, v, bg in (badges or []))
    rh = ("<table>" + "".join(
        f"<tr><td class='k'>{k}</td><td class='v'>{v}</td></tr>"
        for k, v in (rows or [])) + "</table>") if rows else ""
    nh = f'<div class="note">{note}</div>' if note else ""
    return f"""<div class="card">
      <div class="ch" style="background:{color}">{title}</div>
      <div class="cb"><img src="{b64}"/><div class="meta">{bh}{rh}{nh}</div></div>
    </div>"""

# ── 시각화 함수들 ─────────────────────────────────────────────
def draw_roi(rgb, roi):
    img = Image.fromarray(rgb).copy()
    d = ImageDraw.Draw(img)
    x1,y1,x2,y2 = roi
    # 바깥 영역 어둡게
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([0,0,img.width,img.height], fill=(0,0,0,120))
    od.rectangle([x1,y1,x2,y2], fill=(0,0,0,0))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(img)
    d.rectangle([x1,y1,x2,y2], outline=(0,200,255), width=4)
    d.text((x1+8, y1+8), f"ROI: {x2-x1}×{y2-y1}px", fill=(0,200,255), font=FONT_MD)
    return img

def draw_all_masks(rgb, objects, roi=None):
    random.seed(42)
    img = Image.fromarray(rgb).convert("RGBA")
    H, W = rgb.shape[:2]
    for i, obj in enumerate(objects):
        col = (random.randint(80,230), random.randint(80,230), random.randint(80,230))
        lyr = np.zeros((H,W,4), dtype=np.uint8)
        lyr[obj["mask"]] = [*col, 130]
        img = Image.alpha_composite(img, Image.fromarray(lyr))
        x,y,bw,bh = obj["bbox"]
        cx,cy = obj["centroid_px"]
        d = ImageDraw.Draw(img)
        d.rectangle([x,y,x+bw,y+bh], outline=(*col,255), width=2)
        d.ellipse([cx-5,cy-5,cx+5,cy+5], fill=(*col,255))
        d.text((x+3,y+3), str(i), fill=(255,255,255,255), font=FONT_SM)
    if roi:
        d = ImageDraw.Draw(img)
        d.rectangle([roi[0],roi[1],roi[2],roi[3]], outline=(0,200,255,200), width=3)
    return img.convert("RGB")

def draw_crop_grid(objects, crops):
    THUMB = 140
    n = len(crops)
    cols = min(n, 6)
    rows_n = (n + cols - 1) // cols
    grid = Image.new("RGB", (cols*(THUMB+8), rows_n*(THUMB+46)), (20,20,20))
    random.seed(42)
    for i, (obj, crop) in enumerate(zip(objects, crops)):
        col_i = i % cols
        row_i = i // cols
        t = crop.copy(); t.thumbnail((THUMB,THUMB))
        xo = col_i*(THUMB+8) + (THUMB-t.width)//2
        yo = row_i*(THUMB+46) + 4
        grid.paste(t, (xo, yo))
        d = ImageDraw.Draw(grid)
        d.rectangle([col_i*(THUMB+8), row_i*(THUMB+46),
                     col_i*(THUMB+8)+THUMB+6, row_i*(THUMB+46)+THUMB+8],
                    outline=(80,80,80), width=1)
        label = f"[{i}] {obj['area']}px²"
        d.text((col_i*(THUMB+8)+4, row_i*(THUMB+46)+THUMB+12),
               label, fill=(160,160,160), font=FONT_SM)
    return grid

def draw_target_result(rgb, objects, crops, target_idx, id_res, roi=None):
    random.seed(42)
    img = Image.fromarray(rgb).convert("RGBA")
    H, W = rgb.shape[:2]
    # 비타겟 희미하게
    for i, obj in enumerate(objects):
        col = (random.randint(80,230), random.randint(80,230), random.randint(80,230))
        if i == target_idx:
            continue
        lyr = np.zeros((H,W,4), dtype=np.uint8)
        lyr[obj["mask"]] = [*col, 50]
        img = Image.alpha_composite(img, Image.fromarray(lyr))
    # 타겟 강조
    t = objects[target_idx]
    lyr = np.zeros((H,W,4), dtype=np.uint8)
    lyr[t["mask"]] = [0,220,80,180]
    img = Image.alpha_composite(img, Image.fromarray(lyr))
    d = ImageDraw.Draw(img)
    x,y,bw,bh = t["bbox"]
    d.rectangle([x,y,x+bw,y+bh], outline=(0,255,80,255), width=5)
    # 라벨 박스
    conf = id_res.get("confidence","")
    d.rectangle([x, y-32, x+bw, y], fill=(0,160,50,220))
    d.text((x+4, y-28), f"TARGET [{target_idx}]  conf={conf}", fill=(255,255,255,255), font=FONT_MD)
    if roi:
        d.rectangle([roi[0],roi[1],roi[2],roi[3]], outline=(0,200,255,200), width=3)
    return img.convert("RGB")

def draw_ee_output(rgb, target_obj, roi=None):
    img = Image.fromarray(rgb).convert("RGBA")
    H, W = rgb.shape[:2]
    lyr = np.zeros((H,W,4), dtype=np.uint8)
    lyr[target_obj["mask"]] = [0,220,80,160]
    img = Image.alpha_composite(img, Image.fromarray(lyr))
    d = ImageDraw.Draw(img)
    x,y,bw,bh = target_obj["bbox"]
    cx,cy = target_obj["centroid_px"]
    d.rectangle([x,y,x+bw,y+bh], outline=(0,255,80,255), width=4)
    # EE 크로스헤어
    r = 16
    d.ellipse([cx-r,cy-r,cx+r,cy+r], outline=(255,60,0,255), width=4)
    d.line([(cx-28,cy),(cx+28,cy)], fill=(255,60,0,255), width=3)
    d.line([(cx,cy-28),(cx,cy+28)], fill=(255,60,0,255), width=3)
    # 정보 박스
    depth_str = f"{target_obj['depth_mm']:.0f}mm" if target_obj['depth_mm'] else "N/A (RealSense 필요)"
    lines = [
        f"EE target",
        f"pixel  u={cx}, v={cy}",
        f"norm   ({cx/W:.3f}, {cy/H:.3f})",
        f"depth  {depth_str}",
    ]
    bx, by = min(cx+20, W-220), max(cy-70, 10)
    d.rectangle([bx-4, by-4, bx+210, by+len(lines)*20+4], fill=(20,20,20,200))
    for j, ln in enumerate(lines):
        col = (255,60,0,255) if j==0 else (220,220,220,255)
        d.text((bx, by+j*20), ln, fill=col, font=FONT_SM if j>0 else FONT_MD)
    if roi:
        d.rectangle([roi[0],roi[1],roi[2],roi[3]], outline=(0,200,255,150), width=3)
    return img.convert("RGB")

# ══════════════════════════════════════════════════════════════
print("="*55)
print(f"  Pipeline 시각화 — {IMG_PATH.name}")
print(f"  지시: {INSTRUCTION}")
print("="*55)

t_total = time.time()
rgb_np = np.array(Image.open(IMG_PATH).convert("RGB"))
H, W = rgb_np.shape[:2]

# RealSense depth 시도
depth_np = None
try:
    import pyrealsense2 as rs
    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    for _ in range(20): pipe.wait_for_frames()
    frames   = align.process(pipe.wait_for_frames())
    rgb_np   = np.asanyarray(frames.get_color_frame().get_data())[:,:,::-1]
    depth_np = np.asanyarray(frames.get_depth_frame().get_data())
    pipe.stop()
    print("RealSense depth 사용")
except:
    print("RGB fallback 사용 (RealSense 없음)")

mode = "Depth" if depth_np is not None else "RGB fallback"

# ── 각 단계 실행 ──────────────────────────────────────────────
pipeline = DepthClusterPipeline()

# Step 0: 원본 + ROI
img_s0 = draw_roi(rgb_np, ROI)

# Step 1: 객체 분리
print(f"\n[Step 1] 객체 분리 ({mode})...")
t1 = time.time()
objects = pipeline.segment(rgb_np, depth=depth_np, roi=ROI)
t1 = time.time() - t1
print(f"  {len(objects)}개  ({t1*1000:.0f}ms)")
img_s1 = draw_all_masks(rgb_np, objects, roi=ROI)

# Step 2: crops 준비
crops = [pipeline.make_crop(rgb_np, o) for o in objects]
img_s2 = draw_crop_grid(objects, crops)

# Step 3: VLM 식별
print(f"\n[Step 2] VLM 타겟 식별 ({len(objects)}개 crops)...")
t2 = time.time()
target_idx, id_res = pipeline.identify(crops, INSTRUCTION)
t2 = time.time() - t2
print(f"  [{target_idx}번]  conf={id_res.get('confidence')}  ({t2:.1f}s)")
img_s3 = draw_target_result(rgb_np, objects, crops, target_idx, id_res, roi=ROI)

# Step 4: EE 좌표
target = objects[target_idx]
cx, cy = target["centroid_px"]
depth_str = f"{target['depth_mm']:.0f}mm" if target['depth_mm'] else "N/A"
img_s4 = draw_ee_output(rgb_np, target, roi=ROI)

t_total = time.time() - t_total

# ── HTML 생성 ─────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Pipeline Visualization — {IMG_PATH.name}</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0 }}
  body {{ font-family:'Segoe UI',sans-serif; background:#0c0c0c; color:#e0e0e0; padding:24px }}
  h1   {{ text-align:center; color:#fff; font-size:22px; letter-spacing:2px; margin-bottom:6px }}
  .sub {{ text-align:center; color:#666; font-size:13px; margin-bottom:28px }}
  .pipeline {{ display:flex; flex-direction:column; align-items:center; gap:0 }}
  .card {{ background:#161616; border-radius:10px; overflow:hidden;
           width:700px; box-shadow:0 4px 20px rgba(0,0,0,0.6); }}
  .ch   {{ padding:10px 16px; font-weight:bold; font-size:14px; color:#fff; letter-spacing:.5px }}
  .cb   {{ padding:14px }}
  .cb img {{ width:100%; border-radius:6px; display:block }}
  .meta {{ margin-top:10px }}
  .badge {{ display:inline-block; padding:3px 10px; border-radius:10px;
            font-size:12px; color:#fff; margin:2px }}
  table {{ width:100%; border-collapse:collapse; margin-top:8px; font-size:12px }}
  td    {{ padding:5px 8px; border-bottom:1px solid #222; vertical-align:top }}
  td.k  {{ color:#888; width:34%; font-weight:bold }}
  td.v  {{ color:#ddd }}
  .note {{ margin-top:8px; padding:8px 10px; background:#1e1e1e;
           border-left:3px solid #444; font-size:12px; color:#aaa; border-radius:3px }}
  .arrow {{ text-align:center; padding:10px 0; font-size:28px; color:#333 }}
  .final {{ width:700px; background:#0f2a0f; border-radius:12px;
            padding:20px 28px; text-align:center; margin-top:10px;
            box-shadow:0 6px 24px rgba(0,0,0,0.6) }}
  .final-label {{ font-size:26px; font-weight:900; color:#2ecc71; letter-spacing:2px }}
  .final-detail {{ margin-top:12px; font-size:13px; color:#aaa; line-height:2 }}
  .final-detail b {{ color:#e0e0e0 }}
  .timing {{ text-align:center; margin-top:16px; color:#444; font-size:12px }}
  .tag  {{ display:inline-block; padding:1px 8px; border-radius:4px;
           font-size:11px; font-weight:bold; margin-left:6px; vertical-align:middle }}
</style>
</head>
<body>
<h1>Pipeline Visualization</h1>
<div class="sub">
  <b>{IMG_PATH.name}</b> ({W}×{H}) &nbsp;|&nbsp;
  지시: <b>"{INSTRUCTION}"</b> &nbsp;|&nbsp;
  Mode: <b>{mode}</b> &nbsp;|&nbsp;
  Total: <b>{t_total:.1f}s</b>
</div>

<div class="pipeline">

{card("Step 0 · 원본 이미지 + 작업 ROI", img_s0,
    badges=[("ROI", f"({ROI[0]},{ROI[1]}) → ({ROI[2]},{ROI[3]})", "#2c5f8a"),
            ("크기", f"{ROI[2]-ROI[0]}×{ROI[3]-ROI[1]}px", "#444")],
    rows=[("이미지", f"{IMG_PATH.name}  ({W}×{H})"),
          ("ROI 목적", "테이블 작업 영역만 clustering — 배경/로봇팔 제외"),
          ("ROI 설정", "GPT-4o 1회 감지 후 고정 재사용")],
    color="#1a3a5c")}

<div class="arrow">↓</div>

{card(f"Step 1 · {'Depth' if depth_np is not None else 'RGB'} Clustering — {len(objects)}개 객체",
    img_s1,
    badges=[("객체 수", len(objects), "#1a5276"),
            ("시간", f"{t1*1000:.0f}ms", "#2e4057"),
            ("mode", mode, "#555")],
    rows=[("방식", "depth threshold → connected components" if depth_np is not None
                   else "RGB: 흰 테이블 제거 + 엣지 → connected components"),
          ("SAM2", "❌ 사용 안 함"),
          ("파라미터", "min_h=20mm, max_h=350mm, min_area=400px²")],
    note="각 컬러 마스크 = 독립 객체 / 흰 점 = centroid" if depth_np is None
         else "depth 기반 정확한 3D centroid 포함",
    color="#1a3a5c")}

<div class="arrow">↓</div>

{card(f"Step 2 · VLM 입력 Crops ({len(crops)}개)", img_s2,
    badges=[("crops", len(crops), "#5b4a00"),
            ("배경 제거", "✓", "#2e4057")],
    rows=[("전처리", "mask 영역만 남기고 배경 흰색 처리"),
          ("전달 방식", "base64 JPEG → GPT-4o vision"),
          ("목적", "VLM이 instruction과 매칭되는 객체 선택")],
    note="객체를 분리해서 보여주면 VLM 식별 정확도↑ (배경 노이즈 제거)",
    color="#3d3000")}

<div class="arrow">↓</div>

{card(f"Step 3 · GPT-4o 타겟 식별 → [{target_idx}번] 선택", img_s3,
    badges=[("Target", f"[{target_idx}번]", "#1a7a1a"),
            ("Conf", id_res.get('confidence',''), "#7d6608"),
            ("시간", f"{t2:.1f}s", "#444")],
    rows=[("모델", "gpt-4o (vision)"),
          ("지시", f'"{INSTRUCTION}"'),
          ("이유", id_res.get('reason','')[:80]),
          ("VLM 역할", "오직 '어떤 물체?' 식별만 — action은 생성 안 함")],
    color="#145a32")}

<div class="arrow">↓</div>

{card("Step 4 · EE 좌표 추출", img_s4,
    badges=[("pixel (u,v)", f"({cx}, {cy})", "#6a0dad"),
            ("depth", depth_str, "#5b2c6f"),
            ("norm", f"({cx/W:.3f}, {cy/H:.3f})", "#444")],
    rows=[("pixel (u,v)", f"({cx}, {cy})"),
          ("normalized", f"({cx/W:.4f}, {cy/H:.4f})"),
          ("depth", depth_str),
          ("다음 단계", "캘리브레이션 행렬 → 로봇 좌표 (X,Y,Z)")],
    note="depth 없을 시 픽셀 좌표만 출력. RealSense 연결 시 depth_mm 추가됨.",
    color="#3d1a5c")}

</div>

<div class="final">
  <div class="final-label">✅ EE Target</div>
  <div class="final-detail">
    pixel &nbsp;→&nbsp; <b>u = {cx},  v = {cy}</b><br>
    normalized &nbsp;→&nbsp; <b>({cx/W:.4f}, {cy/H:.4f})</b><br>
    depth &nbsp;→&nbsp; <b>{depth_str}</b><br>
    <span style="color:#666;font-size:12px">
      → 캘리브레이션 완료 시: (X, Y, Z)_robot 으로 변환 후 로봇 이동
    </span>
  </div>
</div>

<div class="timing">
  Step1 clustering: {t1*1000:.0f}ms &nbsp;|&nbsp;
  Step2 VLM: {t2:.1f}s &nbsp;|&nbsp;
  <b>Total: {t_total:.1f}s</b>
</div>

</body>
</html>"""

out = OUT_DIR / f"{IMG_PATH.stem}_pipeline.html"
out.write_text(html, encoding="utf-8")
print(f"\n저장: {out}")
print(f"총 소요: {t_total:.1f}s")
