"""
Pipeline A vs B 비교 테스트 + HTML 시각화
base13.jpg 기준 (RealSense depth 있으면 자동 사용)

A: Depth clustering → blind segmentation → VLM 식별 → EE
B: VLM bbox 추출 → bbox 내 depth clustering → EE
"""
import sys, io, base64, time, random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).parent.parent))
from utils.pipeline_a import DepthClusterPipeline
from utils.pipeline_b import BBoxDepthPipeline

ROOT     = Path(__file__).parent.parent
IMG_PATH = ROOT / "data" / "base_images" / "base13.jpg"
OUT_DIR  = ROOT / "results" / "compare_ab"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INSTRUCTION = "초록 원기둥을 집어라"
ROI = (100, 50, 900, 560)   # 테이블 작업 영역 (base13 기준)

try:
    FONT_SM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    FONT_MD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
except:
    FONT_SM = FONT_MD = ImageFont.load_default()

# ── 시각화 헬퍼 ───────────────────────────────────────────────
def p2b(img: Image.Image, max_side=900) -> str:
    img = img.copy(); img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=87)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def draw_all_objects(rgb, objects, target_idx=None, roi=None):
    img = Image.fromarray(rgb).convert("RGBA")
    H, W = rgb.shape[:2]
    random.seed(3)
    for i, obj in enumerate(objects):
        is_t  = (i == target_idx)
        color = (0, 220, 80) if is_t else (
            random.randint(80,220), random.randint(80,220), random.randint(80,220))
        lyr = np.zeros((H, W, 4), dtype=np.uint8)
        lyr[obj["mask"]] = [*color, 130]
        img = Image.alpha_composite(img, Image.fromarray(lyr))
        x, y, bw, bh = obj["bbox"]
        cx, cy = obj["centroid_px"]
        d = ImageDraw.Draw(img)
        d.rectangle([x,y,x+bw,y+bh], outline=(*color,255), width=3 if is_t else 1)
        d.ellipse([cx-7,cy-7,cx+7,cy+7], outline=(*color,255), width=3)
        d.text((x+3, y+3), f"[{i}]"+(" T" if is_t else ""), fill=(255,255,255,255), font=FONT_SM)
    if roi:
        d = ImageDraw.Draw(img)
        d.rectangle([roi[0],roi[1],roi[2],roi[3]], outline=(0,200,255,180), width=3)
    return img.convert("RGB")

def draw_bbox_result(rgb, bbox_px, refine, roi=None):
    img = Image.fromarray(rgb).convert("RGBA")
    H, W = rgb.shape[:2]
    # target mask
    lyr = np.zeros((H, W, 4), dtype=np.uint8)
    lyr[refine["mask"]] = [0, 180, 255, 130]
    img = Image.alpha_composite(img, Image.fromarray(lyr))
    d = ImageDraw.Draw(img)
    bx1, by1, bx2, by2 = bbox_px
    # VLM bbox (노란)
    d.rectangle([bx1,by1,bx2,by2], outline=(255,220,0,255), width=3)
    d.text((bx1+4, by1+4), "VLM bbox", fill=(255,220,0,255), font=FONT_SM)
    # centroid
    cx, cy = refine["centroid_px"]
    r = 12
    d.ellipse([cx-r,cy-r,cx+r,cy+r], outline=(255,60,0,255), width=4)
    d.line([(cx-20,cy),(cx+20,cy)], fill=(255,60,0,255), width=3)
    d.line([(cx,cy-20),(cx,cy+20)], fill=(255,60,0,255), width=3)
    if roi:
        d.rectangle([roi[0],roi[1],roi[2],roi[3]], outline=(0,200,255,180), width=3)
    return img.convert("RGB")

def draw_ee_final(rgb, ee, label, color_hex):
    img = Image.fromarray(rgb).copy()
    d = ImageDraw.Draw(img)
    cx, cy = ee["pixel_uv"]
    r = 16
    col = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (0,2,4))
    d.ellipse([cx-r,cy-r,cx+r,cy+r], outline=col, width=5)
    d.line([(cx-28,cy),(cx+28,cy)], fill=col, width=4)
    d.line([(cx,cy-28),(cx,cy+28)], fill=col, width=4)
    depth_str = f"{ee['depth_mm']:.0f}mm" if ee["depth_mm"] else "N/A"
    txt = f"{label}  u={cx} v={cy}  d={depth_str}"
    d.rectangle([cx-4, cy+r+4, cx+len(txt)*8, cy+r+24], fill=(20,20,20))
    d.text((cx, cy+r+6), txt, fill=col, font=FONT_SM)
    return img

def make_crop_grid(crops, target_idx):
    THUMB = 130
    n = len(crops)
    grid = Image.new("RGB", (n*(THUMB+6), THUMB+36), (25,25,25))
    random.seed(3)
    for i, crop in enumerate(crops):
        t = crop.copy(); t.thumbnail((THUMB,THUMB))
        xo = i*(THUMB+6)+(THUMB-t.width)//2
        grid.paste(t, (xo, 3))
        d = ImageDraw.Draw(grid)
        is_t = (i == target_idx)
        col = (0,220,80) if is_t else (120,120,120)
        d.rectangle([i*(THUMB+6),0,i*(THUMB+6)+THUMB+4,THUMB+4], outline=col, width=3 if is_t else 1)
        d.text((i*(THUMB+6)+3, THUMB+8), f"[{i}]"+(" ✓" if is_t else ""), fill=col, font=FONT_SM)
    return grid

def card(title, img_pil, badges=None, rows=None, color="#2c3e50"):
    b64 = p2b(img_pil)
    bh  = "".join(f'<span class="badge" style="background:{bg}">{l}: <b>{v}</b></span> '
                  for l,v,bg in (badges or []))
    rh  = ("<table>"+"".join(f"<tr><td class='k'>{k}</td><td class='v'>{v}</td></tr>"
                             for k,v in (rows or []))+"</table>") if rows else ""
    return f"""<div class="card">
      <div class="ch" style="background:{color}">{title}</div>
      <div class="cb"><img src="{b64}"/><div class="meta">{bh}{rh}</div></div>
    </div>"""

# ══════════════════════════════════════════════════════════════
print("="*55)
print(f"  A vs B 비교 — {IMG_PATH.name}")
print(f"  지시: {INSTRUCTION}")
print("="*55)

rgb_np = np.array(Image.open(IMG_PATH).convert("RGB"))
H, W   = rgb_np.shape[:2]

# RealSense depth 시도
depth_np = None
try:
    import pyrealsense2 as rs
    print("\n[RealSense] depth 캡처...")
    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    profile = pipe.start(cfg)
    align   = rs.align(rs.stream.color)
    for _ in range(15): pipe.wait_for_frames()
    frames   = align.process(pipe.wait_for_frames())
    rgb_np   = np.asanyarray(frames.get_color_frame().get_data())[:,:,::-1]
    depth_np = np.asanyarray(frames.get_depth_frame().get_data())
    pipe.stop()
    print("  depth 캡처 완료")
except Exception as e:
    print(f"\n[RealSense 없음 → RGB fallback] {e}")

mode = "DEPTH" if depth_np is not None else "RGB fallback"
print(f"\n  Mode: {mode}")
print(f"  ROI: {ROI}\n")

# ══════════════════════════════════════════════════════════════
# Pipeline A 실행
# ══════════════════════════════════════════════════════════════
print("─"*40)
print("▶ Pipeline A: Depth Clustering → VLM")
print("─"*40)
pipe_a = DepthClusterPipeline()
res_a  = pipe_a.run(rgb_np, INSTRUCTION, depth=depth_np, roi=ROI)

img_a1 = draw_all_objects(rgb_np, res_a["objects"], roi=ROI)
img_a2 = make_crop_grid(res_a["crops"], res_a["target_idx"])
img_a3 = draw_all_objects(rgb_np, res_a["objects"], target_idx=res_a["target_idx"], roi=ROI)
img_a_ee = draw_ee_final(rgb_np, res_a["ee"], "A", "#00dc50")

# ══════════════════════════════════════════════════════════════
# Pipeline B 실행
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*40)
print("▶ Pipeline B: VLM BBox → Depth Refine")
print("─"*40)
pipe_b = BBoxDepthPipeline()
res_b  = pipe_b.run(rgb_np, INSTRUCTION, depth=depth_np, roi=ROI)

img_b1 = draw_bbox_result(rgb_np, res_b["bbox_px"], res_b["refine"], roi=ROI)
img_b_ee = draw_ee_final(rgb_np, res_b["ee"], "B", "#ffdc00")

# EE 차이 계산
ax, ay = res_a["ee"]["pixel_uv"]
bx, by = res_b["ee"]["pixel_uv"]
px_diff = ((ax-bx)**2 + (ay-by)**2) ** 0.5

# ══════════════════════════════════════════════════════════════
# HTML
# ══════════════════════════════════════════════════════════════
html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Pipeline A vs B — {IMG_PATH.name}</title>
<style>
body  {{font-family:'Segoe UI',sans-serif;background:#0a0a0a;color:#e0e0e0;margin:0;padding:20px}}
h1    {{text-align:center;color:#fff;letter-spacing:2px;margin-bottom:4px}}
.sub  {{text-align:center;color:#777;margin-bottom:28px;font-size:13px}}
.cols {{display:flex;gap:24px;justify-content:center;flex-wrap:wrap;margin-bottom:24px}}
.col  {{flex:1;min-width:480px;max-width:560px}}
.col-title {{text-align:center;font-size:18px;font-weight:bold;padding:10px;
             border-radius:8px 8px 0 0;color:#fff}}
.card {{background:#181818;border-radius:10px;overflow:hidden;
        margin-bottom:14px;box-shadow:0 4px 14px rgba(0,0,0,0.5)}}
.ch   {{padding:9px 14px;font-weight:bold;font-size:14px;color:#fff}}
.cb   {{padding:12px}}
.cb img {{width:100%;border-radius:5px;display:block}}
.meta {{margin-top:9px}}
.badge{{display:inline-block;padding:2px 9px;border-radius:10px;
        font-size:12px;color:#fff;margin:2px}}
table {{width:100%;border-collapse:collapse;margin-top:7px;font-size:12px}}
td    {{padding:4px 7px;border-bottom:1px solid #252525;vertical-align:top}}
td.k  {{color:#999;width:38%;font-weight:bold}}
td.v  {{color:#ddd;word-break:break-word}}
.summary {{background:#181818;border-radius:12px;padding:22px 28px;
           max-width:900px;margin:0 auto 24px}}
.summary h2 {{text-align:center;margin:0 0 18px;color:#fff}}
.stat-row {{display:flex;justify-content:center;gap:36px;flex-wrap:wrap}}
.stat {{text-align:center}}
.stat .num {{font-size:32px;font-weight:900}}
.stat .lbl {{font-size:12px;color:#888;margin-top:4px}}
.vs-box {{display:flex;gap:20px;justify-content:center;margin-bottom:24px;flex-wrap:wrap}}
.vs-card {{background:#181818;border-radius:10px;padding:18px 24px;
           flex:1;min-width:280px;max-width:380px;text-align:center}}
.vs-label {{font-size:22px;font-weight:900;margin-bottom:10px}}
.vs-ee    {{font-size:14px;line-height:2;color:#ccc}}
.arrow {{text-align:center;font-size:22px;color:#444;margin:4px 0;width:100%}}
</style>
</head>
<body>
<h1>⚔️ Pipeline A vs B</h1>
<div class="sub">
  <b>{IMG_PATH.name}</b> ({W}×{H}) &nbsp;|&nbsp;
  지시: <b>"{INSTRUCTION}"</b> &nbsp;|&nbsp;
  Mode: <b>{mode}</b>
</div>

<div class="cols">

  <!-- ── Pipeline A ── -->
  <div class="col">
    <div class="col-title" style="background:#1a4a7a">
      🅐 Depth Clustering → VLM 식별
    </div>

    {card("A-1 · 객체 분리 (blind)", img_a1,
          badges=[("Objects", len(res_a["objects"]), "#1a5276"),
                  ("Time", f"{res_a['times']['segment']:.2f}s", "#333"),
                  ("Mode", mode, "#555")],
          rows=[("방식", "depth threshold → connected components"),
                ("ROI", str(ROI)),
                ("감지 수", f"{len(res_a['objects'])}개")],
          color="#1a3a5c")}

    <div class="arrow">↓ VLM에 모든 crop 전달</div>

    {card("A-2 · VLM 타겟 식별", img_a2,
          badges=[("Selected", f"[{res_a['target_idx']}번]", "#1a7a1a"),
                  ("Conf", res_a['id_res'].get('confidence',''), "#7d6608"),
                  ("Time", f"{res_a['times']['identify']:.1f}s", "#333")],
          rows=[("Reason", res_a['id_res'].get('reason','')[:80]),
                ("VLM 입력", f"{len(res_a['objects'])}개 crops"),
                ("Model", "gpt-4o")],
          color="#145a32")}

    <div class="arrow">↓ 타겟 centroid 추출</div>

    {card("A · EE 좌표 출력", img_a_ee,
          badges=[("pixel (u,v)", str(res_a['ee']['pixel_uv']), "#00703c"),
                  ("depth", f"{res_a['ee']['depth_mm']:.0f}mm" if res_a['ee']['depth_mm'] else "N/A", "#5b2c6f"),
                  ("Total", f"{res_a['times']['total']:.1f}s", "#333")],
          rows=[("pixel (u,v)", str(res_a['ee']['pixel_uv'])),
                ("uv_norm", str(res_a['ee']['uv_norm'])),
                ("depth_mm", str(res_a['ee']['depth_mm']))],
          color="#00703c")}
  </div>

  <!-- ── Pipeline B ── -->
  <div class="col">
    <div class="col-title" style="background:#7a4a00">
      🅑 VLM BBox → Depth Refine
    </div>

    {card("B-1 · VLM bbox 추출", img_b1,
          badges=[("bbox", str(res_b['bbox_px']), "#7d6608"),
                  ("Conf", res_b['bbox_res'].get('confidence',''), "#7d6608"),
                  ("Time", f"{res_b['times']['bbox']:.1f}s", "#333")],
          rows=[("label", res_b['bbox_res'].get('label','')),
                ("bbox (pixel)", str(res_b['bbox_px'])),
                ("VLM 입력", "전체 이미지 1장"),
                ("Model", "gpt-4o")],
          color="#7a4a00")}

    <div class="arrow">↓ bbox 내 depth clustering</div>

    {card("B-2 · bbox 내 centroid 정제", res_b['bbox_crop'],
          badges=[("centroid", str(res_b['refine']['centroid_px']), "#5b2c6f"),
                  ("depth", f"{res_b['refine']['depth_mm']:.0f}mm" if res_b['refine']['depth_mm'] else "N/A", "#5b2c6f"),
                  ("Time", f"{res_b['times']['refine']:.2f}s", "#333")],
          rows=[("방식", "bbox 내 depth threshold → centroid"),
                ("centroid_px", str(res_b['refine']['centroid_px'])),
                ("depth_mm", str(res_b['refine']['depth_mm']))],
          color="#5b2c6f")}

    <div class="arrow">↓ EE 좌표 출력</div>

    {card("B · EE 좌표 출력", img_b_ee,
          badges=[("pixel (u,v)", str(res_b['ee']['pixel_uv']), "#b8860b"),
                  ("depth", f"{res_b['ee']['depth_mm']:.0f}mm" if res_b['ee']['depth_mm'] else "N/A", "#5b2c6f"),
                  ("Total", f"{res_b['times']['total']:.1f}s", "#333")],
          rows=[("pixel (u,v)", str(res_b['ee']['pixel_uv'])),
                ("uv_norm", str(res_b['ee']['uv_norm'])),
                ("depth_mm", str(res_b['ee']['depth_mm']))],
          color="#b8860b")}
  </div>
</div>

<div class="summary">
  <h2>📊 비교 요약</h2>
  <div class="stat-row">
    <div class="stat">
      <div class="num" style="color:#3498db">{res_a['times']['total']:.1f}s</div>
      <div class="lbl">Pipeline A 총 시간</div>
    </div>
    <div class="stat">
      <div class="num" style="color:#e67e22">{res_b['times']['total']:.1f}s</div>
      <div class="lbl">Pipeline B 총 시간</div>
    </div>
    <div class="stat">
      <div class="num" style="color:#2ecc71">{len(res_a['objects'])}</div>
      <div class="lbl">A: 감지 객체 수</div>
    </div>
    <div class="stat">
      <div class="num" style="color:#f1c40f">{px_diff:.1f}px</div>
      <div class="lbl">EE 픽셀 차이 (A↔B)</div>
    </div>
  </div>
  <table style="margin-top:20px">
    <tr><td class="k"></td><td class="k" style="color:#3498db">Pipeline A</td>
        <td class="k" style="color:#e67e22">Pipeline B</td></tr>
    <tr><td class="k">VLM 입력</td>
        <td class="v">{len(res_a['objects'])}개 crops</td>
        <td class="v">전체 이미지 1장</td></tr>
    <tr><td class="k">VLM 호출 수</td>
        <td class="v">1회 (식별)</td>
        <td class="v">1회 (bbox)</td></tr>
    <tr><td class="k">객체 분리</td>
        <td class="v">blind ({res_a['times']['segment']:.2f}s)</td>
        <td class="v">bbox 후 refine ({res_b['times']['refine']:.2f}s)</td></tr>
    <tr><td class="k">EE pixel (u,v)</td>
        <td class="v" style="color:#00dc50">{res_a['ee']['pixel_uv']}</td>
        <td class="v" style="color:#ffdc00">{res_b['ee']['pixel_uv']}</td></tr>
    <tr><td class="k">depth</td>
        <td class="v">{res_a['ee']['depth_mm']}</td>
        <td class="v">{res_b['ee']['depth_mm']}</td></tr>
    <tr><td class="k">EE 픽셀 차이</td>
        <td class="v" colspan="2" style="text-align:center"><b>{px_diff:.1f}px</b></td></tr>
    <tr><td class="k">강점</td>
        <td class="v">컨텍스트 기반 식별 정확도</td>
        <td class="v">빠른 VLM 호출, 장면 전체 파악</td></tr>
    <tr><td class="k">약점</td>
        <td class="v">많은 객체 시 crop 수 증가</td>
        <td class="v">bbox 오류 시 downstream 영향</td></tr>
  </table>
</div>

</body>
</html>"""

out = OUT_DIR / f"{IMG_PATH.stem}_compare_ab.html"
out.write_text(html, encoding="utf-8")
print(f"\n{'='*55}")
print(f"  저장: {out}")
print(f"  A total: {res_a['times']['total']:.1f}s  |  B total: {res_b['times']['total']:.1f}s")
print(f"  EE 차이: {px_diff:.1f}px")
