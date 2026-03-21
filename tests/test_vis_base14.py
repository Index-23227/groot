"""
base15.jpg CoT 파이프라인 시각화 (VLM only, depth clustering 제외)

Step 0: 원본 이미지 + 씬 객체 목록
Step 1: GPT-4o CoT 추론 (4단계)
Step 2: EE 좌표 + depth lookup
"""
import sys, io, base64, time
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).parent.parent))
from utils.pipeline_a import DepthClusterPipeline, _to_b64

ROOT       = Path(__file__).parent.parent
IMG_PATH   = ROOT / "data" / "base_images" / "base15.jpg"
DEPTH_PATH = ROOT / "data" / "base_images" / "base15_depth.npy"
OUT_DIR    = ROOT / "results" / "vis_base15"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INSTRUCTION = "노란 원기둥을 집어라"

# 씬 내 7개 객체 primitive 서술
SCENE_OBJECTS = [
    "파란 에너지드링크 캔 (Monster Energy, 키 큰 원통)",
    "초록 음료 캔 (스프라이트 계열, 중간 크기)",
    "갈색 음료 캔 (커피 계열, 중간 크기)",
    "노란 원기둥 (작은 플라스틱/나무 원기둥)",
    "초록 원기둥 (작은 플라스틱/나무 원기둥)",
    "파란 원기둥 (작은 플라스틱/나무 원기둥)",
    "빨간 원기둥 (작은 플라스틱/나무 원기둥)",
]

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

# ── CoT 시각화 함수 ──────────────────────────────────────────
def draw_cot_result(rgb, cot_res):
    """CoT 4단계를 이미지에 시각화"""
    img = Image.fromarray(rgb).copy().convert("RGBA")
    H, W = rgb.shape[:2]
    d = ImageDraw.Draw(img)

    # Step2: 객체 전체 bbox (파란색)
    bx1, by1, bx2, by2 = cot_res.get("obj_bbox_px", (0,0,W,H))
    d.rectangle([bx1, by1, bx2, by2], outline=(100, 180, 255, 255), width=3)
    d.rectangle([bx1, by1-22, bx2, by1], fill=(30, 80, 160, 220))
    d.text((bx1+4, by1-20), "Step2: Object BBox", fill=(180, 220, 255, 255), font=FONT_SM)

    # Step3: 윗면 bbox (노란색)
    tx1, ty1, tx2, ty2 = cot_res.get("top_bbox_px", (0,0,W,H))
    d.rectangle([tx1, ty1, tx2, ty2], outline=(255, 230, 0, 255), width=3)
    d.rectangle([tx1, ty1-22, tx2, ty1], fill=(120, 90, 0, 220))
    d.text((tx1+4, ty1-20), "Step3: Top Surface BBox", fill=(255, 230, 100, 255), font=FONT_SM)

    # Step4: 윗면 중심점 (빨간 크로스헤어)
    cx, cy = cot_res.get("ee_px", (W//2, H//2))
    r = 16
    d.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(255, 50, 50, 255), width=4)
    d.line([(cx-30, cy), (cx+30, cy)], fill=(255, 50, 50, 255), width=3)
    d.line([(cx, cy-30), (cx, cy+30)], fill=(255, 50, 50, 255), width=3)

    # 정보 박스
    u_n, v_n = cot_res.get("uv_norm", (0.5, 0.5))
    lines = [
        "Step4: EE Target (Top Center)",
        f"pixel  u={cx}, v={cy}",
        f"norm   ({u_n:.3f}, {v_n:.3f})",
    ]
    bx, by = min(cx+20, W-260), max(cy-60, 10)
    d.rectangle([bx-4, by-4, bx+250, by+len(lines)*20+4], fill=(20, 20, 20, 210))
    for j, ln in enumerate(lines):
        col = (255, 100, 100, 255) if j == 0 else (220, 220, 220, 255)
        d.text((bx, by+j*20), ln, fill=col, font=FONT_SM if j > 0 else FONT_MD)

    return img.convert("RGB")

# ══════════════════════════════════════════════════════════════
print("="*55)
print(f"  CoT Pipeline — {IMG_PATH.name}")
print(f"  지시: {INSTRUCTION}")
print(f"  씬 객체: {len(SCENE_OBJECTS)}개")
print("="*55)

t_total = time.time()
rgb_np  = np.array(Image.open(IMG_PATH).convert("RGB"))
H, W    = rgb_np.shape[:2]

# depth 로드 (EE depth lookup용)
depth_np = None
if DEPTH_PATH.exists():
    depth_np = np.load(DEPTH_PATH)
    d_vals = depth_np[depth_np > 0]
    print(f"Depth 로드: {DEPTH_PATH.name}  range={d_vals.min():.0f}~{d_vals.max():.0f}mm  median={np.median(d_vals):.0f}mm")
else:
    print("depth 파일 없음")

# ── Step 0: 원본 이미지 ──────────────────────────────────────
img_s0 = Image.fromarray(rgb_np).copy()

# ── Step 1: CoT VLM 추론 ────────────────────────────────────
print(f"\n[CoT] GPT-4o 추론 시작...")
pipeline = DepthClusterPipeline()
t_vlm = time.time()
cot_res = pipeline.identify_cot(rgb_np, INSTRUCTION, scene_objects=SCENE_OBJECTS)
t_vlm = time.time() - t_vlm
print(f"  완료  ({t_vlm:.1f}s)")

img_cot = draw_cot_result(rgb_np, cot_res)

# ── EE 좌표 + depth lookup ──────────────────────────────────
ee_px = cot_res.get("ee_px", (0, 0))
cx, cy = ee_px
u_n, v_n = cot_res.get("uv_norm", (0.5, 0.5))

# depth: EE 좌표 근처 depth 값 직접 lookup
depth_str = "N/A"
depth_mm = None
if depth_np is not None:
    # EE 픽셀 주변 5x5 패치에서 median depth
    r = 5
    patch = depth_np[max(0,cy-r):min(H,cy+r), max(0,cx-r):min(W,cx+r)]
    valid = patch[patch > 0]
    if len(valid) > 0:
        depth_mm = float(np.median(valid))
        depth_str = f"{depth_mm:.0f}mm"

t_total = time.time() - t_total

# ── HTML 생성 ─────────────────────────────────────────────────
obj_list_html = "".join(f"<li>{o}</li>" for o in SCENE_OBJECTS)

html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>CoT Pipeline — {IMG_PATH.name}</title>
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
  .obj-list {{ margin-top:8px; padding:6px 14px; background:#1a1a1a;
               border-radius:6px; font-size:12px; color:#bbb }}
  .obj-list ol {{ margin:6px 0 0 16px }}
  .obj-list li {{ margin-bottom:2px }}
</style>
</head>
<body>
<h1>CoT Pipeline Visualization</h1>
<div class="sub">
  <b>{IMG_PATH.name}</b> ({W}x{H}) &nbsp;|&nbsp;
  지시: <b>"{INSTRUCTION}"</b> &nbsp;|&nbsp;
  Total: <b>{t_total:.1f}s</b>
</div>

<div class="pipeline">

{card("Step 0 · 원본 이미지 + 씬 객체 목록", img_s0,
    badges=[("이미지", IMG_PATH.name, "#2c5f8a"),
            ("객체 수", f"{len(SCENE_OBJECTS)}개", "#444")],
    rows=[("해상도", f"{W}x{H}"),
          ("depth", f"{DEPTH_PATH.name}" if depth_np is not None else "없음"),
          ("카메라", "측면+위 약 45도 각도")],
    note=f'<div class="obj-list"><b>씬 Primitive 목록 ({len(SCENE_OBJECTS)}개):</b><ol>{obj_list_html}</ol></div>',
    color="#1a3a5c")}

<div class="arrow">↓</div>

{card("Step 1 · GPT-4o CoT 추론", img_cot,
    badges=[("Conf", cot_res.get('confidence',''), "#7d6608"),
            ("시간", f"{t_vlm:.1f}s", "#444")],
    rows=[("CoT Step1", f"타겟: {cot_res.get('step1_target','')}"),
          ("CoT Step2", f"Object BBox: {cot_res.get('obj_bbox_px','')}"),
          ("CoT Step3", f"Top BBox: {cot_res.get('top_bbox_px','')}"),
          ("CoT Step4", f"Top Center: pixel({cx},{cy})  norm({u_n:.3f},{v_n:.3f})"),
          ("근거", cot_res.get('reason','')[:100])],
    note="파란 박스=객체 전체 BBox / 노란 박스=윗면 BBox / 빨간 십자=EE 타겟 (윗면 중심점)",
    color="#145a32")}

<div class="arrow">↓</div>

{card("Step 2 · EE 좌표 + Depth Lookup", img_cot,
    badges=[("pixel (u,v)", f"({cx}, {cy})", "#6a0dad"),
            ("depth", depth_str, "#5b2c6f"),
            ("norm", f"({u_n:.3f}, {v_n:.3f})", "#444")],
    rows=[("EE pixel (u,v)", f"({cx}, {cy})"),
          ("normalized", f"({u_n:.4f}, {v_n:.4f})"),
          ("depth", f"{depth_str} (EE 픽셀 주변 5x5 median)"),
          ("좌표 기준", "타겟 객체 윗면 중심 → 그리퍼 접근점"),
          ("다음 단계", "캘리브레이션 행렬 → 로봇 좌표 (X,Y,Z)")],
    color="#3d1a5c")}

</div>

<div class="final">
  <div class="final-label">EE Target (윗면 중심점)</div>
  <div class="final-detail">
    <b>CoT 추론 결과</b><br>
    Step1 → <b>{cot_res.get('step1_target','')}</b><br>
    Step2 → Object BBox <b>{cot_res.get('obj_bbox_px','')}</b><br>
    Step3 → Top Surface BBox <b>{cot_res.get('top_bbox_px','')}</b><br>
    Step4 → EE pixel &nbsp;<b>u={cx}, v={cy}</b> &nbsp;|&nbsp; norm <b>({u_n:.4f}, {v_n:.4f})</b><br>
    depth &nbsp;→&nbsp; <b>{depth_str}</b><br>
    <span style="color:#666;font-size:12px">
      → 캘리브레이션 완료 시: (X, Y, Z)_robot 으로 변환 후 로봇 이동
    </span>
  </div>
</div>

<div class="timing">
  CoT VLM: {t_vlm:.1f}s &nbsp;|&nbsp;
  <b>Total: {t_total:.1f}s</b>
</div>

</body>
</html>"""

out = OUT_DIR / "base15_pipeline.html"
out.write_text(html, encoding="utf-8")
print(f"\n저장: {out}")
print(f"총 소요: {t_total:.1f}s")
