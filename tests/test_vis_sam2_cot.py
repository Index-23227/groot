"""
SAM2-tiny + GPT-4o CoT 파이프라인 시각화

Flow:
  Step 0: 원본 이미지 + 씬 primitive 목록
  Step 1: SAM2-tiny blind segmentation → 모든 객체 mask/crop
  Step 2: GPT-4o CoT (crops 전달)
          → 1) 타겟 식별, 2) 타겟 bbox, 3) 윗면 bbox, 4) 윗면 중심점
  Step 3: EE 좌표 (원본 좌표 변환) + depth lookup
"""
import sys, io, base64, time, pickle, random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).parent.parent))

import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from openai import OpenAI

ROOT       = Path(__file__).parent.parent
IMG_PATH   = ROOT / "data" / "base_images" / "base15.jpg"
DEPTH_PATH = ROOT / "data" / "base_images" / "base15_depth.npy"
OUT_DIR    = ROOT / "results" / "vis_sam2_cot"
CACHE_DIR  = ROOT / "results" / "sam2_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
SAM2_CFG   = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

OPENAI_KEY = next(l.strip() for l in (ROOT/"token").read_text().splitlines()
                  if l.strip().startswith("sk-"))

INSTRUCTION = "노란 원기둥을 집어라"

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
def _to_b64(img: Image.Image, max_side: int = 512) -> str:
    img = img.copy(); img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

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

def make_crop(image, mask_data, padding=12):
    """mask 영역 crop, 배경 흰색"""
    mask = mask_data["segmentation"].astype(bool)
    h, w = image.shape[:2]
    x, y, bw, bh = [int(v) for v in mask_data["bbox"]]
    x1, y1 = max(0, x-padding), max(0, y-padding)
    x2, y2 = min(w, x+bw+padding), min(h, y+bh+padding)
    arr = np.array(Image.fromarray(image).convert("RGBA"))
    arr[~mask] = [255, 255, 255, 255]
    return Image.fromarray(arr).convert("RGB").crop((x1, y1, x2, y2))


# ── GPT-4o CoT (crops 기반) ──────────────────────────────────
def identify_cot_with_crops(client, crops, instruction, scene_objects):
    """
    SAM2 crops를 GPT-4o에 전달 → CoT로 타겟 식별 + 윗면 중심점.

    Returns:
      {target_index, step1_target, top_center_crop_norm {u, v},
       confidence, reason}
    """
    obj_list = "\n".join(f"  {i+1}. {o}" for i, o in enumerate(scene_objects))
    prompt = (
        f"현재 테이블 위에 {len(scene_objects)}개의 객체가 있습니다:\n"
        f"{obj_list}\n\n"
        f"지시: {instruction}\n\n"
        f"아래 {len(crops)}개 이미지는 각각 씬에서 분리된 개별 객체입니다.\n"
        "배경은 흰색으로 처리되어 있고, 객체만 보입니다.\n\n"
        "단계별(Chain-of-Thought)로 추론하세요:\n\n"
        "Step 1. 지시에 해당하는 타겟 객체의 번호(0-based)를 식별하세요.\n"
        "Step 2. 선택한 crop 이미지에서 타겟 객체의 전체 영역을 설명하세요.\n"
        "Step 3. 타겟 객체의 윗면(top surface)을 찾으세요.\n"
        "        - 원기둥이면 상단 원형/타원 면\n"
        "        - 캔이면 뚜껑 면\n"
        "        - 카메라가 측면+위 약 45도 각도임을 고려\n"
        "Step 4. crop 이미지 내에서 윗면의 중심점 좌표를 구하세요 (normalized 0~1).\n"
        "        → 이것이 로봇 End-Effector가 접근할 타겟 위치입니다.\n\n"
        "반드시 아래 JSON 형식으로만 답하세요:\n"
        '{"target_index": 0,'
        ' "step1_target": "타겟 객체 설명",'
        ' "step2_description": "객체 전체 설명",'
        ' "step3_top_surface": "윗면 설명",'
        ' "step4_top_center_crop": {"u": 0.5, "v": 0.3},'
        ' "confidence": 0.9,'
        ' "reason": "추론 근거"}'
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
                model="gpt-4o", temperature=0.1, max_tokens=500,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": content}],
            )
            raw = r.choices[0].message.content
            s, e = raw.find("{"), raw.rfind("}")+1
            return json.loads(raw[s:e]) if s >= 0 and e > 0 else {}
        except Exception as e:
            wait = 15 if "429" in str(e) else 8
            print(f"    retry {attempt+1}/3 ({wait}s)")
            time.sleep(wait)
    return {}

import json

# ── 시각화 함수들 ─────────────────────────────────────────────
def draw_sam2_overlay(rgb, masks, target_idx=None):
    """모든 SAM2 마스크 오버레이, 타겟은 강조"""
    random.seed(42)
    img = Image.fromarray(rgb).convert("RGBA")
    H, W = rgb.shape[:2]
    for i, m in enumerate(masks):
        if i == target_idx:
            col = (0, 220, 80)
            alpha = 180
        else:
            col = (random.randint(80,230), random.randint(80,230), random.randint(80,230))
            alpha = 100
        lyr = np.zeros((H, W, 4), dtype=np.uint8)
        lyr[m["segmentation"]] = [*col, alpha]
        img = Image.alpha_composite(img, Image.fromarray(lyr))
        x, y, bw, bh = [int(v) for v in m["bbox"]]
        d = ImageDraw.Draw(img)
        outline_col = (0,255,80,255) if i == target_idx else (*col, 200)
        width = 4 if i == target_idx else 2
        d.rectangle([x, y, x+bw, y+bh], outline=outline_col, width=width)
        d.text((x+3, y+3), str(i), fill=(255,255,255,255), font=FONT_SM)
    return img.convert("RGB")

def draw_crop_grid(masks, crops):
    THUMB = 140
    n = len(crops)
    cols = min(n, 6)
    rows_n = (n + cols - 1) // cols
    grid = Image.new("RGB", (cols*(THUMB+8), rows_n*(THUMB+46)), (20,20,20))
    for i, (m, crop) in enumerate(zip(masks, crops)):
        col_i = i % cols
        row_i = i // cols
        t = crop.copy(); t.thumbnail((THUMB, THUMB))
        xo = col_i*(THUMB+8) + (THUMB-t.width)//2
        yo = row_i*(THUMB+46) + 4
        grid.paste(t, (xo, yo))
        d = ImageDraw.Draw(grid)
        d.rectangle([col_i*(THUMB+8), row_i*(THUMB+46),
                     col_i*(THUMB+8)+THUMB+6, row_i*(THUMB+46)+THUMB+8],
                    outline=(80,80,80), width=1)
        d.text((col_i*(THUMB+8)+4, row_i*(THUMB+46)+THUMB+12),
               f"[{i}] {m['area']}px", fill=(160,160,160), font=FONT_SM)
    return grid

def draw_ee_on_image(rgb, ee_px, depth_str, cot_res, mask_data):
    """원본 이미지 위에 타겟 마스크 + EE 크로스헤어"""
    img = Image.fromarray(rgb).convert("RGBA")
    H, W = rgb.shape[:2]
    # 타겟 마스크 강조
    lyr = np.zeros((H, W, 4), dtype=np.uint8)
    lyr[mask_data["segmentation"]] = [0, 220, 80, 160]
    img = Image.alpha_composite(img, Image.fromarray(lyr))
    d = ImageDraw.Draw(img)
    # bbox
    x, y, bw, bh = [int(v) for v in mask_data["bbox"]]
    d.rectangle([x, y, x+bw, y+bh], outline=(0, 255, 80, 255), width=4)
    # EE 크로스헤어
    cx, cy = ee_px
    r = 16
    d.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(255, 50, 50, 255), width=4)
    d.line([(cx-30, cy), (cx+30, cy)], fill=(255, 50, 50, 255), width=3)
    d.line([(cx, cy-30), (cx, cy+30)], fill=(255, 50, 50, 255), width=3)
    # 정보 박스
    lines = [
        "EE Target (Top Center)",
        f"pixel  u={cx}, v={cy}",
        f"depth  {depth_str}",
    ]
    bx, by = min(cx+20, W-220), max(cy-60, 10)
    d.rectangle([bx-4, by-4, bx+220, by+len(lines)*20+4], fill=(20, 20, 20, 210))
    for j, ln in enumerate(lines):
        col = (255, 100, 100, 255) if j==0 else (220, 220, 220, 255)
        d.text((bx, by+j*20), ln, fill=col, font=FONT_SM if j>0 else FONT_MD)
    return img.convert("RGB")

def draw_cot_on_crop(crop, cot_res):
    """crop 이미지에 CoT top_center 표시"""
    img = crop.copy().convert("RGBA")
    cw, ch = img.size
    tc = cot_res.get("step4_top_center_crop", {})
    u = float(tc.get("u", 0.5))
    v = float(tc.get("v", 0.5))
    cx, cy = int(u * cw), int(v * ch)
    d = ImageDraw.Draw(img)
    r = 10
    d.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(255, 50, 50, 255), width=3)
    d.line([(cx-18, cy), (cx+18, cy)], fill=(255, 50, 50, 255), width=2)
    d.line([(cx, cy-18), (cx, cy+18)], fill=(255, 50, 50, 255), width=2)
    d.text((cx+12, cy-10), f"({u:.2f},{v:.2f})", fill=(255,100,100,255), font=FONT_SM)
    return img.convert("RGB")


# ══════════════════════════════════════════════════════════════
print("="*55)
print(f"  SAM2 + CoT Pipeline — {IMG_PATH.name}")
print(f"  지시: {INSTRUCTION}")
print(f"  씬 객체: {len(SCENE_OBJECTS)}개")
print("="*55)

t_total = time.time()
rgb_np = np.array(Image.open(IMG_PATH).convert("RGB"))
H, W   = rgb_np.shape[:2]

# depth 로드
depth_np = None
if DEPTH_PATH.exists():
    depth_np = np.load(DEPTH_PATH)
    d_vals = depth_np[depth_np > 0]
    print(f"Depth: {d_vals.min():.0f}~{d_vals.max():.0f}mm  median={np.median(d_vals):.0f}mm")

# ── ROI: 좌/중/우 3등분 → 가운데만 ────────────────────────────
ROI_X1 = W // 3 - int(W * 0.10)
ROI_X2 = 2 * W // 3 + int(W * 0.10)
ROI = (ROI_X1, 0, ROI_X2, H)
print(f"ROI (가운데 1/3): x={ROI_X1}~{ROI_X2}  ({ROI_X2-ROI_X1}x{H}px)")

# ── Step 0: 원본 이미지 + ROI 표시 ───────────────────────────
img_s0_pil = Image.fromarray(rgb_np).convert("RGBA")
overlay = Image.new("RGBA", img_s0_pil.size, (0,0,0,0))
od = ImageDraw.Draw(overlay)
od.rectangle([0, 0, ROI_X1, H], fill=(0,0,0,120))
od.rectangle([ROI_X2, 0, W, H], fill=(0,0,0,120))
img_s0 = Image.alpha_composite(img_s0_pil, overlay).convert("RGB")
d0 = ImageDraw.Draw(img_s0)
d0.rectangle([ROI_X1, 0, ROI_X2, H], outline=(0,200,255), width=3)
d0.text((ROI_X1+8, 8), f"ROI: {ROI_X2-ROI_X1}x{H}px", fill=(0,200,255), font=FONT_MD)

# ── Step 1: SAM2 blind segmentation (가운데 ROI만) ───────────
roi_rgb = rgb_np[:, ROI_X1:ROI_X2]
cache_path = CACHE_DIR / "base15_roi_masks.pkl"
print(f"\n[Step 1] SAM2-tiny segmentation (ROI)...")
t1 = time.time()
if cache_path.exists():
    masks = pickle.loads(cache_path.read_bytes())
    print(f"  캐시 로드: {len(masks)}개 마스크")
else:
    print(f"  SAM2 로딩... (device={DEVICE})")
    sam2_model = build_sam2(SAM2_CFG, str(CHECKPOINT), device=DEVICE)
    generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=16,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.90,
        min_mask_region_area=800,
    )
    raw_masks = sorted(generator.generate(roi_rgb), key=lambda x: x["area"], reverse=True)
    # ROI 좌표 → 원본 좌표로 변환
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
    cache_path.write_bytes(pickle.dumps(masks))
    print(f"  {len(masks)}개 마스크 → 캐시 저장")
t1 = time.time() - t1
print(f"  완료 ({t1:.1f}s)")

crops = [make_crop(rgb_np, m) for m in masks]
img_s1 = draw_sam2_overlay(rgb_np, masks)
img_crops = draw_crop_grid(masks, crops)

# ── Step 2: GPT-4o CoT ──────────────────────────────────────
client = OpenAI(api_key=OPENAI_KEY)
print(f"\n[Step 2] GPT-4o CoT ({len(crops)}개 crops)...")
t2 = time.time()
cot_res = identify_cot_with_crops(client, crops, INSTRUCTION, SCENE_OBJECTS)
t2 = time.time() - t2

target_idx = min(int(cot_res.get("target_index", 0)), len(masks)-1)
target_mask = masks[target_idx]
target_crop = crops[target_idx]
print(f"  타겟: [{target_idx}번]  conf={cot_res.get('confidence')}")
print(f"  Step1: {cot_res.get('step1_target','')}")
print(f"  Step3: {cot_res.get('step3_top_surface','')}")
tc = cot_res.get("step4_top_center_crop", {})
print(f"  Step4 (crop norm): u={tc.get('u')}, v={tc.get('v')}")
print(f"  완료 ({t2:.1f}s)")

# 타겟 강조된 SAM2 오버레이
img_s1_target = draw_sam2_overlay(rgb_np, masks, target_idx=target_idx)

# crop 위에 CoT top_center 표시
img_crop_cot = draw_cot_on_crop(target_crop, cot_res)

# ── Step 3: crop 좌표 → 원본 좌표 변환 ──────────────────────
# mask bbox + padding → crop 원점
bx, by, bbw, bbh = [int(v) for v in target_mask["bbox"]]
pad = 12
crop_x1 = max(0, bx - pad)
crop_y1 = max(0, by - pad)
crop_w  = target_crop.width
crop_h  = target_crop.height

u_crop = float(tc.get("u", 0.5))
v_crop = float(tc.get("v", 0.5))

# crop 내 좌표 → 원본 이미지 좌표
ee_x = crop_x1 + int(u_crop * crop_w)
ee_y = crop_y1 + int(v_crop * crop_h)
ee_px = (ee_x, ee_y)
u_n = ee_x / W
v_n = ee_y / H

# depth lookup
depth_str = "N/A"
depth_mm = None
if depth_np is not None:
    r = 5
    patch = depth_np[max(0,ee_y-r):min(H,ee_y+r), max(0,ee_x-r):min(W,ee_x+r)]
    valid = patch[patch > 0]
    if len(valid) > 0:
        depth_mm = float(np.median(valid))
        depth_str = f"{depth_mm:.0f}mm"

print(f"\n[결과] EE pixel: ({ee_x}, {ee_y})  norm: ({u_n:.3f}, {v_n:.3f})  depth: {depth_str}")

img_s3 = draw_ee_on_image(rgb_np, ee_px, depth_str, cot_res, target_mask)

t_total = time.time() - t_total

# ── HTML 생성 ─────────────────────────────────────────────────
obj_list_html = "".join(f"<li>{o}</li>" for o in SCENE_OBJECTS)

html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>SAM2 + CoT Pipeline — {IMG_PATH.name}</title>
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
  .two-col {{ display:flex; gap:10px; margin-top:8px }}
  .two-col img {{ width:50%; border-radius:6px }}
</style>
</head>
<body>
<h1>SAM2 + CoT Pipeline</h1>
<div class="sub">
  <b>{IMG_PATH.name}</b> ({W}x{H}) &nbsp;|&nbsp;
  지시: <b>"{INSTRUCTION}"</b> &nbsp;|&nbsp;
  Total: <b>{t_total:.1f}s</b>
</div>

<div class="pipeline">

{card("Step 0 · 원본 이미지 + ROI (가운데 1/3)", img_s0,
    badges=[("이미지", IMG_PATH.name, "#2c5f8a"),
            ("ROI", f"x={ROI_X1}~{ROI_X2}", "#2c5f8a"),
            ("객체 수", f"{len(SCENE_OBJECTS)}개", "#444")],
    rows=[("해상도", f"{W}x{H}"),
          ("ROI", f"좌/중/우 3등분 → 가운데 ({ROI_X2-ROI_X1}x{H}px)"),
          ("depth", f"{DEPTH_PATH.name}" if depth_np is not None else "없음"),
          ("카메라", "측면+위 약 45도 (고정)")],
    note=f'<div class="obj-list"><b>씬 Primitive ({len(SCENE_OBJECTS)}개):</b><ol>{obj_list_html}</ol></div>',
    color="#1a3a5c")}

<div class="arrow">↓</div>

{card(f"Step 1 · SAM2-tiny (ROI only) — {len(masks)}개 객체", img_s1,
    badges=[("마스크", f"{len(masks)}개", "#1a5276"),
            ("시간", f"{t1:.1f}s", "#2e4057"),
            ("ROI", f"{ROI_X2-ROI_X1}x{H}", "#555")],
    rows=[("방식", "SAM2 AutomaticMaskGenerator (가운데 1/3 ROI만)"),
          ("파라미터", "points_per_side=16, iou=0.80, stability=0.90"),
          ("결과", f"{len(masks)}개 마스크 → 좌표 원본으로 복원")],
    note="ROI 밖(어두운 영역)은 SAM2 처리 제외 → 속도/정확도 향상",
    color="#1a3a5c")}

<div class="arrow">↓</div>

{card(f"SAM2 Crops ({len(crops)}개)", img_crops,
    badges=[("crops", f"{len(crops)}개", "#5b4a00"),
            ("배경 제거", "mask 기반 흰색", "#2e4057")],
    rows=[("전처리", "SAM2 mask로 배경 흰색 처리 → 객체만 crop"),
          ("전달", "전체 crops → GPT-4o CoT")],
    color="#3d3000")}

<div class="arrow">↓</div>

{card(f"Step 2 · GPT-4o CoT → [{target_idx}번] 타겟", img_s1_target,
    badges=[("Target", f"[{target_idx}번]", "#1a7a1a"),
            ("Conf", cot_res.get('confidence',''), "#7d6608"),
            ("시간", f"{t2:.1f}s", "#444")],
    rows=[("CoT Step1", f"타겟: {cot_res.get('step1_target','')}"),
          ("CoT Step2", f"{cot_res.get('step2_description','')[:80]}"),
          ("CoT Step3", f"윗면: {cot_res.get('step3_top_surface','')[:80]}"),
          ("CoT Step4", f"crop 내 윗면 중심: u={tc.get('u')}, v={tc.get('v')}"),
          ("근거", cot_res.get('reason','')[:80])],
    note="초록 강조 = 타겟 객체 / GPT-4o는 crop 내에서 윗면 중심을 추론",
    color="#145a32")}

<div class="arrow">↓</div>

{card("CoT 결과: 타겟 crop + 윗면 중심점", img_crop_cot,
    badges=[("crop 내 norm", f"({tc.get('u','')}, {tc.get('v','')})", "#6a0dad"),
            ("원본 pixel", f"({ee_x}, {ee_y})", "#5b2c6f")],
    rows=[("crop bbox", f"({bx},{by},{bbw},{bbh}) + pad={pad}"),
          ("crop 원점", f"({crop_x1}, {crop_y1})"),
          ("변환", f"crop({u_crop:.2f},{v_crop:.2f}) → 원본({ee_x},{ee_y})")],
    note="빨간 십자 = crop 이미지 내 윗면 중심점 → 원본 좌표로 변환",
    color="#3d1a5c")}

<div class="arrow">↓</div>

{card("Step 3 · EE 좌표 (원본 이미지)", img_s3,
    badges=[("pixel (u,v)", f"({ee_x}, {ee_y})", "#6a0dad"),
            ("depth", depth_str, "#5b2c6f"),
            ("norm", f"({u_n:.3f}, {v_n:.3f})", "#444")],
    rows=[("EE pixel (u,v)", f"({ee_x}, {ee_y})"),
          ("normalized", f"({u_n:.4f}, {v_n:.4f})"),
          ("depth", f"{depth_str} (EE 픽셀 주변 5x5 median)"),
          ("좌표 기준", "SAM2 mask bbox 좌표 + CoT crop 내 윗면 중심"),
          ("다음 단계", "캘리브레이션 행렬 → 로봇 좌표 (X,Y,Z)")],
    color="#3d1a5c")}

</div>

<div class="final">
  <div class="final-label">EE Target (윗면 중심점)</div>
  <div class="final-detail">
    <b>SAM2 + CoT 결과</b><br>
    SAM2 → <b>{len(masks)}개 마스크</b> → 타겟 <b>[{target_idx}번]</b><br>
    CoT Step1 → <b>{cot_res.get('step1_target','')}</b><br>
    CoT Step4 → crop 내 <b>({tc.get('u','')}, {tc.get('v','')})</b><br>
    원본 EE pixel → <b>u={ee_x}, v={ee_y}</b> &nbsp;|&nbsp; norm <b>({u_n:.4f}, {v_n:.4f})</b><br>
    depth → <b>{depth_str}</b><br>
    <span style="color:#666;font-size:12px">
      → 캘리브레이션 완료 시: (X, Y, Z)_robot 으로 변환 후 로봇 이동
    </span>
  </div>
</div>

<div class="timing">
  SAM2: {t1:.1f}s &nbsp;|&nbsp;
  CoT VLM: {t2:.1f}s &nbsp;|&nbsp;
  <b>Total: {t_total:.1f}s</b>
</div>

</body>
</html>"""

out = OUT_DIR / "base15_sam2_cot.html"
out.write_text(html, encoding="utf-8")
print(f"\n저장: {out}")
print(f"총 소요: {t_total:.1f}s")
