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

INSTRUCTION = "빨간 원기둥을 파란 원기둥 옆에 놓아라"

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


# ── GPT-4o CoT (crops 기반, pick & place 대응) ───────────────
def identify_cot_with_crops(client, crops, instruction, scene_objects):
    """
    SAM2 crops → GPT-4o CoT.
    pick & place 지시를 파싱하여:
      - pick 대상 객체 식별 + 윗면 중심점
      - place 위치 참조 객체 식별 + 배치 위치
      - action sequence 생성
    """
    obj_list = "\n".join(f"  {i+1}. {o}" for i, o in enumerate(scene_objects))
    prompt = (
        f"현재 테이블 위에 {len(scene_objects)}개의 객체가 있습니다:\n"
        f"{obj_list}\n\n"
        f"지시: {instruction}\n\n"
        f"아래 {len(crops)}개 이미지는 각각 씬에서 분리된 개별 객체입니다.\n"
        "배경은 흰색으로 처리되어 있고, 객체만 보입니다.\n\n"
        "이 지시는 pick-and-place 태스크입니다.\n"
        "단계별(Chain-of-Thought)로 추론하세요:\n\n"
        "Step 1. PICK 대상: 지시에서 집어야 할 객체의 번호(0-based)를 식별하세요.\n"
        "Step 2. PICK 윗면: 해당 crop에서 객체의 윗면(top surface)의 중심점을 찾으세요.\n"
        "        crop 내 윗면 중심점 좌표를 구하세요 (normalized 0~1).\n"
        "        - 캔이면 뚜껑 면의 중심, 원기둥이면 상단 원형 면의 중심\n"
        "        - 카메라가 측면+위 약 45도 각도임을 고려\n"
        "        - 윗면의 정중앙이 End-Effector가 접근할 위치임\n"
        "Step 3. PLACE 참조: 지시에서 놓을 위치의 참조 객체 번호(0-based)를 식별하세요.\n"
        "        참조 객체가 없으면(예: '테이블 위에 놓아라') null로 하세요.\n"
        "Step 4. PLACE 위치: 참조 객체의 crop에서 놓을 위치를 추론하세요.\n"
        "        - '옆에' → 참조 객체 오른쪽 또는 왼쪽\n"
        "        - '위에' → 참조 객체 윗면 위\n"
        "        - '앞에/뒤에' → 카메라 기준 앞/뒤\n"
        "        - 중요: 참조 객체와 충분한 간격을 두세요. 너무 가까우면 충돌합니다.\n"
        "          참조 객체 크기의 1.5~2배 정도 떨어진 위치가 적절합니다.\n"
        "        참조 crop 내 normalized 좌표 (0~1)로 답하세요.\n"
        "        crop 밖(0 미만 또는 1 초과)도 가능합니다.\n"
        "Step 5. ACTION: 로봇이 수행할 action sequence를 생성하세요.\n\n"
        "반드시 아래 JSON 형식으로만 답하세요.\n"
        "각 step마다 rationale(판단 근거)을 반드시 포함하세요:\n"
        '{"step1_pick_index": 0,'
        ' "step1_pick_description": "집을 객체 설명",'
        ' "step1_rationale": "왜 이 객체를 선택했는지 근거",'
        ' "step2_pick_top_center": {"u": 0.5, "v": 0.3},'
        ' "step2_pick_surface": "윗면 설명",'
        ' "step2_rationale": "윗면을 어떻게 판단했는지, 중심점을 어떻게 결정했는지",'
        ' "step3_place_ref_index": 1,'
        ' "step3_place_ref_description": "참조 객체 설명",'
        ' "step3_rationale": "왜 이 객체가 place 참조인지 근거",'
        ' "step4_place_position": {"u": 0.5, "v": 0.5},'
        ' "step4_place_relation": "옆에/위에/앞에 등",'
        ' "step4_rationale": "place 위치를 어떻게 결정했는지 근거",'
        ' "step5_actions": ["move_to_pick", "grasp", "lift", "move_to_place", "release"],'
        ' "step5_rationale": "action sequence 구성 근거",'
        ' "confidence": 0.9}'
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
                model="gpt-4o", temperature=0.1, max_tokens=800,
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

# ── pick 객체 ────────────────────────────────────────────────
pick_idx = min(int(cot_res.get("step1_pick_index", 0)), len(masks)-1)
pick_mask = masks[pick_idx]
pick_crop = crops[pick_idx]
pick_tc = cot_res.get("step2_pick_top_center", {})

# ── place 참조 객체 ──────────────────────────────────────────
place_ref_idx = cot_res.get("step3_place_ref_index")
place_mask = None
place_crop = None
place_tc = cot_res.get("step4_place_position", {})
if place_ref_idx is not None:
    place_ref_idx = min(int(place_ref_idx), len(masks)-1)
    place_mask = masks[place_ref_idx]
    place_crop = crops[place_ref_idx]

actions = cot_res.get("step5_actions", [])

print(f"  PICK:  [{pick_idx}번] {cot_res.get('step1_pick_description','')}")
print(f"    rationale: {cot_res.get('step1_rationale','')}")
print(f"         윗면: {cot_res.get('step2_pick_surface','')}")
print(f"         crop norm: u={pick_tc.get('u')}, v={pick_tc.get('v')}")
print(f"    rationale: {cot_res.get('step2_rationale','')}")
print(f"  PLACE: [{place_ref_idx}번] {cot_res.get('step3_place_ref_description','')}")
print(f"    rationale: {cot_res.get('step3_rationale','')}")
print(f"         관계: {cot_res.get('step4_place_relation','')}")
print(f"         crop norm: u={place_tc.get('u')}, v={place_tc.get('v')}")
print(f"    rationale: {cot_res.get('step4_rationale','')}")
print(f"  ACTIONS: {actions}")
print(f"    rationale: {cot_res.get('step5_rationale','')}")
print(f"  conf={cot_res.get('confidence')}  ({t2:.1f}s)")

# ── crop 좌표 → 원본 좌표 변환 함수 ──────────────────────────
def crop_to_original(mask_data, crop_pil, u_norm, v_norm, padding=12):
    bx, by, bw, bh = [int(v) for v in mask_data["bbox"]]
    cx1 = max(0, bx - padding)
    cy1 = max(0, by - padding)
    ox = cx1 + int(u_norm * crop_pil.width)
    oy = cy1 + int(v_norm * crop_pil.height)
    return (ox, oy)

def depth_lookup(depth_np, px, py, H, W, r=5):
    if depth_np is None:
        return None, "N/A"
    patch = depth_np[max(0,py-r):min(H,py+r), max(0,px-r):min(W,px+r)]
    valid = patch[patch > 0]
    if len(valid) > 0:
        mm = float(np.median(valid))
        return mm, f"{mm:.0f}mm"
    return None, "N/A"

# PICK EE — depth 기반 윗면 centroid (VLM 없이 정확하게)
def compute_top_surface_ee(mask_data, depth_np):
    """
    SAM2 mask 내에서 두 점의 중점으로 윗면 중심(EE)을 계산.

    Point A: depth가 가장 작은 점 (카메라에 가장 가까운 = 윗면 앞쪽 가장자리)
    Point B: y 픽셀이 가장 작은 점 (이미지 최상단 = 윗면 뒤쪽 가장자리)
    EE = (A + B) / 2 = 윗면 중심

    파라미터 없음, 기하학적으로 정확.
    """
    mask_seg = mask_data["segmentation"]
    obj_bbox = [int(v) for v in mask_data["bbox"]]

    ys, xs = np.where(mask_seg)
    if len(ys) == 0:
        return None, None, None, None, None
    depths = depth_np[ys, xs]
    valid = depths > 0
    if valid.sum() == 0:
        return None, None, None, None, None
    ys, xs, depths = ys[valid], xs[valid], depths[valid]

    # Point A: depth 최소점 (상위 5% centroid로 안정화)
    d_min = depths.min()
    d_range = depths.max() - d_min
    near_th = d_min + max(d_range * 0.05, 3)
    near = depths <= near_th
    pt_a = (int(np.mean(xs[near])), int(np.mean(ys[near])))

    # Point B: y 최소점 (상위 5% centroid로 안정화)
    y_min = ys.min()
    y_th = y_min + max((ys.max() - y_min) * 0.05, 3)
    top_y = ys <= y_th
    pt_b = (int(np.mean(xs[top_y])), int(np.mean(ys[top_y])))

    # EE = 중점
    ee_x = (pt_a[0] + pt_b[0]) // 2
    ee_y = (pt_a[1] + pt_b[1]) // 2
    depth_mm = float(depths[near].mean())

    return (ee_x, ee_y), depth_mm, pt_a, pt_b, obj_bbox, {
        "d_min": float(depths.min()),
        "d_max": float(depths.max()),
        "pt_a_depth": pt_a,
        "pt_b_y_min": pt_b,
    }

print(f"\n  [Step 2b] 윗면 EE = midpoint(depth최소점, y최소점)...")
pick_ee_result = compute_top_surface_ee(pick_mask, depth_np)
top_result = None
pt_a = pt_b = None
obj_bbox_xywh = None

if pick_ee_result[0] is not None:
    pick_ee, pick_depth_mm, pt_a, pt_b, obj_bbox_xywh, top_info = pick_ee_result
    pick_depth_str = f"{pick_depth_mm:.0f}mm"
    top_result = top_info

    ob_x, ob_y, ob_w, ob_h = obj_bbox_xywh
    print(f"    SAM2 bbox: ({ob_x},{ob_y},{ob_w},{ob_h})")
    print(f"    Point A (depth 최소): {pt_a}")
    print(f"    Point B (y 최소):     {pt_b}")
    print(f"    EE = midpoint:        {pick_ee}  depth: {pick_depth_str}")
else:
    pick_u = float(pick_tc.get("u", 0.5))
    pick_v = float(pick_tc.get("v", 0.5))
    pick_ee = crop_to_original(pick_mask, pick_crop, pick_u, pick_v)
    pick_depth_mm, pick_depth_str = depth_lookup(depth_np, *pick_ee, H, W)
    print(f"    [Fallback] depth 없음 → VLM crop norm")

# 시각화: Point A + Point B + EE 중점
if pt_a and pt_b and obj_bbox_xywh:
    ob_x, ob_y, ob_w, ob_h = obj_bbox_xywh
    pad_vis = 30
    vx1 = max(0, ob_x - pad_vis)
    vy1 = max(0, ob_y - pad_vis)
    vx2 = min(W, ob_x + ob_w + pad_vis)
    vy2 = min(H, ob_y + ob_h + pad_vis)
    vis_crop = Image.fromarray(rgb_np[vy1:vy2, vx1:vx2]).convert("RGBA")
    d_vc = ImageDraw.Draw(vis_crop)

    # SAM2 bbox (파란)
    d_vc.rectangle([ob_x-vx1, ob_y-vy1, ob_x+ob_w-vx1, ob_y+ob_h-vy1],
                   outline=(100,180,255,200), width=2)
    d_vc.text((ob_x-vx1+3, ob_y-vy1-18), "SAM2 bbox", fill=(100,180,255,255), font=FONT_SM)

    # Point A (초록) — depth 최소
    ax, ay = pt_a[0]-vx1, pt_a[1]-vy1
    d_vc.ellipse([ax-6, ay-6, ax+6, ay+6], fill=(0,255,0,255))
    d_vc.text((ax+8, ay-8), "A: depth min", fill=(0,255,0,255), font=FONT_SM)

    # Point B (파란) — y 최소
    bx, by = pt_b[0]-vx1, pt_b[1]-vy1
    d_vc.ellipse([bx-6, by-6, bx+6, by+6], fill=(0,200,255,255))
    d_vc.text((bx+8, by+8), "B: y min", fill=(0,200,255,255), font=FONT_SM)

    # A-B 연결선 (점선 느낌)
    d_vc.line([(ax,ay),(bx,by)], fill=(200,200,200,150), width=1)

    # EE (빨간 십자) = 중점
    lcx = pick_ee[0] - vx1
    lcy = pick_ee[1] - vy1
    d_vc.ellipse([lcx-10, lcy-10, lcx+10, lcy+10], outline=(255,50,50,255), width=3)
    d_vc.line([(lcx-18,lcy),(lcx+18,lcy)], fill=(255,50,50,255), width=2)
    d_vc.line([(lcx,lcy-18),(lcx,lcy+18)], fill=(255,50,50,255), width=2)
    d_vc.text((lcx+12, lcy-12), "EE (midpoint)", fill=(255,100,100,255), font=FONT_SM)

    img_top_crop_vis = vis_crop.convert("RGB")
else:
    img_top_crop_vis = pick_crop.copy()

# PLACE EE
place_ee = None
place_depth_str = "N/A"
if place_mask is not None and place_crop is not None:
    place_u = float(place_tc.get("u", 0.5))
    place_v = float(place_tc.get("v", 0.5))
    place_ee = crop_to_original(place_mask, place_crop, place_u, place_v)
    _, place_depth_str = depth_lookup(depth_np, *place_ee, H, W)

print(f"\n[결과]")
print(f"  PICK  EE: {pick_ee}  depth: {pick_depth_str}")
print(f"  PLACE EE: {place_ee}  depth: {place_depth_str}")

# ── 시각화: pick & place 표시 ────────────────────────────────
# SAM2 오버레이 (pick=초록, place=파란)
img_s1_target = draw_sam2_overlay(rgb_np, masks)
img_pick_place = Image.fromarray(rgb_np).convert("RGBA")
H_img, W_img = rgb_np.shape[:2]
# pick mask 초록
lyr = np.zeros((H_img, W_img, 4), dtype=np.uint8)
lyr[pick_mask["segmentation"]] = [0, 220, 80, 180]
img_pick_place = Image.alpha_composite(img_pick_place, Image.fromarray(lyr))
# place mask 파란
if place_mask is not None:
    lyr2 = np.zeros((H_img, W_img, 4), dtype=np.uint8)
    lyr2[place_mask["segmentation"]] = [80, 150, 255, 160]
    img_pick_place = Image.alpha_composite(img_pick_place, Image.fromarray(lyr2))
d_pp = ImageDraw.Draw(img_pick_place)
# pick 크로스헤어 (빨간)
cx, cy = pick_ee
d_pp.ellipse([cx-16,cy-16,cx+16,cy+16], outline=(255,50,50,255), width=4)
d_pp.line([(cx-30,cy),(cx+30,cy)], fill=(255,50,50,255), width=3)
d_pp.line([(cx,cy-30),(cx,cy+30)], fill=(255,50,50,255), width=3)
d_pp.text((cx+20, cy-20), f"PICK ({cx},{cy})", fill=(255,80,80,255), font=FONT_MD)
# place 크로스헤어 (파란)
if place_ee:
    px, py = place_ee
    d_pp.ellipse([px-16,py-16,px+16,py+16], outline=(80,150,255,255), width=4)
    d_pp.line([(px-30,py),(px+30,py)], fill=(80,150,255,255), width=3)
    d_pp.line([(px,py-30),(px,py+30)], fill=(80,150,255,255), width=3)
    d_pp.text((px+20, py-20), f"PLACE ({px},{py})", fill=(120,180,255,255), font=FONT_MD)
    # pick→place 화살표 (점선)
    d_pp.line([(cx,cy),(px,py)], fill=(255,255,0,150), width=2)
img_pick_place = img_pick_place.convert("RGB")

# crop에 CoT 표시
img_crop_pick = draw_cot_on_crop(pick_crop, {"step4_top_center_crop": pick_tc})
img_s3 = img_pick_place

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

{card(f"Step 2 · GPT-4o CoT — Pick & Place 추론", img_s1_target,
    badges=[("PICK", f"[{pick_idx}번]", "#1a7a1a"),
            ("PLACE ref", f"[{place_ref_idx}번]" if place_ref_idx is not None else "없음", "#2c5f8a"),
            ("Conf", cot_res.get('confidence',''), "#7d6608"),
            ("시간", f"{t2:.1f}s", "#444")],
    rows=[("Step1 PICK 대상", f"[{pick_idx}번] {cot_res.get('step1_pick_description','')}"),
          ("  ↳ rationale", cot_res.get('step1_rationale','')),
          ("Step2 PICK 윗면", f"{cot_res.get('step2_pick_surface','')}"),
          ("Step2 PICK 중심", f"crop norm ({pick_tc.get('u','')}, {pick_tc.get('v','')})"),
          ("  ↳ rationale", cot_res.get('step2_rationale','')),
          ("Step3 PLACE 참조", f"[{place_ref_idx}번] {cot_res.get('step3_place_ref_description','')}" if place_ref_idx is not None else "없음"),
          ("  ↳ rationale", cot_res.get('step3_rationale','')),
          ("Step4 PLACE 관계", cot_res.get('step4_place_relation','')),
          ("Step4 PLACE 위치", f"crop norm ({place_tc.get('u','')}, {place_tc.get('v','')})"),
          ("  ↳ rationale", cot_res.get('step4_rationale',''))],
    note="초록 마스크 = PICK 대상 / GPT-4o가 crop에서 윗면 중심 추론",
    color="#145a32")}

<div class="arrow">↓</div>

{card("Step 2b · EE = midpoint(depth최소, y최소)", img_top_crop_vis,
    badges=[("EE pixel", f"{pick_ee}", "#5b2c6f"),
            ("depth", pick_depth_str, "#5b2c6f"),
            ("방식", "기하학적 중점", "#2e4057")],
    rows=[("Point A (depth 최소)", f"{pt_a} — 윗면 앞쪽 (카메라에 가장 가까운)" if pt_a else "N/A"),
          ("Point B (y 최소)", f"{pt_b} — 윗면 뒤쪽 (이미지 최상단)" if pt_b else "N/A"),
          ("EE = (A+B)/2", f"{pick_ee}"),
          ("depth", pick_depth_str),
          ("원리", "45도 카메라: A=앞쪽 가장자리, B=뒤쪽 가장자리 → 중점=윗면 중심")],
    note="초록=A(depth min) / 파란=B(y min) / 빨간=EE(중점) — 파라미터 없이 기하학적으로 정확",
    color="#3d1a5c")}

<div class="arrow">↓</div>

{card("Step 3 · Pick & Place EE 좌표", img_s3,
    badges=[("PICK", f"{pick_ee}", "#c0392b"),
            ("PLACE", f"{place_ee}" if place_ee else "N/A", "#2980b9"),
            ("depth", pick_depth_str, "#5b2c6f")],
    rows=[("PICK EE pixel", f"{pick_ee}"),
          ("PICK 방식", "depth 기반 윗면 centroid (상위 15%)" if top_result else "VLM crop norm (fallback)"),
          ("PICK depth", pick_depth_str),
          ("PLACE EE pixel", f"{place_ee}" if place_ee else "N/A"),
          ("PLACE depth", place_depth_str),
          ("관계", cot_res.get('step4_place_relation',''))],
    note="빨간 십자=PICK 위치 (depth 윗면 중심) / 파란 십자=PLACE 위치 / 노란 선=이동 경로",
    color="#3d1a5c")}

<div class="arrow">↓</div>

{card("Step 4 · Action Sequence", img_s3,
    badges=[("actions", f"{len(actions)}개", "#8e44ad")],
    rows=[(f"Action {i+1}", a) for i, a in enumerate(actions)]
         + [("rationale", cot_res.get('step5_rationale',''))],
    note=f"GPT-4o가 생성한 pick-and-place action sequence",
    color="#4a235a")}

</div>

<div class="final">
  <div class="final-label">Pick & Place Plan</div>
  <div class="final-detail">
    <b>지시:</b> {INSTRUCTION}<br><br>
    <b style="color:#e74c3c">PICK</b> [{pick_idx}번] {cot_res.get('step1_pick_description','')}<br>
    &nbsp;&nbsp;EE → <b>{pick_ee}</b> &nbsp;|&nbsp; depth <b>{pick_depth_str}</b><br><br>
    <b style="color:#3498db">PLACE</b> [{place_ref_idx}번] {cot_res.get('step3_place_ref_description','') if place_ref_idx is not None else ''}<br>
    &nbsp;&nbsp;EE → <b>{place_ee if place_ee else 'N/A'}</b> &nbsp;|&nbsp; depth <b>{place_depth_str}</b><br>
    &nbsp;&nbsp;관계: <b>{cot_res.get('step4_place_relation','')}</b><br><br>
    <b style="color:#9b59b6">Actions:</b> {' → '.join(actions)}<br>
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
