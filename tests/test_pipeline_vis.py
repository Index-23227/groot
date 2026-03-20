"""
SAM2 → Gemini ER 전체 파이프라인 시각화
base10, base11, base12 / 노란 원기둥 pick & place task
Step 1: SAM2 all masks
Step 2: Gemini 타겟 식별 → crop grid (타겟 하이라이트)
Step 3: target mask overlay on original
Step 4: clean crop + Gemini reasoning 결과
"""
import sys, json, time, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random, torch

sys.path.append(str(Path(__file__).parent.parent))

ROOT    = Path(__file__).parent.parent
out_dir = ROOT / "results" / "pipeline_vis"
out_dir.mkdir(parents=True, exist_ok=True)

try:
    font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    font_md = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
except:
    font_sm = font_md = font_lg = ImageFont.load_default()

# ── Gemini 클라이언트 ────────────────────────────────────────
from google import genai
from google.genai import types
token = ROOT / "token"
key   = next(l.strip() for l in token.read_text().splitlines() if l.strip().startswith("AIza"))
client = genai.Client(api_key=key)
MODEL  = "gemini-robotics-er-1.5-preview"

def gemini_call(contents, retries=3):
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=contents,
                config=types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json"),
            )
            return resp
        except Exception as e:
            wait = 15 if "429" in str(e) else 10
            print(f"  재시도 {attempt+1}/{retries} ({wait}s)... ({str(e)[:60]})")
            time.sleep(wait)
    return None

def parse(resp):
    if resp is None:
        return {}
    t = resp.text.strip()
    s, e = t.find("{"), t.rfind("}")+1
    if s < 0 or e <= 0:
        return {}
    return json.loads(t[s:e])

# ── SAM2 ────────────────────────────────────────────────────
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

CKPT   = ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sam2   = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", str(CKPT), device=DEVICE)
gen    = SAM2AutomaticMaskGenerator(
    model=sam2, points_per_side=16,
    pred_iou_thresh=0.80, stability_score_thresh=0.90, min_mask_region_area=800,
)

def make_crop(img_np, mask_data, pad=12):
    mask = mask_data["segmentation"].astype(bool)
    h, w = img_np.shape[:2]
    x, y, bw, bh = [int(v) for v in mask_data["bbox"]]
    x1,y1 = max(0,x-pad), max(0,y-pad)
    x2,y2 = min(w,x+bw+pad), min(h,y+bh+pad)
    arr = np.array(Image.fromarray(img_np).convert("RGBA"))
    arr[~mask] = [255,255,255,255]
    return Image.fromarray(arr).convert("RGB").crop((x1,y1,x2,y2))

def add_panel(img, text, bg=(30,30,30)):
    bar = Image.new("RGB", (img.width, 40), bg)
    ImageDraw.Draw(bar).text((10,10), text, fill=(255,255,255), font=font_md)
    out = Image.new("RGB", (img.width, img.height+40))
    out.paste(bar,(0,0)); out.paste(img,(0,40))
    return out

INSTRUCTION = "노란 원기둥을 집어라"
SCENES = ["base10", "base11", "base12"]

random.seed(42)

for scene in SCENES:
    print(f"\n{'='*55}")
    print(f"  {scene} — {INSTRUCTION}")
    print(f"{'='*55}")

    img_path = ROOT / "data" / "base_images" / f"{scene}.jpg"
    img_np   = np.array(Image.open(img_path).convert("RGB"))
    img_pil  = Image.fromarray(img_np)
    W, H     = img_pil.size

    # ── Step 1: SAM2 ────────────────────────────────────────
    print("[1] SAM2 segmentation...")
    t0 = time.time()
    masks  = gen.generate(img_np)
    masks  = sorted(masks, key=lambda x: x["area"], reverse=True)
    crops  = [make_crop(img_np, m) for m in masks]
    sam_t  = time.time() - t0
    print(f"    {len(masks)}개 mask  ({sam_t:.2f}s)")

    colors = [(random.randint(60,255), random.randint(60,255), random.randint(60,255)) for _ in masks]

    # Step 1 시각화: 모든 mask 오버레이
    ov = img_pil.convert("RGBA")
    for i, (md, col) in enumerate(zip(masks, colors)):
        seg = md["segmentation"].astype(bool)
        lyr = np.zeros((*img_np.shape[:2], 4), dtype=np.uint8)
        lyr[seg] = [*col, 110]
        ov = Image.alpha_composite(ov, Image.fromarray(lyr))
        x,y,bw,bh = [int(v) for v in md["bbox"]]
        d = ImageDraw.Draw(ov)
        d.rectangle([x,y,x+bw,y+bh], outline=(*col,255), width=2)
        d.text((x+3,y+2), str(i), fill=(255,255,255,255), font=font_sm)
    s1 = add_panel(ov.convert("RGB"),
                   f"Step 1: SAM2 blind segmentation — {len(masks)}개 mask ({sam_t:.2f}s)",
                   (40,40,120))
    s1.save(out_dir / f"{scene}_step1_sam2.jpg", quality=92)

    # ── Step 2: crop grid ───────────────────────────────────
    THUMB = 130
    n     = min(len(crops), 12)
    cols  = 4
    rows  = (n+cols-1)//cols
    grid  = Image.new("RGB", (cols*(THUMB+6), rows*(THUMB+28)), (230,230,230))
    for i in range(n):
        cr = crops[i].copy(); cr.thumbnail((THUMB,THUMB))
        cx,cy = i%cols, i//cols
        xo = cx*(THUMB+6)+(THUMB-cr.width)//2
        yo = cy*(THUMB+28)
        grid.paste(cr,(xo,yo))
        ImageDraw.Draw(grid).text((cx*(THUMB+6)+4, yo+THUMB+4),
                                  f"[{i}]", fill=(60,60,60), font=font_sm)
    s2 = add_panel(grid, f"Step 2: {len(crops)}개 crop → Gemini 타겟 식별 입력", (40,100,40))
    s2.save(out_dir / f"{scene}_step2_crops.jpg", quality=92)

    # ── Step 3: Gemini 타겟 식별 ────────────────────────────
    print("[2] Gemini 타겟 식별...")
    t1 = time.time()
    # 너무 작은 mask 제외, 최대 8개만 Gemini에 전송
    filtered = [(i, m, c) for i, (m, c) in enumerate(zip(masks, crops))
                if m["area"] > 1000][:8]
    contents = [
        f"지시: {INSTRUCTION}\n\n"
        f"아래 {len(filtered)}개 이미지는 씬에서 분리된 각 객체입니다.\n"
        f"지시에서 언급된 물체의 번호(0-based index)를 답하세요.\n"
        f'JSON으로만: {{"target_index": 0, "confidence": 0.90, "reason": "이유"}}'
    ]
    idx_map = {}
    for pos, (orig_i, m, cr) in enumerate(filtered):
        contents += [f"[{pos}번]", cr]
        idx_map[pos] = orig_i
    resp   = gemini_call(contents)
    result  = parse(resp)
    pos     = min(int(result.get("target_index", 0)), len(filtered)-1)
    idx     = idx_map.get(pos, 0)
    conf   = result.get("confidence", 0)
    reason = result.get("reason", "")
    gem_t  = time.time() - t1
    print(f"    타겟: [{idx}번]  conf={conf}  ({gem_t:.1f}s)")
    print(f"    근거: {reason[:60]}")

    # Step 3 시각화: 원본에 타겟 mask 초록 하이라이트
    ov2 = img_pil.convert("RGBA")
    seg = masks[idx]["segmentation"].astype(bool)
    lyr = np.zeros((*img_np.shape[:2], 4), dtype=np.uint8)
    lyr[seg] = [0, 220, 80, 170]
    ov2 = Image.alpha_composite(ov2, Image.fromarray(lyr))
    d2  = ImageDraw.Draw(ov2)
    x,y,bw,bh = [int(v) for v in masks[idx]["bbox"]]
    d2.rectangle([x,y,x+bw,y+bh], outline=(0,255,80,255), width=4)
    d2.rectangle([x,y-30,x+bw,y], fill=(0,180,60,220))
    d2.text((x+4,y-26), f"TARGET [{idx}] conf={conf}", fill=(255,255,255,255), font=font_md)
    s3 = add_panel(ov2.convert("RGB"),
                   f"Step 3: Gemini 선택 → [{idx}번] {reason[:45]}  ({gem_t:.1f}s)",
                   (20,100,20))
    s3.save(out_dir / f"{scene}_step3_target.jpg", quality=92)

    # ── Step 4: Gemini reasoning on clean crop ───────────────
    target_crop = crops[idx]
    print("[3] Gemini reasoning (graspability)...")
    t2 = time.time()
    prompt = (
        f"지시: {INSTRUCTION}\n\n"
        "이미지는 배경이 제거된 타겟 객체(노란 원기둥)입니다.\n"
        "그리퍼가 이 객체를 잡을 수 있는 상태인지 판단하세요.\n"
        'JSON으로만: {"graspable": true, "confidence": 0.90, '
        '"alignment": "정렬됨/약간 틀어짐/많이 틀어짐", "reason": "판단 근거"}'
    )
    resp2  = gemini_call([prompt, target_crop])
    res2   = parse(resp2)
    reas_t = time.time() - t2
    print(f"    graspable={res2.get('graspable')}  conf={res2.get('confidence')}  ({reas_t:.1f}s)")
    print(f"    {res2.get('reason','')[:70]}")

    # Step 4 시각화: clean crop + 결과 패널
    cw, ch = target_crop.size
    scale  = min(300/cw, 300/ch, 1.0)
    crop_big = target_crop.resize((int(cw*scale), int(ch*scale)), Image.LANCZOS)

    g_color = (20,160,20) if res2.get("graspable") else (180,20,20)
    g_text  = "GRASPABLE ✓" if res2.get("graspable") else "NOT GRASPABLE ✗"

    s4 = Image.new("RGB", (max(crop_big.width, 500), crop_big.height+110), (30,30,30))
    s4.paste(crop_big, ((s4.width-crop_big.width)//2, 70))
    d4 = ImageDraw.Draw(s4)
    d4.rectangle([0,0,s4.width,40], fill=(100,40,40))
    d4.text((10,8),  "Step 4: clean crop → Gemini reasoning", fill=(255,255,255), font=font_md)
    d4.rectangle([0,40,s4.width,70], fill=g_color)
    d4.text((10,46), f"{g_text}  conf={res2.get('confidence')}  align={res2.get('alignment','')}", fill=(255,255,255), font=font_md)
    d4.text((10, crop_big.height+75), res2.get("reason","")[:80], fill=(200,200,200), font=font_sm)
    s4.save(out_dir / f"{scene}_step4_reasoning.jpg", quality=92)

    # ── 4-step 합본 ─────────────────────────────────────────
    TW = 640
    imgs = [s1, s2, s3, s4]
    resized = []
    for im in imgs:
        r = TW / im.width
        resized.append(im.resize((TW, int(im.height*r)), Image.LANCZOS))

    total_h = sum(r.height for r in resized)
    composite = Image.new("RGB", (TW, total_h), (15,15,15))
    yy = 0
    for r in resized:
        composite.paste(r, (0, yy)); yy += r.height

    composite.save(out_dir / f"{scene}_pipeline_all.jpg", quality=92)
    print(f"  → {scene}_pipeline_all.jpg 저장 완료")

print(f"\n모든 시각화 저장: {out_dir}")
