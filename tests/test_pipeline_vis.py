"""
SAM2 → Gemini ER 파이프라인 시각화 (최적화 v2)

개선 사항:
1. SAM2 mask 결과 캐시
2. crop 필터링: area + 노란 HSV 점수로 top-5
3. 식별(Step3): Flash 순차 → reasoning(Step4): ER 병렬
   Flash는 rate limit 넉넉, ER은 parallel 충돌 방지
"""
import sys, json, time, numpy as np, cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor, as_completed
import random, torch, pickle

sys.path.append(str(Path(__file__).parent.parent))

ROOT      = Path(__file__).parent.parent
out_dir   = ROOT / "results" / "pipeline_vis"
cache_dir = ROOT / "results" / "sam2_cache"
out_dir.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)

try:
    font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    font_md = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
except:
    font_sm = font_md = ImageFont.load_default()

# ── Gemini ──────────────────────────────────────────────────
from google import genai
from google.genai import types
key          = next(l.strip() for l in (ROOT/"token").read_text().splitlines() if l.strip().startswith("AIza"))
client       = genai.Client(api_key=key)
ID_MODEL     = "gemini-2.0-flash"               # Step3: 타겟 식별 (빠름)
REASON_MODEL = "gemini-robotics-er-1.5-preview"  # Step4: reasoning (정밀)

def gemini_call(contents, model, retries=3):
    for attempt in range(retries):
        try:
            return client.models.generate_content(
                model=model, contents=contents,
                config=types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json"),
            )
        except Exception as e:
            wait = 12 if "429" in str(e) else 8
            print(f"    retry {attempt+1}/{retries} ({wait}s) [{model.split('/')[-1]}]")
            time.sleep(wait)
    return None

def parse(resp):
    if not resp: return {}
    t = resp.text.strip()
    s, e = t.find("{"), t.rfind("}")+1
    return json.loads(t[s:e]) if s >= 0 and e > 0 else {}

# ── SAM2 ────────────────────────────────────────────────────
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import warnings; warnings.filterwarnings("ignore")

CKPT   = ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"SAM2 로딩... (device={DEVICE})")
_sam2 = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", str(CKPT), device=DEVICE)
generator = SAM2AutomaticMaskGenerator(
    model=_sam2, points_per_side=16,
    pred_iou_thresh=0.80, stability_score_thresh=0.90, min_mask_region_area=800,
)
print("준비 완료\n")

# ── 유틸 ────────────────────────────────────────────────────
YLW_LO = np.array([18, 80, 80])
YLW_HI = np.array([38, 255, 255])

def yellow_score(crop_pil):
    arr = np.array(crop_pil.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, YLW_LO, YLW_HI)
    return float(mask.sum()) / (arr.shape[0]*arr.shape[1]*255 + 1e-6)

def make_crop(img_np, md, pad=12):
    mask = md["segmentation"].astype(bool)
    h, w = img_np.shape[:2]
    x, y, bw, bh = [int(v) for v in md["bbox"]]
    x1,y1 = max(0,x-pad), max(0,y-pad)
    x2,y2 = min(w,x+bw+pad), min(h,y+bh+pad)
    arr = np.array(Image.fromarray(img_np).convert("RGBA"))
    arr[~mask] = [255,255,255,255]
    return Image.fromarray(arr).convert("RGB").crop((x1,y1,x2,y2))

def filter_crops(masks, crops, top_n=5):
    max_area = max(m["area"] for m in masks) + 1
    scored = [(i, m, c, m["area"]/max_area + yellow_score(c)*3.0)
              for i, (m, c) in enumerate(zip(masks, crops))]
    return sorted(scored, key=lambda x: x[3], reverse=True)[:top_n]

def add_panel(img, text, bg=(30,30,30)):
    bar = Image.new("RGB", (img.width, 40), bg)
    ImageDraw.Draw(bar).text((10,10), text, fill=(255,255,255), font=font_md)
    out = Image.new("RGB", (img.width, img.height+40))
    out.paste(bar,(0,0)); out.paste(img,(0,40))
    return out

def get_masks_cached(scene, img_np):
    p = cache_dir / f"{scene}_masks.pkl"
    if p.exists():
        print(f"    [캐시] {scene}")
        return pickle.loads(p.read_bytes())
    t0 = time.time()
    masks = sorted(generator.generate(img_np), key=lambda x: x["area"], reverse=True)
    p.write_bytes(pickle.dumps(masks))
    print(f"    SAM2 {len(masks)}개 ({time.time()-t0:.1f}s) → 캐시")
    return masks

# ── Step1+2: SAM2 + crop 준비 ────────────────────────────────
INSTRUCTION = "노란 원기둥을 집어라"
SCENES      = ["base10", "base11", "base12"]
random.seed(42)

print("=" * 55)
print(f"  파이프라인: Flash(식별) + ER(reasoning) + 캐시")
print("=" * 55)

t_start  = time.time()
all_data = {}   # scene → {img_np, masks, crops, filtered}

for scene in SCENES:
    img_np = np.array(Image.open(ROOT/"data"/"base_images"/f"{scene}.jpg").convert("RGB"))
    masks  = get_masks_cached(scene, img_np)
    crops  = [make_crop(img_np, m) for m in masks]
    filtered = filter_crops(masks, crops, top_n=5)
    all_data[scene] = {"img_np": img_np, "masks": masks, "crops": crops, "filtered": filtered}

# ── Step3: Flash로 순차 식별 ─────────────────────────────────
print("\n[Step3] Flash 순차 식별...")
for scene in SCENES:
    d = all_data[scene]
    filtered = d["filtered"]
    contents = [
        f"지시: {INSTRUCTION}\n\n"
        f"아래 {len(filtered)}개 이미지는 씬에서 분리된 각 객체입니다.\n"
        f"지시에서 언급된 물체의 번호(0-based)를 답하세요.\n"
        f'JSON으로만: {{"target_index": 0, "confidence": 0.90, "reason": "이유"}}'
    ]
    idx_map = {}
    for pos, (orig_i, m, cr, _) in enumerate(filtered):
        contents += [f"[{pos}번]", cr]
        idx_map[pos] = orig_i

    t1   = time.time()
    resp = gemini_call(contents, model=ID_MODEL)
    res  = parse(resp)
    pos  = min(int(res.get("target_index", 0)), len(filtered)-1)
    idx  = idx_map.get(pos, filtered[0][0])
    conf = res.get("confidence", 0)
    reason = res.get("reason", "")
    id_t = time.time() - t1
    print(f"  [{scene}] → mask[{idx}]  conf={conf}  ({id_t:.1f}s)  {reason[:45]}")
    d["target_idx"]  = idx
    d["target_conf"] = conf
    d["target_reason"] = reason
    d["id_time"]     = id_t

# ── Step4: ER reasoning 병렬 ─────────────────────────────────
print("\n[Step4] ER reasoning 병렬...")

def run_reasoning_and_vis(scene):
    d        = all_data[scene]
    img_np   = d["img_np"]
    masks    = d["masks"]
    crops    = d["crops"]
    filtered = d["filtered"]
    idx      = d["target_idx"]
    conf     = d["target_conf"]
    reason   = d["target_reason"]
    id_t     = d["id_time"]
    img_pil  = Image.fromarray(img_np)
    H, W     = img_np.shape[:2]
    colors   = [(random.randint(60,230), random.randint(60,230), random.randint(60,230)) for _ in masks]

    # Step1 시각화
    ov = img_pil.convert("RGBA")
    for md, col in zip(masks, colors):
        seg = md["segmentation"].astype(bool)
        lyr = np.zeros((H,W,4), dtype=np.uint8); lyr[seg] = [*col, 110]
        ov  = Image.alpha_composite(ov, Image.fromarray(lyr))
        x,y,bw,bh = [int(v) for v in md["bbox"]]
        ImageDraw.Draw(ov).rectangle([x,y,x+bw,y+bh], outline=(*col,255), width=2)
    s1 = add_panel(ov.convert("RGB"), f"Step1: SAM2 — {len(masks)}개 mask (device={DEVICE})", (40,40,120))
    s1.save(out_dir/f"{scene}_step1_sam2.jpg", quality=92)

    # Step2 시각화 (crop grid)
    THUMB = 140
    grid = Image.new("RGB", (len(filtered)*(THUMB+6), THUMB+30), (230,230,230))
    for pos, (orig_i, m, cr, score) in enumerate(filtered):
        thumb = cr.copy(); thumb.thumbnail((THUMB,THUMB))
        xo = pos*(THUMB+6)+(THUMB-thumb.width)//2
        grid.paste(thumb,(xo,0))
        clr = (0,180,0) if orig_i == idx else (60,60,60)
        ImageDraw.Draw(grid).text((pos*(THUMB+6)+4,THUMB+4), f"[{pos}] s={score:.2f}", fill=clr, font=font_sm)
    s2 = add_panel(grid, f"Step2: top-5 crops (노란 점수 기준) → Flash 입력", (40,100,40))
    s2.save(out_dir/f"{scene}_step2_crops.jpg", quality=92)

    # Step3 시각화
    ov2 = img_pil.convert("RGBA")
    seg = masks[idx]["segmentation"].astype(bool)
    lyr = np.zeros((H,W,4), dtype=np.uint8); lyr[seg] = [0,220,80,170]
    ov2 = Image.alpha_composite(ov2, Image.fromarray(lyr))
    x,y,bw,bh = [int(v) for v in masks[idx]["bbox"]]
    d2 = ImageDraw.Draw(ov2)
    d2.rectangle([x,y,x+bw,y+bh], outline=(0,255,80,255), width=4)
    d2.rectangle([x,y-30,x+bw,y], fill=(0,160,50,220))
    d2.text((x+4,y-26), f"TARGET [{idx}] conf={conf}", fill=(255,255,255,255), font=font_md)
    s3 = add_panel(ov2.convert("RGB"),
                   f"Step3: Flash 식별 → [{idx}] {reason[:42]}  ({id_t:.1f}s)", (20,100,20))
    s3.save(out_dir/f"{scene}_step3_target.jpg", quality=92)

    # Step4: ER reasoning
    target_crop = crops[idx]
    prompt = (
        f"지시: {INSTRUCTION}\n\n"
        "이미지는 배경이 제거된 타겟 객체(노란 원기둥)입니다.\n"
        "그리퍼가 이 객체를 잡을 수 있는 상태인지 판단하세요.\n"
        'JSON으로만: {"graspable": true, "confidence": 0.90, '
        '"alignment": "정렬됨/약간 틀어짐/많이 틀어짐", "reason": "판단 근거"}'
    )
    t2   = time.time()
    resp2 = gemini_call([prompt, target_crop], model=REASON_MODEL)
    res2  = parse(resp2)
    reas_t = time.time()-t2
    print(f"  [{scene}] ER → graspable={res2.get('graspable')} conf={res2.get('confidence')} ({reas_t:.1f}s)")

    # Step4 시각화
    cw, ch = target_crop.size
    scale  = min(280/max(cw,1), 280/max(ch,1), 1.0)
    crop_big = target_crop.resize((int(cw*scale), int(ch*scale)), Image.LANCZOS)
    g_color = (20,160,20) if res2.get("graspable") else (180,20,20)
    g_text  = "GRASPABLE ✓" if res2.get("graspable") else "NOT GRASPABLE ✗"
    s4 = Image.new("RGB", (max(crop_big.width,480), crop_big.height+110), (30,30,30))
    s4.paste(crop_big, ((s4.width-crop_big.width)//2, 70))
    d4 = ImageDraw.Draw(s4)
    d4.rectangle([0,0,s4.width,40],  fill=(80,30,30))
    d4.text((10,8),  "Step4: clean crop → ER reasoning", fill=(255,255,255), font=font_md)
    d4.rectangle([0,40,s4.width,70], fill=g_color)
    d4.text((10,46), f"{g_text}  conf={res2.get('confidence')}  {res2.get('alignment','')}", fill=(255,255,255), font=font_md)
    d4.text((10,crop_big.height+75), res2.get("reason","")[:80], fill=(180,180,180), font=font_sm)
    s4.save(out_dir/f"{scene}_step4_reasoning.jpg", quality=92)

    # 합본
    TW = 640
    resized = [im.resize((TW, int(im.height*TW/im.width)), Image.LANCZOS) for im in [s1,s2,s3,s4]]
    canvas  = Image.new("RGB", (TW, sum(r.height for r in resized)), (15,15,15))
    yy = 0
    for r in resized: canvas.paste(r,(0,yy)); yy += r.height
    canvas.save(out_dir/f"{scene}_pipeline_all.jpg", quality=92)

    return {"scene": scene, "graspable": res2.get("graspable"), "reas_t": round(reas_t,1)}

with ThreadPoolExecutor(max_workers=3) as ex:
    futures = {ex.submit(run_reasoning_and_vis, s): s for s in SCENES}
    results = {}
    for f in as_completed(futures):
        r = f.result()
        results[r["scene"]] = r

print(f"\n{'='*55}")
print(f"  전체 소요: {time.time()-t_start:.1f}s")
for s in SCENES:
    r = results.get(s, {})
    print(f"  {s}: graspable={r.get('graspable')}  reasoning={r.get('reas_t')}s")
print(f"  저장: {out_dir}")
