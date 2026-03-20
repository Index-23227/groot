"""
파이프라인 단계별 시각화
Step 1: SAM2 — 모든 mask 오버레이 (API 불필요)
Step 2: Gemini 타겟 식별 결과 (기존 결과 재사용)
Step 3: clean crop
Step 4: reasoning 결과 텍스트 오버레이
"""
import sys, json, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

sys.path.append(str(Path(__file__).parent.parent))

ROOT    = Path(__file__).parent.parent
out_dir = ROOT / "results" / "pipeline_vis"
out_dir.mkdir(parents=True, exist_ok=True)

try:
    font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    font_md = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
except:
    font_sm = font_md = font_lg = ImageFont.load_default()

def add_label(img: Image.Image, text: str, color=(20,20,20)) -> Image.Image:
    h = 36
    bar = Image.new("RGB", (img.width, h), color)
    d = ImageDraw.Draw(bar)
    d.text((10, 8), text, fill=(255,255,255), font=font_md)
    out = Image.new("RGB", (img.width, img.height + h))
    out.paste(bar, (0, 0))
    out.paste(img, (0, h))
    return out

# ── Step 1: SAM2 실행 ────────────────────────────────────────
print("=" * 55)
print("  Step 1: SAM2 blind segmentation")
print("=" * 55)

import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

CHECKPOINT = ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", str(CHECKPOINT), device=DEVICE)
generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=16,
    pred_iou_thresh=0.80,
    stability_score_thresh=0.90,
    min_mask_region_area=800,
)

img_path = ROOT / "data" / "base_images" / "base1.jpg"
image_np = np.array(Image.open(img_path).convert("RGB"))
img_pil  = Image.fromarray(image_np)
img_w, img_h = img_pil.size

import time
t0 = time.time()
masks = generator.generate(image_np)
masks = sorted(masks, key=lambda x: x["area"], reverse=True)
sam_time = time.time() - t0
print(f"  {len(masks)}개 mask 생성  ({sam_time:.2f}s)")

# SAM2 전체 mask 오버레이 시각화
random.seed(42)
colors = [(random.randint(50,255), random.randint(50,255), random.randint(50,255)) for _ in masks]

overlay = img_pil.convert("RGBA").copy()
for i, (mask_data, color) in enumerate(zip(masks, colors)):
    seg = mask_data["segmentation"].astype(bool)
    layer = Image.new("RGBA", img_pil.size, (0,0,0,0))
    arr = np.array(layer)
    arr[seg] = [*color, 120]
    overlay = Image.alpha_composite(overlay, Image.fromarray(arr.astype(np.uint8)))
    # bbox 번호 표시
    x, y, bw, bh = [int(v) for v in mask_data["bbox"]]
    d = ImageDraw.Draw(overlay)
    d.rectangle([x, y, x+bw, y+bh], outline=(*color, 255), width=2)
    d.text((x+3, y+2), str(i), fill=(255,255,255,255), font=font_sm)

step1_img = overlay.convert("RGB")
step1_img = add_label(step1_img, f"Step 1: SAM2 blind segmentation — {len(masks)}개 mask ({sam_time:.2f}s)", (40,40,120))
step1_img.save(out_dir / "step1_sam2_all_masks.jpg", quality=92)
print(f"  저장: step1_sam2_all_masks.jpg")

# ── Step 2: 각 crop 시각화 ───────────────────────────────────
print("\n  Step 2-prep: 각 mask crop 생성")

def make_crop(image_np, mask_data, padding=12):
    mask = mask_data["segmentation"].astype(bool)
    h, w = image_np.shape[:2]
    x, y, bw, bh = [int(v) for v in mask_data["bbox"]]
    x1, y1 = max(0, x-padding), max(0, y-padding)
    x2, y2 = min(w, x+bw+padding), min(h, y+bh+padding)
    arr = np.array(Image.fromarray(image_np).convert("RGBA"))
    arr[~mask] = [255, 255, 255, 255]
    return Image.fromarray(arr).convert("RGB").crop((x1, y1, x2, y2))

crops = [make_crop(image_np, m) for m in masks]

# crop 그리드 시각화 (최대 12개)
THUMB = 120
n_show = min(len(crops), 12)
cols = 4
rows = (n_show + cols - 1) // cols
grid = Image.new("RGB", (cols * (THUMB+4), rows * (THUMB+24)), (240,240,240))

for i in range(n_show):
    crop = crops[i].copy()
    crop.thumbnail((THUMB, THUMB))
    col, row = i % cols, i // cols
    x_off = col * (THUMB + 4) + (THUMB - crop.width) // 2
    y_off = row * (THUMB + 24)
    grid.paste(crop, (x_off, y_off))
    d = ImageDraw.Draw(grid)
    d.text((col*(THUMB+4)+4, row*(THUMB+24)+THUMB+4), f"[{i}] area={masks[i]['area']}", fill=(60,60,60), font=font_sm)

grid = add_label(grid, f"Step 2: SAM2 crops (총 {len(crops)}개) — Gemini가 타겟 선택", (40,100,40))
grid.save(out_dir / "step2_crops_grid.jpg", quality=92)
print(f"  저장: step2_crops_grid.jpg")

# ── Step 3: 타겟 highlight (기존 결과: index=6) ──────────────
TARGET_IDX = 6  # 이전 실행에서 Gemini가 선택한 인덱스
print(f"\n  Step 3: 타겟 [idx={TARGET_IDX}] highlight")

step3_img = img_pil.convert("RGBA").copy()
if TARGET_IDX < len(masks):
    seg = masks[TARGET_IDX]["segmentation"].astype(bool)
    layer = Image.new("RGBA", img_pil.size, (0,0,0,0))
    arr = np.array(layer)
    arr[seg] = [0, 220, 80, 160]
    step3_img = Image.alpha_composite(step3_img, Image.fromarray(arr.astype(np.uint8)))
    x, y, bw, bh = [int(v) for v in masks[TARGET_IDX]["bbox"]]
    d = ImageDraw.Draw(step3_img)
    d.rectangle([x, y, x+bw, y+bh], outline=(0,255,80,255), width=4)
    d.text((x, y-22), f"TARGET [{TARGET_IDX}] sprite_can", fill=(0,255,80,255), font=font_md)

step3_img = step3_img.convert("RGB")
step3_img = add_label(step3_img, "Step 3: Gemini 타겟 식별 → sprite can [idx=6] 선택", (20,100,20))
step3_img.save(out_dir / "step3_target_identified.jpg", quality=92)
print(f"  저장: step3_target_identified.jpg")

# ── Step 4: clean crop ───────────────────────────────────────
print("\n  Step 4: clean crop")
if TARGET_IDX < len(crops):
    target_crop = crops[TARGET_IDX]
    crop_labeled = add_label(target_crop, "Step 4: clean crop (배경 제거) → Gemini reasoning 입력", (100,40,40))
    crop_labeled.save(out_dir / "step4_clean_crop.jpg", quality=92)
    print(f"  저장: step4_clean_crop.jpg  ({target_crop.width}x{target_crop.height}px)")

# ── 전체 요약 ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"  저장 완료: {out_dir}")
print("  step1_sam2_all_masks.jpg   ← SAM2 전체 mask")
print("  step2_crops_grid.jpg       ← 각 객체 crop 그리드")
print("  step3_target_identified.jpg ← Gemini 선택 결과")
print("  step4_clean_crop.jpg       ← reasoning 입력 이미지")
