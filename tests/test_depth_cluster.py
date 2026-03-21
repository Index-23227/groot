"""
Depth-only Clustering 테스트

RealSense depth 값만으로 테이블 위 객체 분리.
RGB는 시각화 / VLM 식별용으로만 사용.

알고리즘:
  1. 테이블 depth D 자동 추정 (ROI 내 median)
  2. D - max_h < pixel < D - min_h → 물체 마스크
  3. Morphological 노이즈 제거
  4. Connected components → 개별 물체
  5. RGB crop 추출 → VLM 식별
"""
import sys, io, base64, time
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).parent.parent))

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "results" / "depth_cluster"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INSTRUCTION = "초록 원기둥을 집어라"

# ── 튜닝 파라미터 ─────────────────────────────────────────────
MIN_H_MM   = 20    # 테이블 위 최소 높이 (노이즈 제거)
MAX_H_MM   = 350   # 최대 높이 (로봇팔 등 제외)
MIN_AREA   = 400   # 최소 객체 픽셀 수
PAD        = 12    # crop 패딩

try:
    FONT_SM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    FONT_MD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
except:
    FONT_SM = FONT_MD = ImageFont.load_default()

# ── RealSense 캡처 ────────────────────────────────────────────
import pyrealsense2 as rs

print("RealSense 연결 중...")
pipe    = rs.pipeline()
cfg     = rs.config()
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16,  30)
profile = pipe.start(cfg)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale  = depth_sensor.get_depth_scale()   # 보통 0.001 (mm 단위)
align = rs.align(rs.stream.color)

print("워밍업 중...")
for _ in range(30): pipe.wait_for_frames()
frames   = align.process(pipe.wait_for_frames())
color_f  = frames.get_color_frame()
depth_f  = frames.get_depth_frame()
pipe.stop()

rgb_np   = np.asanyarray(color_f.get_data())[:, :, ::-1]   # BGR→RGB
depth_np = np.asanyarray(depth_f.get_data())                # uint16, mm 단위
H, W     = rgb_np.shape[:2]

# 이미지 저장
img_path = ROOT / "data" / "base_images" / "base13.jpg"
Image.fromarray(rgb_np).save(img_path, quality=95)
print(f"RGB 저장: {img_path}")

# ── 테이블 depth 추정 ─────────────────────────────────────────
# 이미지 중앙 ROI에서 depth median → 테이블 평면
ROI_EST = (W//6, H//6, 5*W//6, 5*H//6)   # 중앙 2/3 영역
patch   = depth_np[ROI_EST[1]:ROI_EST[3], ROI_EST[0]:ROI_EST[2]]
valid   = patch[patch > 0]
table_d = float(np.median(valid))
print(f"\n테이블 depth 추정: {table_d:.1f}mm  ({table_d*depth_scale*1000:.1f}mm)")

# ── Depth Threshold → 물체 마스크 ────────────────────────────
t_seg = time.time()

obj_mask = (
    (depth_np > 0) &
    (depth_np < table_d - MIN_H_MM) &
    (depth_np > table_d - MAX_H_MM)
).astype(np.uint8) * 255

# Morphological: 노이즈 제거 → 구멍 채우기
k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN,  k_open)
obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, k_close)

# ── Connected Components ───────────────────────────────────────
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(obj_mask, 8)

objects = []
for i in range(1, num_labels):
    area = int(stats[i, cv2.CC_STAT_AREA])
    if area < MIN_AREA:
        continue
    x  = int(stats[i, cv2.CC_STAT_LEFT])
    y  = int(stats[i, cv2.CC_STAT_TOP])
    bw = int(stats[i, cv2.CC_STAT_WIDTH])
    bh = int(stats[i, cv2.CC_STAT_HEIGHT])
    cx = int(centroids[i][0])
    cy = int(centroids[i][1])
    comp_mask = (labels == i)

    # depth centroid (정확한 3D 중심)
    d_vals = depth_np[comp_mask]
    d_vals = d_vals[d_vals > 0]
    depth_mm   = float(np.median(d_vals)) if len(d_vals) else None
    height_mm  = (table_d - depth_mm) if depth_mm else None

    objects.append({
        "idx":         len(objects),
        "bbox":        (x, y, bw, bh),
        "mask":        comp_mask,
        "area":        area,
        "centroid_px": (cx, cy),
        "depth_mm":    depth_mm,
        "height_mm":   height_mm,
    })

objects.sort(key=lambda o: o["area"], reverse=True)
t_seg = time.time() - t_seg

print(f"감지: {len(objects)}개 객체  ({t_seg*1000:.1f}ms)")
for o in objects:
    print(f"  [{o['idx']}] area={o['area']}px  "
          f"centroid=({o['centroid_px'][0]},{o['centroid_px'][1]})  "
          f"depth={o['depth_mm']:.0f}mm  height={o['height_mm']:.0f}mm"
          if o['depth_mm'] else f"  [{o['idx']}] area={o['area']}px")

# ── RGB crop 추출 ─────────────────────────────────────────────
def make_crop(rgb, obj, pad=PAD):
    x, y, bw, bh = obj["bbox"]
    x1 = max(0, x-pad); y1 = max(0, y-pad)
    x2 = min(W, x+bw+pad); y2 = min(H, y+bh+pad)
    arr = np.array(Image.fromarray(rgb).convert("RGBA"))
    arr[~obj["mask"]] = [255, 255, 255, 255]
    return Image.fromarray(arr).convert("RGB").crop((x1, y1, x2, y2))

crops = [make_crop(rgb_np, o) for o in objects]

# ── 시각화 ────────────────────────────────────────────────────
import random; random.seed(5)

# 1) Depth map 컬러
d_clip  = np.clip(depth_np, 0, 2000).astype(np.float32)
d_norm  = (d_clip / 2000 * 255).astype(np.uint8)
d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
d_color = cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB)

# 2) 물체 마스크 오버레이 (RGB 위)
ov = Image.fromarray(rgb_np).convert("RGBA")
for o in objects:
    col = (random.randint(80,220), random.randint(80,220), random.randint(80,220))
    lyr = np.zeros((H, W, 4), dtype=np.uint8)
    lyr[o["mask"]] = [*col, 140]
    ov = Image.alpha_composite(ov, Image.fromarray(lyr))
    x, y, bw, bh = o["bbox"]
    cx, cy = o["centroid_px"]
    d = ImageDraw.Draw(ov)
    d.rectangle([x, y, x+bw, y+bh], outline=(*col, 255), width=2)
    d.ellipse([cx-6, cy-6, cx+6, cy+6], outline=(*col, 255), width=3)
    label = f"[{o['idx']}] {o['height_mm']:.0f}mm" if o['height_mm'] else f"[{o['idx']}]"
    d.text((x+3, y+3), label, fill=(255,255,255,255), font=FONT_SM)
ov_rgb = ov.convert("RGB")

# 3) depth map에도 centroid 표시
d_pil = Image.fromarray(d_color)
dd = ImageDraw.Draw(d_pil)
for o in objects:
    cx, cy = o["centroid_px"]
    dd.ellipse([cx-8, cy-8, cx+8, cy+8], outline=(255,255,255), width=3)
    dd.text((cx+10, cy-8), f"{o['depth_mm']:.0f}mm" if o['depth_mm'] else "", fill=(255,255,255), font=FONT_SM)

# 저장
ov_rgb.save(OUT_DIR / "seg_overlay.jpg", quality=92)
d_pil.save(OUT_DIR / "depth_map.jpg", quality=92)
for i, (o, crop) in enumerate(zip(objects, crops)):
    crop.save(OUT_DIR / f"crop_{i:02d}.jpg", quality=92)

# ── VLM 타겟 식별 ─────────────────────────────────────────────
from utils.pipeline_a import DepthClusterPipeline
pipe_a = DepthClusterPipeline()
print(f"\nVLM 식별: {INSTRUCTION}")
t_vlm = time.time()
target_idx, id_res = pipe_a.identify(crops, INSTRUCTION)
t_vlm = time.time() - t_vlm
target = objects[target_idx]

print(f"결과:")
print(f"  타겟: [{target_idx}번]  conf={id_res.get('confidence')}")
print(f"  EE pixel: {target['centroid_px']}")
print(f"  depth:    {target['depth_mm']:.1f}mm")
print(f"  height:   {target['height_mm']:.1f}mm")
print(f"  시간:     seg={t_seg*1000:.0f}ms  vlm={t_vlm:.1f}s")

# 최종 시각화
final = ov_rgb.copy()
df    = ImageDraw.Draw(final)
cx, cy = target["centroid_px"]
df.ellipse([cx-16,cy-16,cx+16,cy+16], outline=(255,60,0), width=5)
df.line([(cx-28,cy),(cx+28,cy)], fill=(255,60,0), width=4)
df.line([(cx,cy-28),(cx,cy+28)], fill=(255,60,0), width=4)
df.text((cx+20, cy-10),
        f"TARGET\nu={cx} v={cy}\n{target['depth_mm']:.0f}mm",
        fill=(255,60,0), font=FONT_MD)
final.save(OUT_DIR / "final_target.jpg", quality=92)

print(f"\n저장: {OUT_DIR}")
