"""
RealSense RGB + Depth 검증 테스트
노란 원기둥의 픽셀 좌표(u, v)와 depth값 확인
base10, base11, base12 씬 기준
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import time

try:
    import pyrealsense2 as rs
    HAS_RS = True
except ImportError:
    HAS_RS = False

ROOT    = Path(__file__).parent.parent
out_dir = ROOT / "results" / "depth_pixel_vis"
out_dir.mkdir(parents=True, exist_ok=True)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except:
    font = font_sm = ImageFont.load_default()

# ── 노란색 HSV 범위 ───────────────────────────────────────────
# 노란 원기둥: H=20~35, S>100, V>100
YELLOW_HSV_LOW  = np.array([18,  80,  80])
YELLOW_HSV_HIGH = np.array([38, 255, 255])

def detect_yellow(rgb_np: np.ndarray) -> dict | None:
    """HSV 필터로 노란 원기둥 감지 → 픽셀 중심 반환"""
    hsv  = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, YELLOW_HSV_LOW, YELLOW_HSV_HIGH)

    # 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 가장 큰 contour 선택
    c    = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 200:
        return None

    M  = cv2.moments(c)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    x, y, w, h = cv2.boundingRect(c)

    return {"cx": cx, "cy": cy, "area": area,
            "bbox": (x, y, w, h), "mask": mask, "contour": c}

def visualize(rgb_np, depth_np, det, img_name, depth_m=None):
    """결과 시각화 이미지 생성"""
    h, w = rgb_np.shape[:2]
    img_pil = Image.fromarray(rgb_np).convert("RGB")
    draw    = ImageDraw.Draw(img_pil)

    if det:
        cx, cy = det["cx"], det["cy"]
        x, y, bw, bh = det["bbox"]

        # 노란 contour 오버레이
        overlay_arr = np.zeros((h, w, 4), dtype=np.uint8)
        cv2.drawContours(overlay_arr, [det["contour"]], -1, (255, 220, 0, 160), -1)
        overlay_img = Image.fromarray(overlay_arr, "RGBA")
        img_pil = Image.alpha_composite(img_pil.convert("RGBA"), overlay_img).convert("RGB")
        draw    = ImageDraw.Draw(img_pil)

        # bbox
        draw.rectangle([x, y, x+bw, y+bh], outline=(255, 200, 0), width=3)

        # 중심 십자
        r = 8
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(255, 60, 0), width=3)
        draw.line([(cx-20, cy), (cx+20, cy)], fill=(255, 60, 0), width=2)
        draw.line([(cx, cy-20), (cx, cy+20)], fill=(255, 60, 0), width=2)

        # 픽셀 좌표 + depth 텍스트
        depth_str = f"{depth_m*1000:.1f}mm" if depth_m else "N/A"
        info = f"pixel: ({cx}, {cy})   depth: {depth_str}   area: {int(det['area'])}px²"
        draw.rectangle([x, y-26, x+len(info)*9, y-2], fill=(40, 40, 40))
        draw.text((x+4, y-24), info, fill=(255, 220, 0), font=font_sm)

    else:
        draw.text((20, 20), "노란 원기둥 감지 실패", fill=(255, 80, 80), font=font)

    # depth colormap 패널 (오른쪽)
    if depth_np is not None:
        d_clip  = np.clip(depth_np, 0, 2000).astype(np.float32)
        d_norm  = (d_clip / 2000 * 255).astype(np.uint8)
        d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
        d_color = cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB)
        if det:
            cx, cy = det["cx"], det["cy"]
            cv2.circle(d_color, (cx, cy), 10, (255, 255, 255), 3)
        depth_pil = Image.fromarray(d_color)
    else:
        depth_pil = Image.new("RGB", (w, h), (30, 30, 30))
        ImageDraw.Draw(depth_pil).text((10, 10), "Depth N/A", fill=(200,200,200), font=font)

    # 좌: RGB+detection / 우: depth map
    combined = Image.new("RGB", (w * 2, h + 50), (20, 20, 20))
    combined.paste(img_pil,   (0, 50))
    combined.paste(depth_pil, (w, 50))

    d = ImageDraw.Draw(combined)
    d.text((10, 10), f"{img_name} — RGB + Detection", fill=(220,220,220), font=font)
    d.text((w+10, 10), f"{img_name} — Depth map (0~2m, JET)", fill=(220,220,220), font=font)

    return combined

# ── 각 씬 캡처 + 분석 ────────────────────────────────────────
SCENES = ["base10", "base11", "base12"]

if not HAS_RS:
    print("pyrealsense2 없음 — 기존 RGB 이미지로 depth 없이 테스트")

print("=" * 60)
print("  RealSense Depth + Pixel 검증 (노란 원기둥)")
print("=" * 60)

for scene in SCENES:
    print(f"\n[{scene}] 캡처 중...")

    if HAS_RS:
        pipe = rs.pipeline()
        cfg  = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        profile = pipe.start(cfg)

        # depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale  = depth_sensor.get_depth_scale()

        # align depth to color
        align = rs.align(rs.stream.color)

        for _ in range(30): pipe.wait_for_frames()  # 워밍업
        frames  = align.process(pipe.wait_for_frames())
        color_f = frames.get_color_frame()
        depth_f = frames.get_depth_frame()
        pipe.stop()

        rgb_np   = np.asanyarray(color_f.get_data())[:, :, ::-1]  # BGR→RGB
        depth_np = np.asanyarray(depth_f.get_data())              # mm (uint16)

        # RGB 이미지 덮어쓰기 (최신 캡처)
        Image.fromarray(rgb_np).save(ROOT / "data" / "base_images" / f"{scene}.jpg", quality=95)

    else:
        # 기존 이미지 사용 (depth 없음)
        rgb_np   = np.array(Image.open(ROOT / "data" / "base_images" / f"{scene}.jpg").convert("RGB"))
        depth_np = None
        depth_scale = 0.001

    # 노란 원기둥 감지
    det = detect_yellow(rgb_np)

    if det:
        cx, cy = det["cx"], det["cy"]
        if depth_np is not None:
            raw_d   = depth_np[cy, cx]
            depth_m = raw_d * depth_scale
        else:
            depth_m = None

        print(f"  ✅ 감지 성공")
        print(f"     픽셀 좌표: u={cx}, v={cy}")
        print(f"     depth: {depth_m*1000:.1f}mm" if depth_m else "     depth: N/A")
        print(f"     area: {int(det['area'])}px²")

        # 정규화 좌표
        h, w = rgb_np.shape[:2]
        print(f"     bbox_norm: [{det['bbox'][0]/w:.3f}, {det['bbox'][1]/h:.3f}, "
              f"{det['bbox'][2]/w:.3f}, {det['bbox'][3]/h:.3f}]")
    else:
        depth_m = None
        print(f"  ❌ 노란 원기둥 감지 실패 (HSV 범위 조정 필요)")

    # 시각화 저장
    vis = visualize(rgb_np, depth_np, det, scene, depth_m)
    out_path = out_dir / f"{scene}_depth_pixel.jpg"
    vis.save(out_path, quality=92)
    print(f"  저장: {out_path.name}")

print(f"\n완료: {out_dir}")
