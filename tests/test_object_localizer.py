"""
ObjectLocalizer 테스트: SAM2 → Gemini ER pipeline
base1 이미지로 전체 파이프라인 검증
"""
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).parent.parent))
from utils.object_localizer import ObjectLocalizer

img_path = Path(__file__).parent.parent / "data" / "base_images" / "base1.jpg"
out_dir  = Path(__file__).parent.parent / "results" / "localizer_vis"
out_dir.mkdir(parents=True, exist_ok=True)

# 이미지 로드
image = np.array(Image.open(img_path).convert("RGB"))

# 파이프라인 실행
localizer = ObjectLocalizer()
output = localizer.run(
    image=image,
    instruction="스프라이트 캔을 집어라",
    tasks=["graspability", "calibration"],
)

# 결과 출력
print("\n=== 결과 ===")
print(f"총 mask 수: {output['num_masks']}")
for task, result in output["results"].items():
    print(f"[{task}] {result}")

# 시각화: 전체 이미지에 target mask 오버레이
mask_arr = output["target_mask"]["segmentation"]
img_vis  = Image.fromarray(image).convert("RGBA")
overlay  = Image.new("RGBA", img_vis.size, (0, 0, 0, 0))
odraw    = ImageDraw.Draw(overlay)

pts_y, pts_x = np.where(mask_arr)
if len(pts_x) > 0:
    from PIL import ImageFilter
    mask_img = Image.fromarray((mask_arr * 180).astype(np.uint8), mode="L")
    green_overlay = Image.new("RGBA", img_vis.size, (0, 220, 80, 0))
    green_overlay.putalpha(mask_img)
    img_vis = Image.alpha_composite(img_vis, green_overlay)

result_img = img_vis.convert("RGB")

# crop 저장
crop_path = out_dir / "base1_target_crop.jpg"
output["target_crop"].save(crop_path, quality=92)

# overlay 저장
vis_path = out_dir / "base1_mask_overlay.jpg"
result_img.save(vis_path, quality=92)

print(f"\n저장:")
print(f"  crop:    {crop_path}")
print(f"  overlay: {vis_path}")
