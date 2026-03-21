"""
Pipeline B: VLM BBox 추출 → Depth Clustering → EE 좌표

Flow:
  1. GPT-4o: 전체 이미지에서 타겟 객체 bbox 직접 추출
  2. bbox 영역 내 depth clustering → 정확한 centroid
  3. EE 픽셀 좌표 & depth 출력

장점: VLM이 장면 전체를 보고 bbox → depth로 정밀 centroid
단점: bbox 오류 시 downstream 전부 영향
"""
import io, base64, json, time
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from openai import OpenAI

ROOT       = Path(__file__).parent.parent
OPENAI_KEY = next(l.strip() for l in (ROOT/"token").read_text().splitlines()
                  if l.strip().startswith("sk-"))


def _to_b64(img: Image.Image, max_side: int = 1024) -> str:
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


class BBoxDepthPipeline:
    """B: VLM으로 타겟 bbox 추출 → bbox 내 depth clustering → EE 좌표"""

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_KEY)

    # ── Step 1: VLM bbox 추출 ────────────────────────────────
    def detect_bbox(self, rgb: np.ndarray, instruction: str) -> tuple[dict, dict]:
        """GPT-4o에 전체 이미지 → 타겟 bbox (normalized) 반환"""
        img_pil = Image.fromarray(rgb)
        content = [
            {
                "type": "text",
                "text": (
                    f"지시: {instruction}\n\n"
                    "이미지에서 지시에 해당하는 객체 하나의 bounding box를 구하세요.\n"
                    "normalized 좌표 (0~1) 기준.\n"
                    'JSON: {"x1":0.0,"y1":0.0,"x2":1.0,"y2":1.0,'
                    '"confidence":0.9,"label":"객체명"}'
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_to_b64(img_pil)}"},
            },
        ]
        raw = self._call(content)
        res = self._parse(raw)
        H, W = rgb.shape[:2]
        bbox_px = (
            int(res.get("x1", 0) * W),
            int(res.get("y1", 0) * H),
            int(res.get("x2", 1) * W),
            int(res.get("y2", 1) * H),
        )
        print(f"    [B] bbox: {bbox_px}  conf={res.get('confidence')}  label={res.get('label')}")
        return bbox_px, res

    # ── Step 2: bbox 내 centroid 추출 ────────────────────────
    def refine_centroid(
        self,
        rgb:      np.ndarray,
        bbox_px:  tuple,
        depth:    np.ndarray | None = None,
        table_d:  float | None      = None,
        min_h_mm: float = 20,
        max_h_mm: float = 350,
    ) -> dict:
        """
        bbox 내에서 depth clustering 또는 RGB 분석으로 정확한 centroid 추출.
        Returns: {centroid_px, depth_mm, mask}
        """
        H, W = rgb.shape[:2]
        x1, y1, x2, y2 = bbox_px

        if depth is not None:
            # depth threshold: 테이블 위 물체만
            if table_d is None:
                cy, cx = H // 2, W // 2
                patch  = depth[cy-80:cy+80, cx-80:cx+80]
                valid  = patch[patch > 0]
                table_d = float(np.median(valid)) if len(valid) else 600.0
                print(f"    테이블 depth 추정: {table_d:.0f}mm")

            roi_depth = depth.copy().astype(np.float32)
            # bbox 밖은 0 처리
            mask_out = np.zeros((H, W), dtype=bool)
            mask_out[y1:y2, x1:x2] = True
            roi_depth[~mask_out] = 0

            obj_mask = (
                (roi_depth > 0) &
                (roi_depth < table_d - min_h_mm) &
                (roi_depth > table_d - max_h_mm)
            )
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            obj_u8 = obj_mask.astype(np.uint8) * 255
            obj_u8 = cv2.morphologyEx(obj_u8, cv2.MORPH_OPEN,  k)
            obj_u8 = cv2.morphologyEx(obj_u8, cv2.MORPH_CLOSE, k)

            # 가장 큰 컴포넌트만 사용
            num, labels, stats, centroids = cv2.connectedComponentsWithStats(obj_u8, 8)
            if num > 1:
                best = 1 + np.argmax([stats[i, cv2.CC_STAT_AREA] for i in range(1, num)])
                final_mask = (labels == best)
                cx = int(centroids[best][0])
                cy = int(centroids[best][1])
            else:
                # fallback: bbox 중심
                cx, cy = (x1+x2)//2, (y1+y2)//2
                final_mask = obj_mask

            vals = depth[final_mask]
            vals = vals[vals > 0]
            depth_mm = float(np.median(vals)) if len(vals) else None

        else:
            # RGB fallback: bbox 내 흰 테이블 제거 후 centroid
            roi_rgb = rgb[y1:y2, x1:x2]
            hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
            table = ((hsv[:,:,2] > 180) & (hsv[:,:,1] < 40)).astype(np.uint8) * 255
            obj_u8 = cv2.bitwise_not(table)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
            obj_u8 = cv2.morphologyEx(obj_u8, cv2.MORPH_OPEN, k)

            M = cv2.moments(obj_u8)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"]) + x1
                cy = int(M["m01"] / M["m00"]) + y1
            else:
                cx, cy = (x1+x2)//2, (y1+y2)//2

            # full-size mask
            final_mask = np.zeros((H, W), dtype=bool)
            roi_mask = obj_u8 > 0
            final_mask[y1:y2, x1:x2] = roi_mask
            depth_mm = None

        return {
            "centroid_px": (cx, cy),
            "depth_mm":    depth_mm,
            "mask":        final_mask,
            "bbox_px":     bbox_px,
        }

    # ── 전체 실행 ─────────────────────────────────────────────
    def run(self, rgb, instruction, depth=None, roi=None) -> dict:
        t0 = time.time()
        H, W = rgb.shape[:2]
        method = "depth" if depth is not None else "rgb"

        # ROI를 이미지에 합성해서 VLM에 전달 (선택적)
        rgb_input = rgb
        if roi:
            rgb_input = rgb[roi[1]:roi[3], roi[0]:roi[2]]

        print(f"  [B-1] VLM bbox 추출...")
        t1 = time.time()
        bbox_px, bbox_res = self.detect_bbox(rgb_input, instruction)
        # ROI offset 복원
        if roi:
            bbox_px = (
                bbox_px[0] + roi[0],
                bbox_px[1] + roi[1],
                bbox_px[2] + roi[0],
                bbox_px[3] + roi[1],
            )
        t1 = time.time() - t1

        print(f"  [B-2] bbox 내 centroid 추출 ({method})...")
        t2 = time.time()
        refine = self.refine_centroid(rgb, bbox_px, depth=depth)
        t2 = time.time() - t2

        cx, cy = refine["centroid_px"]
        ee = {
            "pixel_uv":  (cx, cy),
            "uv_norm":   (round(cx/W, 4), round(cy/H, 4)),
            "depth_mm":  refine["depth_mm"],
        }
        total = time.time() - t0
        print(f"    EE → pixel={ee['pixel_uv']}  depth={ee['depth_mm']}  ({t2:.2f}s)")

        # bbox crop (배경 포함, VLM이 본 이미지)
        bx1, by1, bx2, by2 = bbox_px
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(W, bx2), min(H, by2)
        bbox_crop = Image.fromarray(rgb).crop((bx1, by1, bx2, by2))

        return {
            "method":      method,
            "bbox_px":     bbox_px,
            "bbox_res":    bbox_res,
            "bbox_crop":   bbox_crop,
            "refine":      refine,
            "target_mask": refine["mask"],
            "ee":          ee,
            "times":       {"bbox": t1, "refine": t2, "total": total},
        }

    def _call(self, content, retries=3):
        for i in range(retries):
            try:
                r = self.client.chat.completions.create(
                    model="gpt-4o", temperature=0.1, max_tokens=200,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": content}],
                )
                return r.choices[0].message.content
            except Exception as e:
                wait = 15 if "429" in str(e) else 8
                print(f"    retry {i+1}/{retries} ({wait}s)")
                time.sleep(wait)
        return None

    def _parse(self, raw):
        if not raw: return {}
        s, e = raw.find("{"), raw.rfind("}")+1
        return json.loads(raw[s:e]) if s >= 0 and e > 0 else {}
