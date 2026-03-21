"""
Pipeline A: Depth Clustering → VLM 식별 → EE 좌표

Flow:
  1. Depth threshold / RGB contour 로 모든 객체 blind segmentation
  2. GPT-4o: crops 보여주고 타겟 선택
  3. 타겟 centroid + depth → EE 픽셀 좌표 & depth

장점: VLM이 전체 context 보고 판단
단점: 모든 객체 crop을 VLM에 전달해야 함
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


def _to_b64(img: Image.Image, max_side: int = 400) -> str:
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


class DepthClusterPipeline:
    """A: 먼저 depth/RGB로 모든 객체 분리 → VLM으로 타겟 선택 → EE 좌표"""

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_KEY)

    # ── Step 1: 객체 분리 ─────────────────────────────────────
    def segment(
        self,
        rgb:         np.ndarray,
        depth:       np.ndarray | None = None,
        table_d:     float | None      = None,
        min_h_mm:    float = 20,
        max_h_mm:    float = 350,
        min_area:    int   = 400,
        roi:         tuple | None      = None,
    ) -> list[dict]:
        """
        Returns list of:
          {bbox, mask, area, centroid_px, depth_mm}
        """
        H, W = rgb.shape[:2]

        # ROI 마스크
        roi_mask = np.zeros((H, W), dtype=np.uint8)
        x1r, y1r, x2r, y2r = roi if roi else (0, 0, W, H)
        roi_mask[y1r:y2r, x1r:x2r] = 255

        if depth is not None:
            obj_u8 = self._mask_from_depth(depth, roi_mask, table_d, min_h_mm, max_h_mm)
        else:
            obj_u8 = self._mask_from_rgb(rgb, roi_mask)

        return self._components(rgb, depth, obj_u8, min_area)

    def _mask_from_depth(self, depth, roi_mask, table_d, min_h, max_h):
        if table_d is None:
            H, W = depth.shape
            cy, cx = H // 2, W // 2
            patch = depth[cy-80:cy+80, cx-80:cx+80]
            valid = patch[patch > 0]
            table_d = float(np.median(valid)) if len(valid) else 600.0
            print(f"    테이블 depth 추정: {table_d:.0f}mm")
        obj = (
            (depth > 0) &
            (depth < table_d - min_h) &
            (depth > table_d - max_h)
        ).astype(np.uint8) * 255
        obj = cv2.bitwise_and(obj, roi_mask)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN,  k)
        obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, k)
        return obj

    def _mask_from_rgb(self, rgb, roi_mask):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # 흰 테이블 제거
        table = ((hsv[:,:,2] > 180) & (hsv[:,:,1] < 40)).astype(np.uint8) * 255
        obj   = cv2.bitwise_not(table)
        gray  = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)
        obj   = cv2.bitwise_or(obj, edges)
        obj   = cv2.bitwise_and(obj, roi_mask)
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, k5)
        obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
        return obj

    def _components(self, rgb, depth, mask_u8, min_area):
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
        results = []
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x  = int(stats[i, cv2.CC_STAT_LEFT])
            y  = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx = int(centroids[i][0])
            cy = int(centroids[i][1])
            comp_mask = (labels == i)
            depth_mm  = None
            if depth is not None:
                vals = depth[comp_mask]
                vals = vals[vals > 0]
                depth_mm = float(np.median(vals)) if len(vals) else None
            results.append({
                "bbox":        (x, y, bw, bh),
                "mask":        comp_mask,
                "area":        area,
                "centroid_px": (cx, cy),
                "depth_mm":    depth_mm,
            })
        results.sort(key=lambda r: r["area"], reverse=True)
        return results

    def make_crop(self, rgb: np.ndarray, obj: dict, pad: int = 10) -> Image.Image:
        H, W = rgb.shape[:2]
        x, y, bw, bh = obj["bbox"]
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(W, x+bw+pad), min(H, y+bh+pad)
        arr = np.array(Image.fromarray(rgb).convert("RGBA"))
        arr[~obj["mask"]] = [255, 255, 255, 255]
        return Image.fromarray(arr).convert("RGB").crop((x1, y1, x2, y2))

    # ── Step 2: VLM 타겟 식별 ────────────────────────────────
    def identify(self, crops: list[Image.Image], instruction: str) -> tuple[int, dict]:
        content = [{
            "type": "text",
            "text": (
                f"지시: {instruction}\n\n"
                f"아래 {len(crops)}개는 씬의 각 객체입니다.\n"
                f"지시에 맞는 물체 번호(0-based)를 답하세요.\n"
                f'JSON: {{"target_index":0,"confidence":0.9,"reason":"이유"}}'
            ),
        }]
        for i, c in enumerate(crops):
            content += [
                {"type": "text", "text": f"[{i}번]"},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{_to_b64(c)}"}},
            ]
        raw = self._call(content)
        res = self._parse(raw)
        idx = min(int(res.get("target_index", 0)), len(crops)-1)
        print(f"    [A] 타겟: [{idx}번]  conf={res.get('confidence')}  {res.get('reason','')[:60]}")
        return idx, res

    # ── 전체 실행 ─────────────────────────────────────────────
    def run(self, rgb, instruction, depth=None, roi=None) -> dict:
        t0 = time.time()
        method = "depth" if depth is not None else "rgb"
        print(f"  [A-1] 객체 분리 ({method})...")
        t1 = time.time()
        objs  = self.segment(rgb, depth=depth, roi=roi)
        crops = [self.make_crop(rgb, o) for o in objs]
        t1 = time.time() - t1
        print(f"    {len(objs)}개  ({t1:.2f}s)")

        print(f"  [A-2] VLM 타겟 식별 ({len(objs)}개 crops)...")
        t2 = time.time()
        idx, id_res = self.identify(crops, instruction)
        target = objs[idx]
        t2 = time.time() - t2

        H, W = rgb.shape[:2]
        cx, cy = target["centroid_px"]
        ee = {
            "pixel_uv":   (cx, cy),
            "uv_norm":    (round(cx/W, 4), round(cy/H, 4)),
            "depth_mm":   target["depth_mm"],
        }
        total = time.time() - t0
        print(f"    EE → pixel={ee['pixel_uv']}  depth={ee['depth_mm']}  ({t2:.1f}s)")
        return {
            "method":     method,
            "objects":    objs,
            "crops":      crops,
            "target_idx": idx,
            "target":     target,
            "target_crop":crops[idx],
            "id_res":     id_res,
            "ee":         ee,
            "times":      {"segment": t1, "identify": t2, "total": total},
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
