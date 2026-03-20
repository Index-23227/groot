"""
RGBD Localizer: SAM2 없는 경량 파이프라인

버드뷰 + Depth 카메라 기준 설계:
  1. Depth threshold → 테이블 평면 제거, 물체 픽셀만 추출
  2. Connected components → 각 물체 분리
  3. RGB crop 추출
  4. GPT-4o → 타겟 식별
  5. GPT-4o → reasoning (graspability 등)
  6. Depth → 3D 좌표 직접 계산

Depth 없을 시 RGB fallback (엣지+contour 기반) 자동 사용.
"""
import io, base64, json, time
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from openai import OpenAI

ROOT       = Path(__file__).parent.parent
token      = ROOT / "token"
OPENAI_KEY = next(l.strip() for l in token.read_text().splitlines() if l.strip().startswith("sk-"))
GPT_MODEL  = "gpt-4o"


def _pil_to_b64(img: Image.Image, max_side: int = 512) -> str:
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


class RGBDLocalizer:
    """SAM2 없는 RGBD 파이프라인"""

    def __init__(self, depth_scale: float = 0.001):
        """
        Args:
            depth_scale: depth 값 → 미터 변환 계수
                         RealSense 기본값 0.001 (uint16 mm → m)
        """
        self.client      = OpenAI(api_key=OPENAI_KEY)
        self.depth_scale = depth_scale

    # ── Step 1: 객체 분리 ────────────────────────────────────
    def segment_objects(
        self,
        rgb:          np.ndarray,
        depth:        np.ndarray | None = None,
        table_depth:  float | None      = None,
        min_h_mm:     float             = 20,
        max_h_mm:     float             = 400,
        min_area_px:  int               = 500,
        roi:          tuple | None      = None,
    ) -> list[dict]:
        """
        Args:
            rgb:         (H,W,3) uint8 RGB
            depth:       (H,W) uint16 mm  (RealSense raw). None이면 RGB fallback.
            table_depth: 테이블 depth 값(mm). None이면 중앙 영역 median으로 자동 추정.
            min_h_mm:    테이블 위 최소 높이 (물체로 인정할 최소값)
            max_h_mm:    테이블 위 최대 높이 (로봇팔 등 제외)
            min_area_px: 최소 컴포넌트 넓이 (노이즈 제거)
            roi:         (x1,y1,x2,y2) — 이 영역 안에서만 감지

        Returns:
            list of {
              "bbox":       (x,y,w,h),
              "mask":       bool ndarray (H,W),
              "area":       int,
              "centroid_px": (cx, cy),
              "depth_mm":   float | None,   # 물체 중심 depth
              "xyz_m":      (x,y,z) | None, # 카메라 좌표계 3D (intrinsic 필요 시 None)
              "method":     "depth" | "rgb",
            }
        """
        if depth is not None:
            return self._segment_depth(rgb, depth, table_depth,
                                       min_h_mm, max_h_mm, min_area_px, roi)
        else:
            return self._segment_rgb(rgb, min_area_px, roi)

    def _segment_depth(self, rgb, depth, table_depth,
                       min_h_mm, max_h_mm, min_area_px, roi):
        H, W = depth.shape

        # ROI 마스크
        roi_mask = np.zeros((H, W), dtype=bool)
        if roi:
            x1, y1, x2, y2 = roi
            roi_mask[y1:y2, x1:x2] = True
        else:
            roi_mask[:] = True

        # 테이블 depth 자동 추정 (ROI 중앙 영역 median)
        if table_depth is None:
            cy, cx = H // 2, W // 2
            patch  = depth[cy-60:cy+60, cx-60:cx+60]
            valid  = patch[patch > 0]
            table_depth = float(np.median(valid)) if len(valid) > 0 else 600.0
            print(f"  테이블 depth 추정: {table_depth:.0f}mm")

        # 물체 마스크: 테이블보다 가깝고(높고) 일정 범위 내
        obj_mask = (
            roi_mask &
            (depth > 0) &
            (depth < table_depth - min_h_mm) &
            (depth > table_depth - max_h_mm)
        ).astype(np.uint8) * 255

        # Morphological 정리
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN,  k)
        obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, k)

        return self._components_to_dicts(rgb, depth, obj_mask, min_area_px, method="depth")

    def _segment_rgb(self, rgb, min_area_px, roi):
        """Depth 없을 때 RGB 기반 fallback (엣지 + contour)"""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        H, W = gray.shape

        # ROI 마스크 적용
        roi_mask = np.zeros((H, W), dtype=np.uint8)
        if roi:
            x1, y1, x2, y2 = roi
            roi_mask[y1:y2, x1:x2] = 255
        else:
            roi_mask[:] = 255

        # 배경 제거: 밝은 테이블(흰색) 위의 어두운/채도 높은 물체 감지
        # 방법: 테이블 색(밝고 채도 낮음)과 다른 픽셀 추출
        hsv  = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # 테이블 마스크: V>180, S<40 (흰색 테이블)
        table_mask = ((hsv[:,:,2] > 180) & (hsv[:,:,1] < 40)).astype(np.uint8) * 255
        # 물체 = 테이블 아닌 픽셀
        obj_rough = cv2.bitwise_not(table_mask)
        obj_rough = cv2.bitwise_and(obj_rough, roi_mask)

        # 엣지 강화
        edges = cv2.Canny(gray, 30, 100)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)

        combined = cv2.bitwise_or(obj_rough, edges)
        combined = cv2.bitwise_and(combined, roi_mask)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

        return self._components_to_dicts(rgb, None, combined, min_area_px, method="rgb")

    def _components_to_dicts(self, rgb, depth, mask_u8, min_area_px, method):
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_u8, connectivity=8
        )
        results = []
        for i in range(1, num):  # 0 = 배경
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area_px:
                continue
            x  = int(stats[i, cv2.CC_STAT_LEFT])
            y  = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx = int(centroids[i][0])
            cy = int(centroids[i][1])
            comp_mask = (labels == i)

            depth_mm = None
            if depth is not None:
                patch = depth[comp_mask]
                valid = patch[patch > 0]
                depth_mm = float(np.median(valid)) if len(valid) > 0 else None

            results.append({
                "bbox":        (x, y, bw, bh),
                "mask":        comp_mask,
                "area":        area,
                "centroid_px": (cx, cy),
                "depth_mm":    depth_mm,
                "xyz_m":       None,   # intrinsic 있으면 채울 수 있음
                "method":      method,
            })

        # 면적 내림차순 정렬
        results.sort(key=lambda r: r["area"], reverse=True)
        return results

    # ── crop 추출 ─────────────────────────────────────────────
    def make_crop(self, rgb: np.ndarray, obj: dict, padding: int = 12) -> Image.Image:
        """mask 영역 crop, 배경 흰색 처리"""
        H, W  = rgb.shape[:2]
        mask  = obj["mask"]
        x, y, bw, bh = obj["bbox"]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(W, x + bw + padding)
        y2 = min(H, y + bh + padding)
        arr = np.array(Image.fromarray(rgb).convert("RGBA"))
        arr[~mask] = [255, 255, 255, 255]
        return Image.fromarray(arr).convert("RGB").crop((x1, y1, x2, y2))

    # ── Step 2: GPT-4o 타겟 식별 ─────────────────────────────
    def identify_target(self, crops: list[Image.Image], instruction: str) -> int:
        content = [
            {
                "type": "text",
                "text": (
                    f"지시: {instruction}\n\n"
                    f"아래 {len(crops)}개 이미지는 씬에서 분리된 각 객체입니다.\n"
                    f"지시에서 언급된 물체의 번호(0-based)를 답하세요.\n"
                    f'JSON으로만: {{"target_index": 0, "confidence": 0.9, "reason": "이유"}}'
                ),
            }
        ]
        for i, crop in enumerate(crops):
            content.append({"type": "text", "text": f"[{i}번]"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_pil_to_b64(crop)}"},
            })
        resp = self._call(content)
        obj  = self._parse(resp)
        idx  = int(obj.get("target_index", 0))
        print(f"  타겟: [{idx}번]  conf={obj.get('confidence')}  {obj.get('reason','')[:60]}")
        return idx

    # ── Step 3: GPT-4o reasoning ──────────────────────────────
    def reason(self, crop: Image.Image, instruction: str, task: str) -> dict:
        if task == "graspability":
            prompt = (
                f"지시: {instruction}\n\n"
                "이미지는 배경이 제거된 타겟 객체입니다. 버드뷰(위에서 아래로 촬영) 기준입니다.\n"
                "그리퍼가 이 객체를 잡을 수 있는 상태인지 판단하세요.\n"
                'JSON으로만: {"graspable": true, "confidence": 0.9, '
                '"alignment": "정렬됨/약간 틀어짐/많이 틀어짐", "reason": "판단 근거"}'
            )
        elif task == "calibration":
            prompt = (
                f"지시: {instruction}\n\n"
                "이미지는 배경이 제거된 타겟 객체입니다. 버드뷰(위에서 아래로 촬영) 기준입니다.\n"
                "객체의 중심 위치를 normalized 좌표로 추정하세요.\n"
                'JSON으로만: {"center_norm": [cx, cy], "confidence": 0.9, '
                '"orientation": "upright/tilted/lying", "notes": "설명"}'
            )
        else:
            prompt = task
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{_pil_to_b64(crop)}"}},
        ]
        resp = self._call(content)
        return self._parse(resp)

    # ── 전체 파이프라인 ───────────────────────────────────────
    def run(
        self,
        rgb:         np.ndarray,
        instruction: str,
        depth:       np.ndarray | None = None,
        table_depth: float | None      = None,
        tasks:       list[str] | None  = None,
        roi:         tuple | None      = None,
        min_area_px: int               = 500,
    ) -> dict:
        """
        Returns:
            {
              "objects":      list[dict],   # 감지된 모든 객체
              "target_idx":   int,
              "target_obj":   dict,
              "target_crop":  PIL.Image,
              "results":      {task: dict},
              "method":       "depth" | "rgb",
            }
        """
        tasks = tasks or []
        t0 = time.time()

        # Step 1: 객체 분리
        mode = "depth" if depth is not None else "rgb(fallback)"
        print(f"[1/3] 객체 분리 ({mode})...")
        t1 = time.time()
        objects = self.segment_objects(rgb, depth=depth, table_depth=table_depth,
                                       min_area_px=min_area_px, roi=roi)
        crops   = [self.make_crop(rgb, o) for o in objects]
        print(f"  {len(objects)}개 객체  ({time.time()-t1:.2f}s)")

        # Step 2: GPT-4o 타겟 식별
        print(f"[2/3] GPT-4o 타겟 식별 ({len(crops)}개 crops)...")
        t2 = time.time()
        idx = self.identify_target(crops, instruction)
        idx = min(idx, len(objects) - 1)
        target_obj  = objects[idx]
        target_crop = crops[idx]
        print(f"  완료  ({time.time()-t2:.1f}s)")

        # Step 3: reasoning
        results = {}
        for i, task in enumerate(tasks):
            print(f"[3/3] GPT-4o reasoning [{i+1}/{len(tasks)}]: {task}...")
            t3 = time.time()
            results[task] = self.reason(target_crop, instruction, task)
            print(f"  {results[task]}  ({time.time()-t3:.1f}s)")

        print(f"총 소요: {time.time()-t0:.1f}s")
        return {
            "objects":     objects,
            "target_idx":  idx,
            "target_obj":  target_obj,
            "target_crop": target_crop,
            "results":     results,
            "method":      objects[0]["method"] if objects else "none",
        }

    # ── 내부 ─────────────────────────────────────────────────
    def _call(self, content: list, retries: int = 3) -> str | None:
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=300,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                return resp.choices[0].message.content
            except Exception as e:
                wait = 15 if "429" in str(e) else 10
                print(f"  재시도 {attempt+1}/{retries} ({wait}s)... [{e}]")
                time.sleep(wait)
        return None

    def _parse(self, resp: str | None) -> dict:
        if not resp:
            return {}
        text = resp.strip()
        s, e = text.find("{"), text.rfind("}") + 1
        if s < 0 or e <= 0:
            return {}
        return json.loads(text[s:e])
