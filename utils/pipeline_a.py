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

    # ── Step 2-CoT: 전체 이미지 기반 CoT 식별 ─────────────────
    def identify_cot(
        self,
        rgb: np.ndarray,
        instruction: str,
        scene_objects: list[str] | None = None,
    ) -> dict:
        """
        Chain-of-Thought 방식으로 타겟 객체 EE 좌표 추출.

        Args:
            rgb: 전체 씬 이미지
            instruction: 자연어 지시 (예: "노란 원기둥을 집어라")
            scene_objects: 씬에 있는 객체 목록 (primitive 서술)
                ex) ["파란 에너지드링크 캔", "초록 캔", ...]

        CoT 순서:
          1. 타겟 객체 (무엇인가?)
          2. 타겟 객체 전체 bounding box (normalized 0~1)
          3. 타겟 객체 윗면 bounding box (normalized 0~1)
          4. 타겟 객체 윗면 중심점 → EE target (normalized 0~1)

        Returns dict with CoT results + ee_px, obj_bbox_px, top_bbox_px
        """
        H, W = rgb.shape[:2]
        img_pil = Image.fromarray(rgb)

        # 씬 객체 목록 서술
        if scene_objects:
            obj_list = "\n".join(f"  {i+1}. {o}" for i, o in enumerate(scene_objects))
            scene_desc = (
                f"현재 테이블 위에 {len(scene_objects)}개의 객체가 있습니다:\n"
                f"{obj_list}\n\n"
            )
        else:
            scene_desc = ""

        prompt = (
            f"{scene_desc}"
            f"지시: {instruction}\n\n"
            "이미지를 보고 아래 순서대로 단계별(Chain-of-Thought)로 추론하세요.\n\n"
            "Step 1. 위 객체 목록과 이미지를 참고하여, 지시에 해당하는 타겟 객체가 어떤 것인지 식별하세요.\n"
            "Step 2. 타겟 객체 전체를 감싸는 bounding box를 구하세요 (normalized 0~1 좌표).\n"
            "Step 3. 타겟 객체의 윗면(top surface)만 감싸는 bounding box를 구하세요 (normalized 0~1 좌표).\n"
            "        - 원기둥이면 상단 원형/타원 면, 캔이면 뚜껑 면, 박스면 상단 직사각형 면\n"
            "        - 카메라가 측면+위 약 45도 각도로 촬영하고 있음을 고려할 것\n"
            "Step 4. 윗면 bounding box의 중심점(center)을 구하세요.\n"
            "        → 이것이 로봇 End-Effector가 접근할 타겟 위치입니다.\n\n"
            "반드시 아래 JSON 형식으로만 답하세요:\n"
            '{"step1_target": "타겟 객체 설명",'
            ' "step2_object_bbox": {"x1":0.0,"y1":0.0,"x2":1.0,"y2":1.0},'
            ' "step3_top_surface_bbox": {"x1":0.0,"y1":0.0,"x2":1.0,"y2":1.0},'
            ' "step4_top_center": {"u":0.5,"v":0.5},'
            ' "confidence": 0.9,'
            ' "reason": "추론 근거"}'
        )
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{_to_b64(img_pil, max_side=1024)}"}},
        ]
        raw = self._call(content, max_tokens=500)
        res = self._parse(raw)

        # EE 픽셀 좌표 계산
        tc = res.get("step4_top_center", {})
        u_norm = float(tc.get("u", 0.5))
        v_norm = float(tc.get("v", 0.5))
        ee_px  = (int(u_norm * W), int(v_norm * H))

        # bbox 픽셀 좌표 계산
        ob = res.get("step2_object_bbox", {})
        tb = res.get("step3_top_surface_bbox", {})
        obj_bbox_px = (
            int(ob.get("x1", 0) * W), int(ob.get("y1", 0) * H),
            int(ob.get("x2", 1) * W), int(ob.get("y2", 1) * H),
        )
        top_bbox_px = (
            int(tb.get("x1", 0) * W), int(tb.get("y1", 0) * H),
            int(tb.get("x2", 1) * W), int(tb.get("y2", 1) * H),
        )

        res["ee_px"]         = ee_px
        res["obj_bbox_px"]   = obj_bbox_px
        res["top_bbox_px"]   = top_bbox_px
        res["uv_norm"]       = (u_norm, v_norm)

        print(f"    [CoT] target: {res.get('step1_target','')}")
        print(f"    [CoT] obj_bbox: {obj_bbox_px}")
        print(f"    [CoT] top_bbox: {top_bbox_px}")
        print(f"    [CoT] EE pixel: {ee_px}  (norm={u_norm:.3f},{v_norm:.3f})")
        print(f"    [CoT] conf={res.get('confidence')}  {res.get('reason','')[:60]}")
        return res

    # ── Step 2: VLM 타겟 식별 (기존) ────────────────────────────────
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

    def _call(self, content, retries=3, max_tokens=200):
        for i in range(retries):
            try:
                r = self.client.chat.completions.create(
                    model="gpt-4o", temperature=0.1, max_tokens=max_tokens,
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
