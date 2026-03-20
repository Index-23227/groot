"""
Object Localizer: SAM2 → GPT-4o pipeline

Flow:
  1. SAM2 (instruction 없음): 테이블 영역 내 객체 blind segmentation
  2. GPT-4o vision (instruction + top crops): 타겟 객체 식별 (1회)
  3. GPT-4o vision (instruction + target clean crop): embodied reasoning (N회)

Usage:
    localizer = ObjectLocalizer()
    result = localizer.run(
        image=image_np,
        instruction="노란 원기둥을 집어라",
        tasks=["graspability"],
        table_region=(x1, y1, x2, y2),   # 선택: 테이블 ROI (픽셀 좌표)
    )
"""
import base64
import io
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import cv2

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from openai import OpenAI

# ── 경로 설정 ────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
CHECKPOINT = ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
SAM2_CFG   = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

token      = ROOT / "token"
OPENAI_KEY = next(l.strip() for l in token.read_text().splitlines() if l.strip().startswith("sk-"))
GPT_MODEL  = "gpt-4o"   # vision + reasoning

# yellow HSV filter (top-N 필터링용)
_YLW_LO = np.array([18,  80,  80])
_YLW_HI = np.array([38, 255, 255])


def _pil_to_b64(img: Image.Image, max_side: int = 512) -> str:
    """PIL 이미지를 base64 JPEG 문자열로 변환 (GPT vision 입력용)"""
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


class ObjectLocalizer:
    def __init__(self, table_region: tuple | None = None):
        """
        Args:
            table_region: (x1, y1, x2, y2) 픽셀 좌표.
              None이면 detect_table_region()으로 자동 감지 후 캐시.
              직접 고정값을 넣으면 GPT 호출 없이 바로 사용.
        """
        print(f"SAM2 로딩... (device={DEVICE})")
        sam2_model = build_sam2(SAM2_CFG, str(CHECKPOINT), device=DEVICE)
        self.generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=16,
            pred_iou_thresh=0.80,
            stability_score_thresh=0.90,
            min_mask_region_area=800,
        )
        self.client = OpenAI(api_key=OPENAI_KEY)
        self._table_region = table_region  # None이면 첫 run()에서 자동 감지
        print("준비 완료")

    # ── 테이블 ROI 자동 감지 (최초 1회) ─────────────────────
    def detect_table_region(self, image: np.ndarray) -> tuple:
        """GPT-4o로 회색 테이블 영역 bbox 감지 → pixel (x1,y1,x2,y2)"""
        H, W = image.shape[:2]
        b64 = _pil_to_b64(Image.fromarray(image), max_side=1024)
        content = [
            {
                "type": "text",
                "text": (
                    "이 이미지에서 테이블 위에 놓인 모든 조작 가능한 객체들(캔, 블록, 원기둥 등)을 "
                    "tight하게 감싸는 bounding box를 구해주세요.\n"
                    "로봇 팔, 테이블 표면 자체, 배경은 제외하고 "
                    "오직 집을 수 있는 물체들만 포함하세요.\n"
                    "물체들을 모두 포함할 수 있도록 약간의 여백(padding)을 추가하세요.\n"
                    "normalized 좌표 (0~1)로 답해주세요.\n"
                    'JSON으로만: {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "confidence": 0.9}'
                ),
            },
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
        resp = self._call(content)
        obj  = self._parse(resp)
        roi  = (
            int(obj.get("x1", 0) * W),
            int(obj.get("y1", 0) * H),
            int(obj.get("x2", 1) * W),
            int(obj.get("y2", 1) * H),
        )
        print(f"  테이블 ROI 감지: {roi}  (conf={obj.get('confidence','')})")
        return roi

    # ── Step 1: SAM2 ─────────────────────────────────────────
    def segment_all(self, image: np.ndarray,
                    table_region: tuple | None = None) -> list[dict]:
        """
        blind segmentation. instruction 없음.

        table_region: (x1, y1, x2, y2) 픽셀 좌표.
          지정하면 해당 ROI만 SAM2에 입력 → 마스크를 원본 좌표계로 복원.
        """
        if table_region is None:
            masks = self.generator.generate(image)
            masks = sorted(masks, key=lambda x: x["area"], reverse=True)
            return masks

        x1, y1, x2, y2 = [int(v) for v in table_region]
        H, W = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        cropped = image[y1:y2, x1:x2]
        raw_masks = self.generator.generate(cropped)

        translated = []
        for m in raw_masks:
            seg_full = np.zeros((H, W), dtype=bool)
            seg_full[y1:y2, x1:x2] = m["segmentation"]
            bx, by, bw, bh = m["bbox"]
            translated.append({
                **m,
                "segmentation": seg_full,
                "bbox": (bx + x1, by + y1, bw, bh),
            })

        translated = sorted(translated, key=lambda x: x["area"], reverse=True)
        return translated

    def make_crop(self, image: np.ndarray, mask_data: dict,
                  padding: int = 12) -> Image.Image:
        """mask 영역 crop, 배경은 흰색 처리"""
        mask = mask_data["segmentation"].astype(bool)
        h, w = image.shape[:2]
        x, y, bw, bh = [int(v) for v in mask_data["bbox"]]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + bw + padding)
        y2 = min(h, y + bh + padding)

        arr = np.array(Image.fromarray(image).convert("RGBA"))
        arr[~mask] = [255, 255, 255, 255]
        return Image.fromarray(arr).convert("RGB").crop((x1, y1, x2, y2))

    def filter_top_crops(self, masks: list[dict], crops: list[Image.Image],
                         top_n: int = 5) -> list[tuple]:
        """area + yellow HSV 점수로 상위 N개 (orig_idx, mask, crop, score)"""
        max_area = max(m["area"] for m in masks) + 1

        def yellow_score(crop_pil):
            arr = np.array(crop_pil.convert("RGB"))
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, _YLW_LO, _YLW_HI)
            return float(mask.sum()) / (arr.shape[0] * arr.shape[1] * 255 + 1e-6)

        scored = [
            (i, m, c, m["area"] / max_area + yellow_score(c) * 3.0)
            for i, (m, c) in enumerate(zip(masks, crops))
        ]
        return sorted(scored, key=lambda x: x[3], reverse=True)[:top_n]

    # ── Step 2: GPT-4o — 타겟 식별 ──────────────────────────
    def identify_target(self, crops: list[Image.Image],
                        instruction: str) -> int:
        """GPT-4o vision: instruction + crop들 → 타겟 인덱스 반환"""
        content = [
            {
                "type": "text",
                "text": (
                    f"지시: {instruction}\n\n"
                    f"아래 {len(crops)}개 이미지는 씬에서 분리된 각 객체입니다.\n"
                    f"지시에서 언급된 물체의 번호(0-based)를 답하세요.\n"
                    f'JSON으로만: {{"target_index": 0, "confidence": 0.90, "reason": "이유"}}'
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
        print(f"  타겟: [{idx}번]  conf={obj.get('confidence')}  {obj.get('reason','')[:50]}")
        return idx

    # ── Step 3: GPT-4o — reasoning ───────────────────────────
    def reason(self, crop: Image.Image, instruction: str, task: str) -> dict:
        """clean crop + instruction으로 embodied reasoning"""
        if task == "graspability":
            prompt = (
                f"지시: {instruction}\n\n"
                "이미지는 배경이 제거된 타겟 객체만의 이미지입니다.\n"
                "그리퍼가 이 객체를 잡을 수 있는 위치에 있는지 판단하세요.\n"
                'JSON으로만: {"graspable": true, "confidence": 0.90, '
                '"alignment": "정렬됨/약간 틀어짐/많이 틀어짐", "reason": "판단 근거"}'
            )
        elif task == "calibration":
            prompt = (
                f"지시: {instruction}\n\n"
                "이미지는 배경이 제거된 타겟 객체만의 이미지입니다.\n"
                "객체의 정확한 중심 위치를 normalized 좌표로 추정하세요.\n"
                'JSON으로만: {"center_norm": [cx, cy], "confidence": 0.90, '
                '"orientation": "upright/tilted/lying", "notes": "설명"}'
            )
        else:
            prompt = task

        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_pil_to_b64(crop)}"},
            },
        ]
        resp = self._call(content)
        return self._parse(resp)

    # ── 전체 파이프라인 ──────────────────────────────────────
    def run(self, image: np.ndarray, instruction: str,
            tasks: list[str] | None = None,
            top_n: int = 5) -> dict:
        """
        단일 이미지에 대한 전체 파이프라인.
        테이블 ROI는 __init__에서 설정하거나 첫 호출 시 자동 감지 후 캐시됨.

        Args:
            image:       RGB numpy array (H, W, 3)
            instruction: 자연어 지시 (예: "노란 원기둥을 집어라")
            tasks:       reasoning 태스크 목록 ["graspability", "calibration", ...]
            top_n:       GPT-4o에 전달할 상위 crop 수

        Returns:
            {
              "num_masks":    int,
              "target_index": int,
              "target_mask":  dict,
              "target_crop":  PIL.Image,
              "table_region": tuple,
              "results":      {task: dict, ...},
            }
        """
        tasks = tasks or []
        t0 = time.time()

        # 테이블 ROI: 캐시된 값 사용, 없으면 GPT-4o로 자동 감지 후 캐시
        if self._table_region is None:
            print("[0/3] GPT-4o: 테이블 ROI 자동 감지 (최초 1회)...")
            t_roi = time.time()
            self._table_region = self.detect_table_region(image)
            print(f"  완료  ({time.time()-t_roi:.1f}s) → 이후 캐시 사용")
        else:
            print(f"[0/3] 테이블 ROI 캐시 사용: {self._table_region}")

        # Step 1: SAM2
        roi_str = f"ROI={self._table_region}"
        print(f"[1/3] SAM2: blind segmentation ({roi_str})...")
        t1 = time.time()
        masks = self.segment_all(image, table_region=self._table_region)
        crops = [self.make_crop(image, m) for m in masks]
        print(f"  {len(masks)}개 객체 감지  ({time.time()-t1:.2f}s)")

        # Step 2: GPT-4o — top-N 필터 후 타겟 식별
        print(f"[2/3] GPT-4o: 타겟 식별 (top-{top_n} crops)...")
        t2 = time.time()
        filtered = self.filter_top_crops(masks, crops, top_n=top_n)
        filtered_crops = [c for _, _, c, _ in filtered]
        pos = self.identify_target(filtered_crops, instruction)
        pos = min(pos, len(filtered) - 1)
        orig_idx = filtered[pos][0]
        target_mask = masks[orig_idx]
        target_crop = crops[orig_idx]
        print(f"  완료  ({time.time()-t2:.1f}s)")

        # Step 3: GPT-4o reasoning
        results = {}
        for i, task in enumerate(tasks):
            print(f"[3/3] GPT-4o reasoning [{i+1}/{len(tasks)}]: {task}...")
            t3 = time.time()
            results[task] = self.reason(target_crop, instruction, task)
            print(f"  {results[task]}  ({time.time()-t3:.1f}s)")

        print(f"총 소요: {time.time()-t0:.1f}s")
        return {
            "num_masks":    len(masks),
            "target_index": orig_idx,
            "target_mask":  target_mask,
            "target_crop":  target_crop,
            "table_region": self._table_region,
            "results":      results,
        }

    # ── 내부 유틸 ────────────────────────────────────────────
    def _call(self, content: list, retries: int = 3) -> str | None:
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=512,
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
