"""
Object Localizer: SAM2 → Gemini ER pipeline

Flow:
  1. SAM2 (instruction 없음): 씬의 모든 객체 blind segmentation
  2. Gemini (instruction + all crops): 타겟 객체 식별 (1회)
  3. Gemini (instruction + target clean crop): embodied reasoning (N회)
"""
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image
import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from google import genai
from google.genai import types

# ── 경로 설정 ────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
CHECKPOINT = ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
SAM2_CFG   = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

token      = ROOT / "token"
GOOGLE_KEY = next(l.strip() for l in token.read_text().splitlines() if l.strip().startswith("AIza"))
GEMINI_MODEL = "gemini-robotics-er-1.5-preview"


class ObjectLocalizer:
    def __init__(self):
        print(f"SAM2 로딩... (device={DEVICE})")
        sam2_model = build_sam2(SAM2_CFG, str(CHECKPOINT), device=DEVICE)
        self.generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=16,
            pred_iou_thresh=0.80,
            stability_score_thresh=0.90,
            min_mask_region_area=800,
        )
        self.client = genai.Client(api_key=GOOGLE_KEY)
        print("준비 완료")

    # ── Step 1: SAM2 (no instruction) ───────────────────────
    def segment_all(self, image: np.ndarray) -> list[dict]:
        """씬의 모든 객체를 blind segmentation. instruction 없음."""
        masks = self.generator.generate(image)
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        return masks

    def make_crop(self, image: np.ndarray, mask_data: dict, padding: int = 12) -> Image.Image:
        """mask 영역 crop, 배경은 흰색 처리"""
        mask  = mask_data["segmentation"].astype(bool)
        h, w  = image.shape[:2]
        x, y, bw, bh = [int(v) for v in mask_data["bbox"]]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + bw + padding)
        y2 = min(h, y + bh + padding)

        arr = np.array(Image.fromarray(image).convert("RGBA"))
        arr[~mask] = [255, 255, 255, 255]
        return Image.fromarray(arr).convert("RGB").crop((x1, y1, x2, y2))

    # ── Step 2: Gemini — 타겟 식별 (instruction 첫 등장) ────
    def identify_target(self, crops: list[Image.Image], instruction: str) -> int:
        """
        Gemini가 instruction + crop들을 보고 타겟을 선택.
        SAM2는 이 단계에 관여하지 않음.
        """
        contents = [
            f"지시: {instruction}\n\n"
            f"아래 {len(crops)}개 이미지는 씬에서 분리된 각 객체입니다.\n"
            f"지시에서 언급된 물체의 번호(0-based)를 답하세요.\n"
            f"JSON으로만: {{\"target_index\": 0, \"confidence\": 0.90, \"reason\": \"이유\"}}"
        ]
        for i, crop in enumerate(crops):
            contents.append(f"[{i}번]")
            contents.append(crop)

        resp = self._call(contents)
        obj  = self._parse(resp)
        idx  = int(obj.get("target_index", 0))
        print(f"  타겟: [{idx}번]  conf={obj.get('confidence')}  {obj.get('reason','')[:50]}")
        return idx

    # ── Step 3: Gemini — clean crop으로 reasoning ───────────
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

        resp = self._call([prompt, crop])
        return self._parse(resp)

    # ── 전체 파이프라인 ──────────────────────────────────────
    def run(self, image: np.ndarray, instruction: str, tasks: list[str] = None) -> dict:
        tasks = tasks or []
        t0 = time.time()

        # Step 1: SAM2 — blind segmentation (instruction 없음)
        print("[1/3] SAM2: blind segmentation...")
        t1 = time.time()
        masks = self.segment_all(image)
        crops = [self.make_crop(image, m) for m in masks]
        print(f"  {len(masks)}개 객체 감지  ({time.time()-t1:.2f}s)")

        # Step 2: Gemini — instruction으로 타겟 선택
        print("[2/3] Gemini: 타겟 식별...")
        t2 = time.time()
        idx = self.identify_target(crops, instruction)
        idx = min(idx, len(masks) - 1)
        target_mask = masks[idx]
        target_crop = crops[idx]
        print(f"  완료  ({time.time()-t2:.1f}s)")

        # Step 3: Gemini — clean crop으로 reasoning
        results = {}
        for i, task in enumerate(tasks):
            print(f"[3/3] Gemini reasoning [{i+1}/{len(tasks)}]: {task}...")
            t3 = time.time()
            results[task] = self.reason(target_crop, instruction, task)
            print(f"  {results[task]}  ({time.time()-t3:.1f}s)")

        print(f"총 소요: {time.time()-t0:.1f}s")
        return {
            "num_masks": len(masks),
            "target_index": idx,
            "target_mask": target_mask,
            "target_crop": target_crop,
            "results": results,
        }

    # ── 내부 유틸 ────────────────────────────────────────────
    def _call(self, contents, retries=3):
        for attempt in range(retries):
            try:
                return self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                    ),
                )
            except Exception as e:
                wait = 15 if "429" in str(e) else 10
                print(f"  재시도 {attempt+1}/{retries} ({wait}s)...")
                time.sleep(wait)
        raise RuntimeError("Gemini 호출 실패")

    def _parse(self, resp) -> dict:
        text = resp.text.strip()
        s, e = text.find("{"), text.rfind("}") + 1
        return json.loads(text[s:e])
