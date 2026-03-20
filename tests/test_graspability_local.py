"""
Cosmos-Reason2 — Graspability Test (Local VLM)
base1~9 이미지에서 그리퍼가 캔을 잡을 수 있는 상태인지 판단

사용법:
  python tests/test_graspability_local.py                    # 2B (기본)
  python tests/test_graspability_local.py --model 8B
"""
import sys, json, time, argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
IMG_DIR = Path(__file__).parent.parent / "data" / "base_images"

PROMPT = (
    "이미지에서 로봇 그리퍼(집게)와 스프라이트 캔을 관찰하세요.\n\n"
    "판단 기준:\n"
    "1. 그리퍼가 캔 바로 위 또는 캔을 집을 수 있는 위치에 있는가?\n"
    "2. 그리퍼가 캔을 향해 내려오면 잡을 수 있는가?\n"
    "3. 그리퍼와 캔 사이의 거리/정렬 상태는?\n\n"
    "다음 JSON 형식으로만 답하세요:\n"
    "{\n"
    '  "can_visible": true,\n'
    '  "gripper_visible": true,\n'
    '  "graspable": true or false,\n'
    '  "confidence": 0.85,\n'
    '  "gripper_position": "캔 위 / 캔 옆 / 멀리 / 캔 잡는 중",\n'
    '  "reason": "판단 근거 한 문장"\n'
    "}"
)

def load_model(model_size="2B"):
    model_path = str(MODELS_DIR / f"Cosmos-Reason2-{model_size}")
    print(f"모델 로딩: {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if device == "cpu":
        model = model.to(device)

    processor = AutoProcessor.from_pretrained(model_path)
    print(f"모델 로드 완료 (device={device}, dtype={dtype})")
    return model, processor, device


def run_inference(model, processor, device, image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
        )

    # 입력 토큰 이후만 디코딩
    generated = output_ids[:, inputs.input_ids.shape[1]:]
    return processor.decode(generated[0], skip_special_tokens=True)


def parse_json(text):
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {"raw": text}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {"raw": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="2B", choices=["2B", "8B"])
    args = parser.parse_args()

    model_size = args.model
    model, processor, device = load_model(model_size)

    results = []
    print("\n" + "=" * 60)
    print(f"  Cosmos-Reason2-{model_size} — Graspability Test (base1~9)")
    print("=" * 60)

    for i in range(1, 10):
        path = IMG_DIR / f"base{i}.jpg"
        if not path.exists():
            print(f"base{i}: 파일 없음")
            continue

        t0 = time.time()
        raw_text = run_inference(model, processor, device, path)
        elapsed = time.time() - t0

        result = parse_json(raw_text)
        result["image"] = f"base{i}"
        result["latency_s"] = round(elapsed, 1)
        result["raw_output"] = raw_text
        results.append(result)

        graspable = result.get("graspable", "?")
        conf = result.get("confidence", "?")
        pos = result.get("gripper_position", "?")
        reason = result.get("reason", "")[:80]
        mark = "✅" if graspable is True else ("❌" if graspable is False else "❓")

        print(f"\nbase{i}: {mark}  graspable={graspable}  confidence={conf}  ({elapsed:.1f}s)")
        print(f"  위치: {pos}")
        print(f"  근거: {reason}")

    # 요약
    print("\n" + "=" * 60)
    print("  요약")
    print("=" * 60)
    graspable_list = [r for r in results if r.get("graspable") is True]
    not_graspable = [r for r in results if r.get("graspable") is False]
    uncertain = [r for r in results if r.get("graspable") not in (True, False)]
    print(f"  ✅ 집을 수 있음: {[r['image'] for r in graspable_list]}")
    print(f"  ❌ 집을 수 없음: {[r['image'] for r in not_graspable]}")
    if uncertain:
        print(f"  ❓ 판단 불가:   {[r['image'] for r in uncertain]}")
    if results:
        confs = [r.get("confidence", 0) for r in results if isinstance(r.get("confidence"), (int, float))]
        if confs:
            print(f"  평균 confidence: {sum(confs)/len(confs):.2f}")

    # 저장
    out = Path(__file__).parent.parent / "results" / "auto_eval" / f"graspability_{model_size}_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
