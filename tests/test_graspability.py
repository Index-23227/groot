"""
Gemini ER 1.5 — Graspability Test
base1~9 이미지에서 그리퍼가 캔을 잡을 수 있는 상태인지 판단
"""
import sys, json, time
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

# API 키 로드
token = Path(__file__).parent.parent / "token"
google_key = ""
for line in token.read_text().splitlines():
    line = line.strip()
    if line.startswith("AIza"):
        google_key = line
        break

from google import genai
from google.genai import types

client = genai.Client(api_key=google_key)

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
    '  "graspable": true,\n'
    '  "confidence": 0.85,\n'
    '  "gripper_position": "캔 위 / 캔 옆 / 멀리 / 캔 잡는 중",\n'
    '  "alignment": "정렬됨 / 약간 틀어짐 / 많이 틀어짐",\n'
    '  "can_bbox_norm": [cx, cy, w, h],\n'
    '  "gripper_bbox_norm": [cx, cy, w, h],\n'
    '  "reason": "판단 근거 한 문장"\n'
    "}"
)

img_dir = Path(__file__).parent.parent / "data" / "base_images"
results = []

print("=" * 60)
print("  Gemini ER 1.5 — Graspability Test (base1~9)")
print("=" * 60)

for i in range(1, 10):
    path = img_dir / f"base{i}.jpg"
    if not path.exists():
        print(f"base{i}: 파일 없음")
        continue

    pil = Image.open(path)
    t0  = time.time()

    resp = client.models.generate_content(
        model="gemini-robotics-er-1.5-preview",
        contents=[PROMPT, pil],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    elapsed = time.time() - t0

    text  = resp.text.strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    try:
        result = json.loads(text[start:end])
    except Exception:
        result = {"raw": text}

    result["image"]     = f"base{i}"
    result["latency_s"] = round(elapsed, 1)
    results.append(result)

    graspable = result.get("graspable", "?")
    conf      = result.get("confidence", "?")
    pos       = result.get("gripper_position", "?")
    align     = result.get("alignment", "?")
    reason    = result.get("reason", "")[:70]
    mark      = "✅" if graspable else "❌"

    print(f"\nbase{i}: {mark}  graspable={graspable}  confidence={conf}  ({elapsed:.1f}s)")
    print(f"  위치: {pos}  |  정렬: {align}")
    print(f"  근거: {reason}")

# 요약
print("\n" + "=" * 60)
print("  요약")
print("=" * 60)
graspable_list = [r for r in results if r.get("graspable") is True]
not_graspable  = [r for r in results if r.get("graspable") is False]
print(f"  집을 수 있음:  {[r['image'] for r in graspable_list]}")
print(f"  집을 수 없음:  {[r['image'] for r in not_graspable]}")
avg_conf = sum(r.get("confidence", 0) for r in results) / len(results) if results else 0
print(f"  평균 confidence: {avg_conf:.2f}")

# 저장
out = Path(__file__).parent.parent / "results" / "auto_eval" / "graspability_test.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(f"\n결과 저장: {out}")
