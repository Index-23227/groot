"""
Gemini ER 1.5 — Object Detection Only (gripper 제외)
base1~9 이미지에서 객체(캔, 박스 등)만 bbox 감지
그리퍼 bbox 요청 제거 → 할루시네이션 감소 목적
"""
import sys, json, time
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

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

INSTRUCTION = "스프라이트 캔을 집어라"

def make_prompt(instruction: str) -> str:
    return (
        f"지시: {instruction}\n\n"
        "위 지시에서 언급된 물체 하나만 이미지에서 찾으세요. "
        "로봇 팔과 그리퍼는 무시하세요.\n\n"
        "다음 JSON 형식으로만 답하세요 (물체가 없으면 null):\n"
        "{\n"
        '  "label": "sprite_can",\n'
        '  "visible": true,\n'
        '  "confidence": 0.90,\n'
        '  "bbox_norm": [cx, cy, w, h],\n'
        '  "notes": "물체 특징 한 문장"\n'
        "}\n\n"
        "bbox_norm: 이미지 크기로 정규화된 [중심x, 중심y, 너비, 높이] (0~1 범위)"
    )

img_dir = Path(__file__).parent.parent / "data" / "base_images"
results = []

print("=" * 60)
print(f"  Gemini ER 1.5 — Target Object Detection")
print(f"  지시: {INSTRUCTION}")
print("=" * 60)

def query_gemini(pil_img, prompt):
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model="gemini-robotics-er-1.5-preview",
                contents=[prompt, pil_img],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            return resp
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 15
                print(f"  rate limit → {wait}s 대기 후 재시도 ({attempt+1}/3)...")
                time.sleep(wait)
            elif "503" in str(e) or "UNAVAILABLE" in str(e):
                wait = 10
                print(f"  서버 과부하 → {wait}s 대기 후 재시도 ({attempt+1}/3)...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("최대 재시도 초과")

for i in range(1, 10):
    path = img_dir / f"base{i}.jpg"
    if not path.exists():
        print(f"base{i}: 파일 없음")
        continue

    pil    = Image.open(path)
    prompt = make_prompt(INSTRUCTION)
    t0     = time.time()
    resp   = query_gemini(pil, prompt)
    elapsed = time.time() - t0

    text  = resp.text.strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    try:
        obj = json.loads(text[start:end])
    except Exception:
        obj = None
        print(f"  base{i}: JSON 파싱 실패 → {text[:80]}")

    entry = {"image": f"base{i}", "latency_s": round(elapsed, 1),
             "instruction": INSTRUCTION, "object": obj}
    results.append(entry)

    if obj:
        visible = obj.get("visible", True)
        conf    = obj.get("confidence", "?")
        bbox    = obj.get("bbox_norm", [])
        notes   = obj.get("notes", "")[:60]
        mark    = "✅" if visible else "❌"
        print(f"\nbase{i}: {mark} visible={visible}  conf={conf}  ({elapsed:.1f}s)")
        print(f"  bbox={bbox}")
        print(f"  {notes}")
    else:
        print(f"\nbase{i}: ❌ 감지 실패  ({elapsed:.1f}s)")

out = Path(__file__).parent.parent / "results" / "auto_eval" / "object_detection.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(f"\n결과 저장: {out}")
print(f"총 {sum(1 for r in results if r.get('object') and r['object'].get('visible'))}장에서 타겟 감지")
