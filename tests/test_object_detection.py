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

PROMPT = (
    "이미지에서 보이는 모든 물체(object)를 감지하세요. 로봇 팔과 그리퍼는 무시하세요.\n\n"
    "감지 대상 예시: 캔, 컵, 상자, 병, 작은 물건 등 테이블 위 물체\n\n"
    "각 물체에 대해 다음 JSON 배열 형식으로만 답하세요:\n"
    "[\n"
    "  {\n"
    '    "label": "sprite_can",\n'
    '    "confidence": 0.90,\n'
    '    "bbox_norm": [cx, cy, w, h],\n'
    '    "color": "green",\n'
    '    "notes": "물체 특징 한 문장"\n'
    "  }\n"
    "]\n\n"
    "bbox_norm: 이미지 크기로 정규화된 [중심x, 중심y, 너비, 높이] (0~1 범위)\n"
    "물체가 없으면 빈 배열 [] 반환"
)

img_dir = Path(__file__).parent.parent / "data" / "base_images"
results = []

print("=" * 60)
print("  Gemini ER 1.5 — Object Detection (gripper 제외)")
print("=" * 60)

for i in range(1, 10):
    path = img_dir / f"base{i}.jpg"
    if not path.exists():
        print(f"base{i}: 파일 없음")
        continue

    pil = Image.open(path)
    t0  = time.time()

    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model="gemini-robotics-er-1.5-preview",
                contents=[PROMPT, pil],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 40 * (attempt + 1)
                print(f"  rate limit → {wait}s 대기 후 재시도 ({attempt+1}/5)...")
                time.sleep(wait)
            elif "503" in str(e) or "UNAVAILABLE" in str(e):
                wait = 20 * (attempt + 1)
                print(f"  서버 과부하 → {wait}s 대기 후 재시도 ({attempt+1}/5)...")
                time.sleep(wait)
            else:
                raise
    elapsed = time.time() - t0

    text  = resp.text.strip()
    start = text.find("[")
    end   = text.rfind("]") + 1
    try:
        objects = json.loads(text[start:end])
    except Exception:
        objects = []
        print(f"  base{i}: JSON 파싱 실패 → {text[:80]}")

    entry = {"image": f"base{i}", "latency_s": round(elapsed, 1), "objects": objects}
    results.append(entry)

    print(f"\nbase{i}: {len(objects)}개 물체 감지  ({elapsed:.1f}s)")
    for obj in objects:
        label = obj.get("label", "?")
        conf  = obj.get("confidence", "?")
        bbox  = obj.get("bbox_norm", [])
        notes = obj.get("notes", "")[:50]
        print(f"  [{label}] conf={conf}  bbox={bbox}")
        print(f"    {notes}")

out = Path(__file__).parent.parent / "results" / "auto_eval" / "object_detection.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(f"\n결과 저장: {out}")
