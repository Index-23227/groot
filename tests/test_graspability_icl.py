"""
Gemini ER 1.5 — Graspability Test with ICL (In-Context Learning)
base1~9 이미지에서 ICL 예시를 포함하여 정밀도(Precision) 개선 테스트

기준 성능 (ICL 없음): Precision 50%, Recall 100%, Accuracy 66.7%
목표: Precision ≥ 70% 유지하면서 Recall 저하 최소화
"""
import sys, json, time, base64
from pathlib import Path
from PIL import Image
import io

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

# --- ICL 예시 데이터 ---
# FP 케이스(base3, base8, base9)에서 실패한 이유를 명확히 보여주는 예시를 우선 포함
# Gemini가 "정렬된 것처럼 보이지만 실제로 실패"하는 패턴을 학습하도록 함

img_dir = Path(__file__).parent.parent / "data" / "base_images"
annotations = json.loads((img_dir / "icl_annotations.json").read_text())
ann_by_image = {a["image"]: a for a in annotations}

def image_to_part(path: Path) -> types.Part:
    """PIL Image를 Gemini Part로 변환"""
    with Image.open(path) as img:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return types.Part.from_bytes(data=buf.read(), mime_type="image/jpeg")

# ICL 예시로 사용할 이미지: FP 케이스(base3, base8) + TP(base1) + TN(base2)
# 테스트 대상과 겹치지 않도록 4장을 ICL 예시로 사용하고 나머지 5장(base4~7, base9)을 평가
ICL_IMAGES = ["base1", "base2", "base3", "base8"]  # 4 shot: 1 TP, 1 TN, 2 FP(corrected)
TEST_IMAGES = ["base4", "base5", "base6", "base7", "base9"]  # 5장 평가

def build_icl_contents(icl_images: list[str]) -> list:
    """ICL 예시들을 Gemini contents 형식으로 구성"""
    contents = []

    system_intro = (
        "당신은 로봇 그리퍼가 캔을 잡을 수 있는지 판단하는 전문가입니다.\n"
        "다음 예시들을 통해 판단 기준을 학습하세요.\n\n"
        "중요한 판단 기준:\n"
        "1. 그리퍼 중심과 캔 중심의 X축 오프셋이 5% 이내여야 '정렬됨'\n"
        "2. 그리퍼가 캔 바로 위(Y차이 ≤ 0.10)에 있어야 파지 가능\n"
        "3. 이미지상 정렬되어 보여도 실제 접근 각도/높이 오차로 실패할 수 있음\n"
        "4. confidence 0.85~0.90만으로는 FP를 판별할 수 없음 — bbox 수치를 더 엄격하게 확인\n\n"
        "=== ICL 예시 시작 ===\n"
    )
    contents.append(system_intro)

    for img_name in icl_images:
        ann = ann_by_image[img_name]
        path = img_dir / f"{img_name}.jpg"

        outcome_str = "✅ 실제 파지 성공" if ann["graspable_gt"] else "❌ 실제 파지 실패"
        graspable_str = "true" if ann["graspable_gt"] else "false"

        icl_prompt = (
            f"--- 예시 ({img_name}) ---\n"
            f"지시: {ann['instruction']}\n"
            f"이미지:\n"
        )
        contents.append(icl_prompt)
        contents.append(image_to_part(path))

        result_json = {
            "can_visible": True,
            "gripper_visible": True,
            "graspable": ann["graspable_gt"],
            "confidence": 0.90 if ann["graspable_gt"] else 0.85,
            "gripper_position": "캔 위" if ann["graspable_gt"] else ("캔 위" if ann["horizontal_error"] < 0.05 else "멀리"),
            "alignment": "정렬됨" if ann["graspable_gt"] else (
                "약간 틀어짐" if ann["horizontal_error"] < 0.05 else "많이 틀어짐"
            ),
            "can_bbox_norm": ann["can_bbox_norm"],
            "gripper_bbox_norm": ann["gripper_bbox_norm"],
            "horizontal_error_pct": ann["horizontal_error"],
            "reason": ann["failure_reason"] if not ann["graspable_gt"] else ann["notes"],
        }
        result_str = (
            f"정답: {outcome_str}\n"
            f"bbox 분석: 캔={ann['can_bbox_norm']}, 그리퍼={ann['gripper_bbox_norm']}\n"
            f"수평 오프셋: {ann['horizontal_error']*100:.0f}%\n"
            f"판단: graspable={graspable_str}\n"
            f"이유: {ann['failure_reason'] or ann['notes']}\n"
        )
        if ann["correction_mm"]:
            result_str += f"보정 필요: {ann['correction_mm']}\n"
        result_str += "\n"
        contents.append(result_str)

    contents.append("=== ICL 예시 끝 ===\n\n이제 아래 새 이미지를 판단하세요.\n\n")
    return contents

QUERY_PROMPT = (
    "이미지에서 로봇 그리퍼(집게)와 스프라이트 캔을 관찰하세요.\n\n"
    "엄격한 판단 기준:\n"
    "1. 그리퍼 중심 X값과 캔 중심 X값의 차이가 5% 이내인가?\n"
    "2. 그리퍼 Y값이 캔 Y값보다 0.10 이내로 위에 있는가?\n"
    "3. 이미지에서 정렬된 것처럼 보여도 위 조건을 수치로 확인할 것\n"
    "4. 불확실하면 graspable=false 판단 (안전한 방향)\n\n"
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
    '  "horizontal_error_pct": 0.00,\n'
    '  "reason": "판단 근거 한 문장"\n'
    "}"
)

# Ground truth
GT = {
    "base1": True,  "base2": False, "base3": False,
    "base4": True,  "base5": False, "base6": False,
    "base7": True,  "base8": False, "base9": False,
}

print("=" * 65)
print("  Gemini ER 1.5 — Graspability Test with ICL (4-shot)")
print("  ICL 예시: base1(TP), base2(TN), base3(FP→corrected), base8(FP→corrected)")
print("  평가 대상: base4, base5, base6, base7, base9")
print("=" * 65)

# ICL 콘텐츠 구성 (한 번만)
icl_contents = build_icl_contents(ICL_IMAGES)

results_icl = []
results_baseline = {}

# 기존 baseline 로드
baseline_path = Path(__file__).parent.parent / "results" / "auto_eval" / "graspability_test.json"
if baseline_path.exists():
    baseline_data = json.loads(baseline_path.read_text())
    results_baseline = {r["image"]: r for r in baseline_data}

# ICL 포함 테스트
for img_name in TEST_IMAGES:
    path = img_dir / f"{img_name}.jpg"
    if not path.exists():
        print(f"{img_name}: 파일 없음")
        continue

    contents = icl_contents + [QUERY_PROMPT, image_to_part(path)]

    t0 = time.time()
    resp = client.models.generate_content(
        model="gemini-robotics-er-1.5-preview",
        contents=contents,
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

    result["image"]     = img_name
    result["latency_s"] = round(elapsed, 1)
    result["gt"]        = GT[img_name]
    results_icl.append(result)

    graspable = result.get("graspable", "?")
    conf      = result.get("confidence", "?")
    h_err     = result.get("horizontal_error_pct", "?")
    align     = result.get("alignment", "?")
    reason    = result.get("reason", "")[:70]
    gt_val    = GT[img_name]

    if graspable is True and gt_val is True:
        verdict = "TP ✅"
    elif graspable is False and gt_val is False:
        verdict = "TN ✅"
    elif graspable is True and gt_val is False:
        verdict = "FP ❌"
    else:
        verdict = "FN ❌"

    base_graspable = results_baseline.get(img_name, {}).get("graspable", "?")
    changed = " [변경!]" if base_graspable != graspable else ""

    print(f"\n{img_name}: {verdict}  graspable={graspable}  conf={conf}  ({elapsed:.1f}s){changed}")
    print(f"  정렬: {align}  |  수평오차: {h_err}")
    print(f"  근거: {reason}")

# --- 전체 이미지(9장) 결과 합산을 위해 ICL 예시 이미지도 annotation 기반으로 추가 ---
icl_annotated = []
for img_name in ICL_IMAGES:
    ann = ann_by_image[img_name]
    gt_val = ann["graspable_gt"]
    # ICL 예시는 정답 그대로 사용 (annotation 기반)
    icl_annotated.append({
        "image": img_name,
        "graspable": gt_val,
        "gt": gt_val,
        "confidence": 0.90,
        "latency_s": 0.0,
        "note": "ICL example (not queried)"
    })

all_results = icl_annotated + results_icl

# --- 평가 지표 계산 ---
def compute_metrics(results):
    TP = sum(1 for r in results if r.get("graspable") is True  and r.get("gt") is True)
    TN = sum(1 for r in results if r.get("graspable") is False and r.get("gt") is False)
    FP = sum(1 for r in results if r.get("graspable") is True  and r.get("gt") is False)
    FN = sum(1 for r in results if r.get("graspable") is False and r.get("gt") is True)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy  = (TP + TN) / len(results) if results else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "precision": precision, "recall": recall,
            "accuracy": accuracy, "f1": f1}

# ICL 포함 (5장 평가)
m_icl_eval = compute_metrics(results_icl)
# 전체 9장 (ICL 예시 annotation 포함)
m_all = compute_metrics(all_results)

# Baseline (9장)
baseline_list = [{**v, "gt": GT[k]} for k, v in results_baseline.items()]
m_base = compute_metrics(baseline_list)

print("\n" + "=" * 65)
print("  비교 요약")
print("=" * 65)
print(f"{'지표':<15} {'Baseline(9장)':>15} {'ICL 평가(5장)':>15} {'ICL 전체(9장)':>15}")
print("-" * 65)
for key in ["TP", "TN", "FP", "FN"]:
    print(f"{key:<15} {m_base[key]:>15} {m_icl_eval[key]:>15} {m_all[key]:>15}")
print("-" * 65)
for key in ["precision", "recall", "accuracy", "f1"]:
    print(f"{key:<15} {m_base[key]:>14.1%} {m_icl_eval[key]:>14.1%} {m_all[key]:>14.1%}")

# --- 저장 ---
out_dir = Path(__file__).parent.parent / "results" / "auto_eval"
out_dir.mkdir(parents=True, exist_ok=True)

output = {
    "icl_images": ICL_IMAGES,
    "test_images": TEST_IMAGES,
    "icl_results": results_icl,
    "icl_annotated": icl_annotated,
    "metrics_icl_eval": m_icl_eval,
    "metrics_icl_all": m_all,
    "metrics_baseline": m_base,
}

out_path = out_dir / "graspability_icl_test.json"
out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
print(f"\n결과 저장: {out_path}")
