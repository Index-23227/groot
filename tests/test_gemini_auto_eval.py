"""
Gemini Embodied Reasoning 자동 평가 파이프라인
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

흐름:
  1. 카메라로 씬 캡처
  2. GPT-4o가 씬을 보고 → 적합한 embodied reasoning 태스크 + instruction 결정
  3. Gemini ER에게 이미지 + instruction 전달
  4. GPT-4o가 Gemini 응답을 평가 (pass/fail + 구체적 피드백)
  5. 결과 시각화 (3단 패널)

사용법:
  python tests/test_gemini_auto_eval.py               # 인터랙티브 (카메라)
  python tests/test_gemini_auto_eval.py --image scene.jpg
  python tests/test_gemini_auto_eval.py --rounds 5    # 5번 연속 자동 테스트
  python tests/test_gemini_auto_eval.py --no-eval     # Gemini 응답만 보기 (평가 스킵)

API 키 파일:
  groot/token           → Google API key (Gemini)
  groot/openai_token    → OpenAI API key (GPT-4o)
"""

import os, sys, json, time, re, argparse, io
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
from utils.doosan_recorder import CameraCapture

# ══════════════════════════════════════════════════════════════════════════════
# 클라이언트 초기화
# ══════════════════════════════════════════════════════════════════════════════

def load_api_keys():
    """groot/token 파일에서 두 키를 읽는다.
    1번 줄: Google API key (AIza...)
    2번 줄: OpenAI API key  (sk-proj-...)
    """
    root = Path(__file__).parent.parent

    google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    token_path = root / "token"
    if token_path.exists():
        lines = [l.strip() for l in token_path.read_text().splitlines() if l.strip()]
        for line in lines:
            if line.startswith("AIza") and not google_key:
                google_key = line
                os.environ["GOOGLE_API_KEY"] = google_key
            elif line.startswith("sk-") and not openai_key:
                openai_key = line
                os.environ["OPENAI_API_KEY"] = openai_key

    if not google_key:
        raise EnvironmentError("Google API key 없음 (AIza...). groot/token 1번 줄 확인.")
    if not openai_key:
        raise EnvironmentError("OpenAI API key 없음 (sk-...). groot/token 2번 줄 확인.")

    return google_key, openai_key


def init_clients(google_key, openai_key, gemini_model, gpt_model):
    from google import genai as google_genai
    from openai import OpenAI

    gemini = google_genai.Client(api_key=google_key)
    gpt = OpenAI(api_key=openai_key)
    return gemini, gpt


# ══════════════════════════════════════════════════════════════════════════════
# 이미지 유틸
# ══════════════════════════════════════════════════════════════════════════════

def ndarray_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))

def ndarray_to_b64jpeg(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="JPEG", quality=90)
    import base64
    return base64.b64encode(buf.getvalue()).decode()

def parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = re.sub(r"```\s*$", "", text).strip()
    try:
        s, e = text.find("{"), text.rfind("}") + 1
        return json.loads(text[s:e])
    except Exception:
        return {"raw": text, "parse_error": True}


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: GPT-4o — 씬 분석 + instruction 생성
# ══════════════════════════════════════════════════════════════════════════════

SCENE_ANALYZER_PROMPT = """
당신은 물리 로봇 embodied reasoning을 엄격하게 평가하는 전문가입니다.
이미지는 Doosan E0509 6-DOF 로봇 팔 (평행 그리퍼, 최대 벌림 120mm)의 작업공간입니다.

씬을 보고 아래 JSON을 작성하세요. instruction_for_gemini는 반드시 다음 항목을
모두 요구하는 엄격한 프롬프트여야 합니다:
  - 모든 물체의 bbox_norm [cx,cy,w,h] (0~1)
  - 각 물체의 grasp_point_norm [cx,cy]
  - approach_direction (from_top / from_left / from_right / from_front)
  - gripper_opening_mm (실제 mm 추정값)
  - pregrasp_offset_mm {x,y,z} (그리퍼 접근 전 오프셋)
  - collision_risk (low/medium/high) + 이유
  - pick → lift → move → place 각 단계의 좌표
  - 물리적 불가능한 경우 명시적으로 "infeasible" 표기

JSON만 답하세요 (코드블록 없이):
{
  "scene_summary": "씬 한 줄 요약",
  "detected_objects": [{"label": "...", "shape": "cylindrical/box/flat", "size_estimate_mm": 80}],
  "chosen_task": "affordance | action_plan | collision_check | placement_check",
  "task_reason": "이 태스크를 선택한 이유",
  "instruction_for_gemini": "Gemini에게 줄 완전한 프롬프트. bbox/grasp_point/approach/gripper_mm/pregrasp_offset/collision_risk/단계별 좌표를 모두 JSON으로 반환하도록 명시",
  "expected_output_keys": ["bbox_norm","grasp_point_norm","approach_direction","gripper_opening_mm","pregrasp_offset_mm","collision_risk","action_steps"],
  "evaluation_criteria": [
    "bbox_norm 4개 값 모두 존재하고 0~1 범위인가",
    "grasp_point_norm이 bbox 내부에 위치하는가",
    "gripper_opening_mm이 물체 크기보다 크고 120mm 이하인가",
    "approach_direction이 씬의 장애물을 고려했는가",
    "pregrasp_offset_mm z값이 양수(위에서 접근)인가",
    "collision_risk 판단 근거가 구체적인가",
    "action_steps에 pick/lift/move/place가 모두 포함되는가"
  ],
  "difficulty": "easy/medium/hard",
  "task_category": "perception/manipulation/language/safety/sequential"
}
"""


def step1_gpt_design(gpt, gpt_model: str, img_rgb: np.ndarray) -> dict:
    """GPT-4o가 씬을 보고 테스트 태스크 + instruction 설계"""
    print(f"\n[Step 1] GPT-4o가 씬 분석 중... ({gpt_model})")
    t0 = time.time()

    img_b64 = ndarray_to_b64jpeg(img_rgb)
    resp = gpt.chat.completions.create(
        model=gpt_model,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": SCENE_ANALYZER_PROMPT},
            ]
        }]
    )
    latency = round(time.time() - t0, 2)
    result = parse_json(resp.choices[0].message.content)
    result["_latency_s"] = latency

    if not result.get("parse_error"):
        print(f"  씬: {result.get('scene_summary', '?')}")
        print(f"  태스크: {result.get('chosen_task', '?')}  ({result.get('difficulty', '?')})")
        print(f"  이유: {result.get('task_reason', '?')}")
        print(f"  Instruction: {result.get('instruction_for_gemini', '')[:100]}...")
        print(f"  latency: {latency}s")
    else:
        print(f"  [!] 파싱 실패: {result.get('raw','')[:200]}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Gemini — embodied reasoning 실행
# ══════════════════════════════════════════════════════════════════════════════

GEMINI_SYSTEM_SUFFIX = """

응답은 반드시 JSON만 하세요 (코드블록 없이).
bbox가 있으면 0~1 정규화 [cx, cy, w, h] 형식으로 표현하세요.
"""

def step2_gemini_reason(gemini, gemini_model: str,
                         img_rgb: np.ndarray, instruction: str) -> dict:
    """Gemini ER에게 이미지 + instruction 전달"""
    print("\n[Step 2] Gemini가 embodied reasoning 중...")
    t0 = time.time()

    pil = ndarray_to_pil(img_rgb)
    full_prompt = instruction + GEMINI_SYSTEM_SUFFIX

    resp = gemini.models.generate_content(
        model=gemini_model,
        contents=[full_prompt, pil]
    )
    latency = round(time.time() - t0, 2)
    result = parse_json(resp.text)
    result["_latency_s"] = latency
    result["_raw"] = resp.text[:500]

    if not result.get("parse_error"):
        print(f"  응답 키: {[k for k in result if not k.startswith('_')]}")
        print(f"  latency: {latency}s")
    else:
        print(f"  [!] 파싱 실패 — raw: {resp.text[:300]}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: GPT-4o — Gemini 응답 평가
# ══════════════════════════════════════════════════════════════════════════════

def build_eval_prompt(design: dict, gemini_result: dict) -> str:
    criteria = design.get("evaluation_criteria", [])
    return f"""
당신은 물리 로봇 AI를 엄격하게 평가하는 심사위원입니다.
아래 기준은 실제 로봇 제어에 필요한 최소 요건입니다. 누락·불합리한 값은 무조건 fail입니다.

[평가 기준 — 각 항목 15점, 총 100점]
{json.dumps(criteria, ensure_ascii=False, indent=2)}

[Gemini가 받은 instruction]
{design.get("instruction_for_gemini", "")}

[Gemini 응답]
{json.dumps({k: v for k, v in gemini_result.items() if not k.startswith('_')},
            ensure_ascii=False, indent=2)}

[원본 씬 이미지도 첨부됨 — bbox 좌표가 실제 물체 위치와 일치하는지 직접 확인하세요]

채점 규칙:
- bbox_norm 값 범위 오류(0~1 벗어남) → 즉시 0점
- grasp_point가 bbox 밖에 있음 → -10점
- gripper_opening_mm > 120 → 물리 불가능, 즉시 fail
- pregrasp_offset z < 0 → 물리 불가능, 즉시 fail
- 키 자체가 응답에 없으면 해당 기준 0점
- 모호한 서술("적절히", "오른쪽으로") → 구체적 수치 없으면 -5점

JSON만 답하세요:
{{
  "overall": "pass/partial/fail",
  "score": 0,
  "criteria_results": [
    {{
      "criterion": "기준 설명",
      "result": "pass/fail",
      "score": 15,
      "comment": "구체적 근거 (값 인용)"
    }}
  ],
  "missing_fields": ["응답에서 누락된 필드 목록"],
  "physical_violations": ["물리 법칙 위반 항목"],
  "strengths": ["실제로 잘 된 점 (구체적으로)"],
  "weaknesses": ["실제로 부족한 점 (구체적으로)"],
  "embodied_reasoning_quality": "excellent/good/fair/poor",
  "robot_usable": true,
  "summary": "한 문장 평가"
}}
"""

def step3_gpt_evaluate(gpt, gpt_model: str,
                        img_rgb: np.ndarray,
                        design: dict, gemini_result: dict) -> dict:
    """GPT-4o가 Gemini의 응답을 평가"""
    print(f"\n[Step 3] GPT-4o가 Gemini 응답 평가 중... ({gpt_model})")
    t0 = time.time()

    img_b64 = ndarray_to_b64jpeg(img_rgb)
    eval_prompt = build_eval_prompt(design, gemini_result)

    resp = gpt.chat.completions.create(
        model=gpt_model,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": eval_prompt},
            ]
        }]
    )
    latency = round(time.time() - t0, 2)
    result = parse_json(resp.choices[0].message.content)
    result["_latency_s"] = latency

    if not result.get("parse_error"):
        overall = result.get("overall", "?")
        score = result.get("score", "?")
        quality = result.get("embodied_reasoning_quality", "?")
        emoji = {"pass": "✅", "partial": "⚠️", "fail": "❌"}.get(overall, "?")
        print(f"  {emoji} {overall.upper()}  Score: {score}/100  Quality: {quality}")
        print(f"  요약: {result.get('summary', '')}")
        for c in result.get("criteria_results", []):
            icon = "✅" if c["result"] == "pass" else "❌"
            print(f"    {icon} {c['criterion'][:40]}: {c['comment'][:50]}")
        print(f"  latency: {latency}s")
    else:
        print(f"  [!] 평가 파싱 실패")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 시각화 — 발표용 대시보드
# ══════════════════════════════════════════════════════════════════════════════
#
#  ┌────────────────────┬──────────────────────────────────┐
#  │                    │  HEADER: 제목 + 모델 정보         │
#  │  원본 씬            ├──────────────────────────────────┤
#  │  + bbox 오버레이    │  GPT-4o: 씬 분석 + 태스크         │
#  │                    ├──────────────────────────────────┤
#  │                    │  Gemini ER: 응답 필드 요약         │
#  ├────────────────────┼──────────────────────────────────┤
#  │  점수 게이지        │  기준별 채점표 (항목별 pass/fail)  │
#  └────────────────────┴──────────────────────────────────┘

FONT  = cv2.FONT_HERSHEY_SIMPLEX
FONTB = cv2.FONT_HERSHEY_DUPLEX
PALETTE = [(52,211,153),(251,146,60),(96,165,250),(232,121,249),(251,191,36),(34,211,238)]
C_BG      = (15, 17, 26)
C_CARD    = (28, 32, 48)
C_BORDER  = (55, 65, 90)
C_TEXT    = (220, 225, 235)
C_MUTED   = (120, 130, 150)
C_GREEN   = (52, 211, 153)
C_ORANGE  = (251, 146, 60)
C_RED     = (99,  75, 239)
C_BLUE    = (96, 165, 250)
C_YELLOW  = (251, 191,  36)

DW, DH = 1280, 720   # 대시보드 고정 크기 (16:9)


def _t(img, text, org, scale=0.45, fg=C_TEXT, bold=False, thickness=1):
    f = FONTB if bold else FONT
    cv2.putText(img, str(text), (int(org[0]), int(org[1])),
                f, scale, fg, thickness, cv2.LINE_AA)

def _rect(img, x1, y1, x2, y2, color, filled=True, radius=0, t=1):
    if filled:
        cv2.rectangle(img, (x1,y1), (x2,y2), color, -1)
    else:
        cv2.rectangle(img, (x1,y1), (x2,y2), color, t)

def _card(img, x1, y1, x2, y2, title="", title_color=C_BLUE):
    _rect(img, x1, y1, x2, y2, C_CARD)
    _rect(img, x1, y1, x2, y2, C_BORDER, filled=False, t=1)
    if title:
        _rect(img, x1, y1, x2, y1+22, title_color)
        _t(img, title, (x1+6, y1+15), scale=0.42, fg=(0,0,0), bold=True)
    return y1 + (24 if title else 4)

def _wrap(text, max_chars):
    words = str(text).split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines

def _score_gauge(img, cx, cy, r, score, overall):
    color = C_GREEN if overall=="pass" else (C_ORANGE if overall=="partial" else C_RED)
    # 배경 원
    cv2.circle(img, (cx,cy), r, C_BORDER, 2)
    # 호 그리기 (0~360 * score/100)
    angle = int(360 * score / 100)
    for a in range(-90, -90 + angle, 3):
        rad = a * 3.14159 / 180
        x = int(cx + (r-4) * np.cos(rad))
        y = int(cy + (r-4) * np.sin(rad))
        cv2.circle(img, (x,y), 3, color, -1)
    # 점수 텍스트
    label = f"{score}"
    (tw,th),_ = cv2.getTextSize(label, FONTB, 1.1, 2)
    cv2.putText(img, label, (cx-tw//2, cy+th//2), FONTB, 1.1, color, 2, cv2.LINE_AA)
    verdict = {"pass":"PASS","partial":"PARTIAL","fail":"FAIL"}.get(overall,"?")
    (vw,vh),_ = cv2.getTextSize(verdict, FONT, 0.45, 1)
    cv2.putText(img, verdict, (cx-vw//2, cy+th//2+20), FONT, 0.45, color, 1, cv2.LINE_AA)


def build_dashboard(frame_bgr, design, gemini_result, eval_result, _=None):
    dash = np.full((DH, DW, 3), C_BG, dtype=np.uint8)
    H_orig, W_orig = frame_bgr.shape[:2]

    overall  = eval_result.get("overall", "?")
    score    = int(eval_result.get("score", 0))
    quality  = eval_result.get("embodied_reasoning_quality", "?")
    o_color  = C_GREEN if overall=="pass" else (C_ORANGE if overall=="partial" else C_RED)

    # ── HEADER ────────────────────────────────────────────────────────────────
    _rect(dash, 0, 0, DW, 44, (22, 27, 42))
    _t(dash, "Gemini Embodied Reasoning  Eval", (14,28), scale=0.65, fg=C_BLUE, bold=True, thickness=1)
    _t(dash, f"Gemini: gemini-robotics-er-1.5-preview   Judge: GPT-4o   Task: {design.get('chosen_task','?')}",
       (14, 40), scale=0.35, fg=C_MUTED)
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    (tw,_),_ = cv2.getTextSize(ts_str, FONT, 0.38, 1)
    _t(dash, ts_str, (DW-tw-10, 28), scale=0.38, fg=C_MUTED)

    # ── LEFT: 씬 이미지 + bbox (top-left, 580x400) ───────────────────────────
    IMG_X, IMG_Y, IMG_W, IMG_H = 10, 54, 580, 400
    scene_vis = frame_bgr.copy()
    # bbox 오버레이
    sh, sw = scene_vis.shape[:2]
    objects = []
    for key in ["objects","affordances","steps","action_steps"]:
        for obj in gemini_result.get(key, []):
            b = obj.get("bbox_norm") or obj.get("bbox", [])
            if len(b) == 4:
                objects.append((obj, b))
    # 단일 bbox 키도 체크
    for key in ["target_object","destination"]:
        obj = gemini_result.get(key)
        if isinstance(obj, dict) and obj.get("bbox_norm"):
            objects.append((obj, obj["bbox_norm"]))

    for i, (obj, b) in enumerate(objects):
        color = PALETTE[i % len(PALETTE)]
        x1 = int((b[0]-b[2]/2)*sw); y1 = int((b[1]-b[3]/2)*sh)
        x2 = int((b[0]+b[2]/2)*sw); y2 = int((b[1]+b[3]/2)*sh)
        cv2.rectangle(scene_vis,(x1,y1),(x2,y2),color,2)
        label = obj.get("label") or obj.get("object") or obj.get("description","")
        label = str(label)[:20]
        (lw,lh),_ = cv2.getTextSize(label, FONT, 0.42, 1)
        cv2.rectangle(scene_vis,(x1,max(y1-lh-6,0)),(x1+lw+4,y1),color,-1)
        cv2.putText(scene_vis,label,(x1+2,max(y1-3,lh)),FONT,0.42,(0,0,0),1,cv2.LINE_AA)
        # grasp point
        gp = obj.get("grasp_point_norm",[])
        if len(gp)==2:
            gpx,gpy=int(gp[0]*sw),int(gp[1]*sh)
            cv2.drawMarker(scene_vis,(gpx,gpy),color,cv2.MARKER_CROSS,18,2)

    scene_resized = cv2.resize(scene_vis, (IMG_W, IMG_H))
    dash[IMG_Y:IMG_Y+IMG_H, IMG_X:IMG_X+IMG_W] = scene_resized
    _rect(dash, IMG_X, IMG_Y, IMG_X+IMG_W, IMG_Y+IMG_H, C_BORDER, filled=False, t=2)
    _t(dash, "  Scene + Gemini Detection", (IMG_X, IMG_Y+IMG_H+16), scale=0.4, fg=C_MUTED)

    # ── LEFT BOTTOM: 점수 게이지 + 요약 ──────────────────────────────────────
    GAUGE_Y = IMG_Y + IMG_H + 30
    yc = _card(dash, IMG_X, GAUGE_Y, IMG_X+IMG_W, DH-10,
               title="  Evaluation Result", title_color=o_color)
    # 게이지 원
    _score_gauge(dash, IMG_X+70, yc+60, 52, score, overall)
    # 요약 텍스트
    summary = eval_result.get("summary","")
    tx = IMG_X + 140
    _t(dash, f"Quality: {quality}", (tx, yc+20), scale=0.45, fg=C_YELLOW, bold=True)
    _t(dash, f"Robot usable: {'YES' if eval_result.get('robot_usable') else 'NO'}",
       (tx, yc+42), scale=0.44,
       fg=C_GREEN if eval_result.get("robot_usable") else C_RED)
    for j, line in enumerate(_wrap(summary, 45)[:3]):
        _t(dash, line, (tx, yc+66+j*18), scale=0.40, fg=C_TEXT)

    # Missing / Violations
    missing = eval_result.get("missing_fields", [])
    violations = eval_result.get("physical_violations", [])
    my = yc + 120
    if missing:
        _t(dash, "Missing:", (IMG_X+6, my), scale=0.38, fg=C_RED)
        _t(dash, ", ".join(missing)[:70], (IMG_X+6, my+15), scale=0.36, fg=(160,120,120))
        my += 34
    if violations:
        _t(dash, "Violations:", (IMG_X+6, my), scale=0.38, fg=C_ORANGE)
        _t(dash, ", ".join(violations)[:70], (IMG_X+6, my+15), scale=0.36, fg=(180,140,100))

    # ── RIGHT TOP: GPT-4o 씬 분석 ────────────────────────────────────────────
    RX = IMG_X + IMG_W + 14
    RW = DW - RX - 10
    R1Y = 54
    R1H = 170
    yc = _card(dash, RX, R1Y, RX+RW, R1Y+R1H,
               title="  Step 1  GPT-4o — Scene Analysis & Task Design", title_color=C_BLUE)

    diff  = design.get("difficulty","?")
    dc    = {"easy":C_GREEN,"medium":C_YELLOW,"hard":C_RED}.get(diff, C_MUTED)
    _t(dash, f"Task: {design.get('chosen_task','?')}", (RX+8, yc+2), scale=0.48, fg=C_YELLOW, bold=True)
    _t(dash, f"Difficulty: {diff}", (RX+8, yc+22), scale=0.4, fg=dc)
    _t(dash, f"Category: {design.get('task_category','?')}", (RX+160, yc+22), scale=0.4, fg=C_MUTED)
    _t(dash, "Reason:", (RX+8, yc+42), scale=0.38, fg=C_MUTED)
    for j, line in enumerate(_wrap(design.get("task_reason",""), 55)[:2]):
        _t(dash, line, (RX+8, yc+58+j*16), scale=0.38, fg=C_TEXT)
    _t(dash, "Instruction →", (RX+8, yc+94), scale=0.38, fg=C_MUTED)
    instr = design.get("instruction_for_gemini","")
    for j, line in enumerate(_wrap(instr, 60)[:3]):
        _t(dash, line, (RX+8, yc+110+j*16), scale=0.37, fg=(180,200,240))

    # ── RIGHT MID: Gemini 응답 요약 ───────────────────────────────────────────
    R2Y = R1Y + R1H + 8
    R2H = 180
    yc = _card(dash, RX, R2Y, RX+RW, R2Y+R2H,
               title="  Step 2  Gemini ER — Embodied Reasoning Response", title_color=C_YELLOW)

    lat2 = gemini_result.get("_latency_s", 0)
    _t(dash, f"Latency: {lat2:.1f}s", (RX+RW-100, yc+2), scale=0.38, fg=C_MUTED)

    if gemini_result.get("parse_error"):
        _t(dash, "JSON PARSE ERROR", (RX+8, yc+30), scale=0.6, fg=C_RED, bold=True)
        raw = gemini_result.get("raw","")
        for j, line in enumerate(_wrap(raw, 65)[:5]):
            _t(dash, line, (RX+8, yc+58+j*18), scale=0.36, fg=(160,120,120))
    else:
        # 핵심 필드 유무 표시
        KEY_FIELDS = ["bbox_norm","grasp_point_norm","approach_direction",
                      "gripper_opening_mm","pregrasp_offset_mm","collision_risk","action_steps"]
        raw_str = json.dumps(gemini_result)
        gy = yc + 4
        for j, field in enumerate(KEY_FIELDS):
            present = field in raw_str
            icon = "v" if present else "x"
            fc = C_GREEN if present else C_RED
            col = RX + 8 + (j % 2) * (RW // 2)
            row = gy + (j // 2) * 22
            _t(dash, f"[{icon}] {field}", (col, row), scale=0.38, fg=fc)

        # 실제 값 스니펫
        _t(dash, "Response snippet:", (RX+8, gy+80), scale=0.38, fg=C_MUTED)
        skip = {"_latency_s","_model","_mode","_raw"}
        lines = []
        for k, v in gemini_result.items():
            if k.startswith("_") or k in skip: continue
            if isinstance(v, (str,int,float,bool)):
                lines.append(f"{k}: {v}")
            elif isinstance(v, list) and v:
                lines.append(f"{k}: [{len(v)} items]")
        for j, line in enumerate(lines[:4]):
            _t(dash, line[:65], (RX+8, gy+98+j*17), scale=0.37, fg=C_TEXT)

    # ── RIGHT BOTTOM: 기준별 채점표 ───────────────────────────────────────────
    R3Y = R2Y + R2H + 8
    R3H = DH - R3Y - 10
    yc = _card(dash, RX, R3Y, RX+RW, DH-10,
               title="  Step 3  GPT-4o — Criteria Evaluation", title_color=o_color)

    lat3 = eval_result.get("_latency_s", 0)
    _t(dash, f"Latency: {lat3:.1f}s", (RX+RW-100, yc+2), scale=0.38, fg=C_MUTED)

    criteria = eval_result.get("criteria_results", [])
    col_w = RW // 2
    for j, c in enumerate(criteria[:8]):
        col = RX + 8 + (j % 2) * col_w
        row = yc + 6 + (j // 2) * 46
        passed = c.get("result","fail") == "pass"
        sc = c.get("score", "?")
        fc = C_GREEN if passed else C_RED
        icon = "PASS" if passed else "FAIL"
        # 기준 배경 박스
        _rect(dash, col-2, row-2, col+col_w-16, row+40, (35,40,58))
        _rect(dash, col-2, row-2, col+col_w-16, row+40, fc, filled=False, t=1)
        _t(dash, icon, (col+2, row+13), scale=0.38, bold=True, fg=fc)
        _t(dash, f"{sc}pt", (col+col_w-55, row+13), scale=0.38, fg=fc)
        crit_short = c.get("criterion","")[:38]
        _t(dash, crit_short, (col+2, row+27), scale=0.33, fg=C_TEXT)
        comment_short = c.get("comment","")[:42]
        _t(dash, comment_short, (col+2, row+40), scale=0.30, fg=C_MUTED)

    # strengths / weaknesses 한 줄씩
    sw_y = yc + 6 + (len(criteria[:8])//2 + (1 if len(criteria[:8])%2 else 0)) * 46 + 8
    for s in eval_result.get("strengths",[])[:1]:
        _t(dash, f"+ {s[:70]}", (RX+8, sw_y), scale=0.37, fg=C_GREEN); sw_y+=17
    for w in eval_result.get("weaknesses",[])[:2]:
        _t(dash, f"- {w[:70]}", (RX+8, sw_y), scale=0.37, fg=C_ORANGE); sw_y+=17

    return dash


# ══════════════════════════════════════════════════════════════════════════════
# 전체 파이프라인
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(gemini, gpt, gemini_model, gpt_model,
                 img_rgb: np.ndarray, eval_mode: bool = True) -> dict:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Step 1
    design = step1_gpt_design(gpt, gpt_model, img_rgb)
    instruction = design.get("instruction_for_gemini", "이 씬에서 로봇이 할 수 있는 태스크를 분석하세요.")

    # Step 2
    gemini_result = step2_gemini_reason(gemini, gemini_model, img_rgb, instruction)

    # Step 3
    if eval_mode and not design.get("parse_error"):
        eval_result = step3_gpt_evaluate(gpt, gpt_model, img_rgb, design, gemini_result)
    else:
        eval_result = {"overall": "skipped", "score": 0, "summary": "평가 스킵",
                       "criteria_results": [], "strengths": [], "weaknesses": [],
                       "embodied_reasoning_quality": "-"}

    total_latency = (design.get("_latency_s",0) +
                     gemini_result.get("_latency_s",0) +
                     eval_result.get("_latency_s",0))
    print(f"\n  총 소요: {total_latency:.1f}s")

    dashboard = build_dashboard(img_bgr, design, gemini_result, eval_result)

    return {
        "design": design,
        "gemini": gemini_result,
        "eval": eval_result,
        "dashboard": dashboard,
        "total_latency_s": total_latency,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 인터랙티브 루프
# ══════════════════════════════════════════════════════════════════════════════

def interactive(gemini, gpt, gemini_model, gpt_model, eval_mode: bool):
    camera = CameraCapture()
    out_dir = Path("./results/auto_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    last_dashboard = None
    last_output = None

    print(f"\n{'━'*65}")
    print(f"  Gemini Auto Eval — 씬을 세팅하고 [c]를 누르세요")
    print(f"  Gemini: {gemini_model}")
    print(f"  GPT:    {gpt_model}")
    print(f"  평가 모드: {'ON' if eval_mode else 'OFF'}")
    print(f"{'━'*65}")
    print("  [c] 캡처 + 전체 파이프라인 실행")
    print("  [s] 결과 저장 (PNG + JSON)")
    print("  [q] 종료\n")

    WIN = "Gemini Embodied Reasoning — Auto Eval"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    while True:
        frame_rgb = camera.read()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        H, W = frame_bgr.shape[:2]

        display = frame_bgr.copy()
        _tbg(display, "씬 준비 후 [c] 눌러 분석 시작", (10, 22), scale=0.55, fg=(0,255,255))
        _tbg(display, f"Gemini: {gemini_model}", (10, 44), scale=0.4, fg=(200,200,200))
        _tbg(display, f"Eval: {'ON' if eval_mode else 'OFF'}", (10, 60), scale=0.4, fg=(200,200,200))

        if last_dashboard is not None:
            # 대시보드를 우측 1/3에 미니어처로
            mini_w = W // 2
            mini_h = int(mini_w * last_dashboard.shape[0] / last_dashboard.shape[1])
            mini = cv2.resize(last_dashboard, (mini_w, mini_h))
            mini_h = min(mini_h, H)
            display[0:mini_h, W-mini_w:W] = mini[:mini_h]

        cv2.imshow(WIN, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c'):
            print(f"\n{'═'*65}")
            print(f"  캡처 + 파이프라인 시작  {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'═'*65}")
            output = run_pipeline(gemini, gpt, gemini_model, gpt_model,
                                  frame_rgb, eval_mode)
            last_dashboard = output["dashboard"]
            last_output = output

            cv2.imshow(WIN, last_dashboard)
            cv2.resizeWindow(WIN, last_dashboard.shape[1], last_dashboard.shape[0])

        elif key == ord('s') and last_output is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            task = last_output["design"].get("chosen_task", "unknown")
            overall = last_output["eval"].get("overall", "?")
            stem = f"{ts}_{task}_{overall}"

            cv2.imwrite(str(out_dir / f"{stem}.png"), last_dashboard)
            save_data = {
                "timestamp": ts,
                "gemini_model": gemini_model,
                "gpt_model": gpt_model,
                "design": {k:v for k,v in last_output["design"].items() if not k.startswith("_")},
                "gemini_response": {k:v for k,v in last_output["gemini"].items() if not k.startswith("_")},
                "evaluation": {k:v for k,v in last_output["eval"].items() if not k.startswith("_")},
                "total_latency_s": last_output["total_latency_s"],
            }
            with open(out_dir / f"{stem}.json", "w") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            print(f"[저장] {out_dir}/{stem}.*")

    camera.release()
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# 연속 라운드 모드
# ══════════════════════════════════════════════════════════════════════════════

def run_rounds(gemini, gpt, gemini_model, gpt_model, rounds: int, eval_mode: bool):
    camera = CameraCapture()
    out_dir = Path("./results/auto_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    print(f"\n[연속 모드] {rounds}라운드")
    for r in range(1, rounds+1):
        print(f"\n{'━'*65}")
        print(f"  Round {r}/{rounds}")
        input("  씬 세팅 후 Enter...")

        frame_rgb = camera.read()
        output = run_pipeline(gemini, gpt, gemini_model, gpt_model,
                              frame_rgb, eval_mode)
        all_results.append(output)

        cv2.imshow(f"Round {r}", output["dashboard"])
        cv2.waitKey(2000)

    camera.release()
    cv2.destroyAllWindows()

    # 최종 요약
    print(f"\n{'━'*65}")
    print(f"  ROUNDS SUMMARY")
    print(f"{'━'*65}")
    total_score = 0
    for i, out in enumerate(all_results, 1):
        task = out["design"].get("chosen_task","?")
        overall = out["eval"].get("overall","?")
        score = out["eval"].get("score",0)
        total_score += score
        emoji = {"pass":"✅","partial":"⚠️","fail":"❌"}.get(overall,"?")
        print(f"  Round {i}: {emoji} {task:25s} Score:{score}/100")
    avg = total_score / len(all_results)
    print(f"\n  평균 점수: {avg:.1f}/100")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "gemini_model": gemini_model,
        "gpt_model": gpt_model,
        "rounds": rounds,
        "average_score": avg,
        "results": [
            {
                "task": o["design"].get("chosen_task","?"),
                "overall": o["eval"].get("overall","?"),
                "score": o["eval"].get("score",0),
                "latency_s": o["total_latency_s"],
            } for o in all_results
        ]
    }
    path = out_dir / f"rounds_summary_{ts}.json"
    with open(path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  요약 저장: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Gemini Embodied Reasoning 자동 평가")
    p.add_argument("--gemini-model", default="gemini-robotics-er-1.5-preview")
    p.add_argument("--gpt-model", default="gpt-4o",
                   help="씬 분석 + 평가에 사용할 GPT 모델 (기본: gpt-4o)")
    p.add_argument("--image", default=None, help="카메라 대신 이미지 파일 사용")
    p.add_argument("--rounds", type=int, default=0,
                   help="연속 테스트 라운드 수 (0=인터랙티브)")
    p.add_argument("--no-eval", action="store_true", help="GPT 평가 스킵")
    args = p.parse_args()

    try:
        google_key, openai_key = load_api_keys()
    except EnvironmentError as e:
        print(f"[Error] {e}")
        sys.exit(1)

    gemini, gpt = init_clients(google_key, openai_key,
                               args.gemini_model, args.gpt_model)
    eval_mode = not args.no_eval

    if args.image:
        img_rgb = np.array(Image.open(args.image).convert("RGB"))
        output = run_pipeline(gemini, gpt, args.gemini_model, args.gpt_model,
                              img_rgb, eval_mode)
        cv2.imshow("Auto Eval Result", output["dashboard"])
        print("\n[아무 키나 누르면 종료]")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.rounds > 0:
        run_rounds(gemini, gpt, args.gemini_model, args.gpt_model,
                   args.rounds, eval_mode)

    else:
        interactive(gemini, gpt, args.gemini_model, args.gpt_model, eval_mode)


if __name__ == "__main__":
    main()
