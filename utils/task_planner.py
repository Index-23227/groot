"""
Task Planner — Gemini ER가 Plan + Affordance + Verify 담당

SayCan과의 차이:
  SayCan: LLM(텍스트) + 별도 Affordance 모델(학습) + 별도 Value function
  여기:   Gemini ER 하나로 Planning + Affordance + Verification 통합

  핵심 수식:
    SayCan: score(skill) = p_LLM(skill | instruction) × p_affordance(skill | state)
    여기:   score(skill) = Gemini(image + instruction + skill) → probability

현대적 개선점:
  - Zero-shot affordance (학습 불필요, 이미지 보고 판단)
  - Chain-of-Thought reasoning (왜 이 skill인지 설명)
  - 동적 재계획 (실패 시 새 이미지 보고 plan 수정)
  - 구조화 JSON 출력 (파싱 에러 없음)
"""

import sys, os, json, time, base64
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Optional, Any
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.skill_library import SkillLibrary


# ── Gemini 클라이언트 ───────────────────────────────────────────────

def _load_google_key() -> str:
    token = Path(__file__).parent.parent / "token"
    if token.exists():
        for line in token.read_text().splitlines():
            line = line.strip()
            if line.startswith("AIza"):
                return line
    key = os.environ.get("GOOGLE_API_KEY", "")
    if key:
        return key
    raise RuntimeError("Google API 키 없음")


def _query_gemini_json(pil_img, prompt: str, model="gemini-robotics-er-1.5-preview") -> dict:
    """Gemini ER에 이미지 + 프롬프트 전송 → JSON dict 반환."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_load_google_key())
    resp   = client.models.generate_content(
        model=model,
        contents=[prompt, pil_img],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    text  = resp.text.strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1:
        return {"error": text}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {"error": text, "raw": text}


def _to_pil(rgb: np.ndarray):
    from PIL import Image
    return Image.fromarray(rgb)


# ── 프롬프트 ────────────────────────────────────────────────────────

def _plan_prompt(instruction: str, skill_descriptions: str) -> str:
    return f"""
당신은 약국 로봇(Doosan E0509)의 태스크 플래너입니다.
이미지를 보고 주어진 instruction을 수행하기 위한 skill 시퀀스를 계획하세요.

Instruction: "{instruction}"

사용 가능한 Skills:
{skill_descriptions}

로봇 환경:
- 약국 조제 보조 로봇
- Intel RealSense D435 카메라로 씬 관찰
- 그리퍼: ROBOTIS RH-P12-RN-A (최대 120mm 열림)

다음 순서로 분석하세요 (Chain of Thought):
1. 이미지에서 무엇이 보이는가? (약품 종류, 위치, 상태)
2. Instruction을 완료하려면 어떤 단계가 필요한가?
3. 각 skill의 현재 수행 가능성(affordance)은?
4. 최적 skill 시퀀스는?

JSON으로 답하세요:
{{
  "scene_analysis": "씬에서 관찰된 것 (약품명, 위치, 상태)",
  "detected_objects": [
    {{"name": "타이레놀", "bbox_norm": [0.4, 0.5, 0.1, 0.2], "graspable": true}}
  ],
  "affordances": {{
    "pick":     {{"score": 0.9, "reason": "타이레놀이 명확히 보이고 집을 수 있는 위치"}},
    "place":    {{"score": 0.8, "reason": "트레이가 비어있음"}},
    "inspect":  {{"score": 0.9, "reason": "물체를 집으면 라벨 확인 가능"}},
    "home":     {{"score": 1.0, "reason": "언제든 가능"}},
    "handover": {{"score": 0.7, "reason": "사람이 근처에 있는 것으로 보임"}}
  }},
  "plan": [
    {{
      "step": 1,
      "skill": "pick",
      "params": {{"object": "타이레놀", "grasp_style": "top"}},
      "reason": "타이레놀을 집어야 다음 단계 가능",
      "expected_result": "타이레놀을 들고 있음"
    }}
  ],
  "overall_confidence": 0.85,
  "potential_issues": ["라벨이 가려져 있을 수 있음"]
}}
"""


def _affordance_prompt(skill: str, params: dict, skill_desc: str) -> str:
    params_str = json.dumps(params, ensure_ascii=False)
    return f"""
당신은 로봇 조작의 affordance 전문가입니다.
현재 이미지를 보고, 다음 skill을 지금 바로 실행할 수 있는지 평가하세요.

평가 대상 Skill: "{skill}"
파라미터: {params_str}
Skill 설명: {skill_desc}

평가 기준:
- 필요한 물체가 카메라에 보이는가?
- 물체가 접근 가능한 위치에 있는가?
- 경로에 장애물이 있는가?
- 그리퍼와 물체 크기가 맞는가?

JSON으로 답하세요:
{{
  "score": 0.85,           // 0~1, 실행 가능성
  "executable": true,      // 0.5 초과면 true
  "reason": "타이레놀이 명확히 보이고 집기 적합한 위치",
  "obstacles": [],         // 방해 요소
  "alternative": null      // 대안 skill (낮을 때), 없으면 null
}}
"""


def _verify_prompt(skill: str, params: dict, expected_result: str) -> str:
    params_str = json.dumps(params, ensure_ascii=False)
    return f"""
방금 로봇이 "{skill}" skill을 실행했습니다.
파라미터: {params_str}
예상 결과: {expected_result}

현재 이미지를 보고 skill이 성공했는지 평가하세요.

평가 포인트:
- 예상 결과가 실제로 달성됐는가?
- 물체 상태는 어떠한가?
- 다음 skill 실행에 문제가 없는가?

JSON으로 답하세요:
{{
  "success": true,
  "score": 90,             // 성공도 0~100
  "observation": "타이레놀이 그리퍼에 안정적으로 잡혀 있음",
  "next_state": "물체를 들고 있음",
  "issues": [],            // 발견된 문제점
  "retry_recommended": false
}}
"""


def _replan_prompt(instruction: str, original_plan: list,
                   failed_step: int, failure_reason: str,
                   skill_descriptions: str) -> str:
    remaining = original_plan[failed_step:]
    remaining_str = json.dumps(remaining, ensure_ascii=False, indent=2)
    return f"""
로봇이 태스크 실행 중 문제가 발생했습니다.

원래 Instruction: "{instruction}"
실패한 Step: {failed_step + 1}
실패 이유: "{failure_reason}"
남은 계획: {remaining_str}

현재 이미지를 보고 새로운 계획을 수립하세요.

사용 가능한 Skills:
{skill_descriptions}

JSON으로 답하세요:
{{
  "situation_assessment": "현재 상황 분석",
  "recovery_plan": [
    {{
      "step": 1,
      "skill": "home",
      "params": {{}},
      "reason": "안전한 상태로 복귀",
      "expected_result": "홈 포지션"
    }}
  ],
  "confidence": 0.7,
  "abort_recommended": false,
  "abort_reason": null
}}
"""


def _inspect_label_prompt(expected_label: Optional[str]) -> str:
    expect_str = f'기대하는 약품: "{expected_label}"' if expected_label else "라벨 확인 후 보고"
    return f"""
로봇 그리퍼가 약품을 들고 카메라 앞에 있습니다.
{expect_str}

이미지를 보고 약품 라벨을 판독하세요.

JSON으로 답하세요:
{{
  "label_readable": true,
  "medicine_name": "타이레놀",
  "dosage": "500mg",
  "quantity": "20정",
  "expiry": "2026-12",
  "correct": true,         // expected_label과 일치 여부
  "confidence": 0.95,
  "issues": []
}}
"""


# ── Task Planner 클래스 ────────────────────────────────────────────

class TaskPlanner:
    """
    Gemini ER 기반 Task Planner.

    SayCan의 3가지 구성요소를 Gemini 하나로 통합:
      1. Plan      : instruction → skill sequence 생성
      2. Affordance: skill 실행 가능성 평가 (이미지 기반, 학습 불필요)
      3. Verify    : skill 실행 후 성공 여부 판단
      4. Replan    : 실패 시 새 plan 수립
    """

    def __init__(self, skill_library: SkillLibrary):
        self.skill_lib    = skill_library
        self.query_count  = 0
        self.query_log    = []

    def _query(self, rgb: np.ndarray, prompt: str, label: str) -> dict:
        pil = _to_pil(rgb)
        t0  = time.time()
        print(f"\n  [Gemini Q{self.query_count+1}] {label}...")
        result  = _query_gemini_json(pil, prompt)
        elapsed = time.time() - t0
        self.query_count += 1
        self.query_log.append({"label": label, "latency_s": round(elapsed, 2), "result": result})
        print(f"  → {elapsed:.1f}s")

        # 에러 체크
        if "error" in result:
            print(f"  ⚠️  파싱 에러: {result['error'][:100]}")
        return result

    # ── 1. Plan ─────────────────────────────────────────────────────

    def plan(self, rgb: np.ndarray, instruction: str) -> dict:
        """
        이미지 + instruction → skill 시퀀스 생성.
        affordance score도 함께 계산하여 낮은 skill은 제외.
        """
        prompt = _plan_prompt(instruction, self.skill_lib.descriptions_for_prompt())
        result = self._query(rgb, prompt, f"Plan: {instruction[:30]}")

        # affordance score 낮은 skill을 plan에서 필터링
        affordances = result.get("affordances", {})
        plan = result.get("plan", [])
        filtered = []
        for step in plan:
            skill     = step.get("skill", "")
            aff_score = affordances.get(skill, {}).get("score", 1.0)
            if aff_score >= 0.4:
                step["affordance_score"] = aff_score
                filtered.append(step)
            else:
                reason = affordances.get(skill, {}).get("reason", "")
                print(f"  ⚠️  '{skill}' 제외 (affordance={aff_score:.2f}: {reason})")

        result["plan"] = filtered
        self._print_plan(result)
        return result

    # ── 2. Affordance ────────────────────────────────────────────────

    def check_affordance(self, rgb: np.ndarray, skill: str, params: dict) -> dict:
        """
        skill 실행 직전, 현재 이미지로 실행 가능성 재확인.
        Plan 단계의 affordance와 별도로, 실행 직전에 한 번 더 체크.
        """
        skill_desc = self.skill_lib.DESCRIPTIONS.get(skill, {}).get("desc", skill)
        prompt     = _affordance_prompt(skill, params, skill_desc)
        result     = self._query(rgb, prompt, f"Affordance: {skill}")

        score = result.get("score", 0.5)
        exe   = result.get("executable", score > 0.5)
        print(f"  Affordance '{skill}': {score:.2f} ({'OK' if exe else 'LOW'})"
              f" — {result.get('reason', '')}")
        return result

    # ── 3. Verify ────────────────────────────────────────────────────

    def verify_skill(self, rgb: np.ndarray, skill: str,
                     params: dict, expected_result: str) -> dict:
        """skill 실행 후 새 이미지로 성공 여부 판단."""
        prompt = _verify_prompt(skill, params, expected_result)
        result = self._query(rgb, prompt, f"Verify: {skill}")

        success = result.get("success", False)
        score   = result.get("score", 0)
        print(f"  Verify '{skill}': {'✅' if success else '❌'} "
              f"({score}/100) — {result.get('observation', '')}")

        # inspect skill이면 라벨도 판독
        if skill == "inspect":
            expected_label = params.get("expected_label")
            label_result   = self.read_label(rgb, expected_label)
            result["label_result"] = label_result

        return result

    # ── 4. Replan ────────────────────────────────────────────────────

    def replan(self, rgb: np.ndarray, instruction: str,
               original_plan: list, failed_step: int,
               failure_reason: str) -> dict:
        """실패 시 현재 이미지 보고 새 plan 수립."""
        print(f"\n  [Replan] Step {failed_step+1} 실패: {failure_reason}")
        prompt = _replan_prompt(
            instruction, original_plan, failed_step, failure_reason,
            self.skill_lib.descriptions_for_prompt()
        )
        result = self._query(rgb, prompt, "Replan")

        if result.get("abort_recommended"):
            print(f"  ⛔ 중단 권고: {result.get('abort_reason')}")
        else:
            self._print_plan(result, key="recovery_plan")
        return result

    # ── 5. Label Inspect ─────────────────────────────────────────────

    def read_label(self, rgb: np.ndarray, expected_label: Optional[str] = None) -> dict:
        """inspect skill 전용: 약품 라벨 판독."""
        prompt = _inspect_label_prompt(expected_label)
        result = self._query(rgb, prompt, "Label Read")

        if result.get("label_readable"):
            name    = result.get("medicine_name", "?")
            dosage  = result.get("dosage", "?")
            correct = result.get("correct", True)
            mark    = "✅" if correct else "⚠️"
            print(f"  라벨: {mark} {name} {dosage}")
        else:
            print("  라벨 판독 실패")
        return result

    # ── 헬퍼 ─────────────────────────────────────────────────────────

    def _print_plan(self, plan_result: dict, key: str = "plan"):
        steps = plan_result.get(key, [])
        conf  = plan_result.get("overall_confidence") or plan_result.get("confidence", "?")
        print(f"\n  📋 Plan (confidence={conf}):")
        for step in steps:
            n     = step.get("step", "?")
            skill = step.get("skill", "?")
            parms = step.get("params", {})
            rsn   = step.get("reason", "")
            aff   = step.get("affordance_score", "")
            aff_s = f" [aff={aff:.2f}]" if aff != "" else ""
            print(f"    {n}. {skill}{aff_s} {json.dumps(parms, ensure_ascii=False)} — {rsn}")

    def summary(self) -> dict:
        total = sum(q["latency_s"] for q in self.query_log)
        return {
            "total_queries": self.query_count,
            "total_latency_s": round(total, 1),
            "queries": self.query_log,
        }
