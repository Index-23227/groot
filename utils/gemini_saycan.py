"""
GeminiSayCan — ReAct 기반 Pharmacy Robot Orchestrator

SayCan (2022) 대비 2026 현대화 포인트:
  1. 단일 모델 통합: LLM + Affordance + Verify = Gemini ER 하나
  2. Zero-shot affordance: 학습된 value function 없이 이미지 보고 판단
  3. ReAct 루프: Reason → Act → Observe → Reason → Act ...
  4. 동적 재계획: 실패 시 현재 이미지 보고 즉시 새 plan
  5. Chain-of-Thought: Gemini가 매 단계 이유 설명
  6. 구조화 JSON 출력: 파싱 에러 없는 안정적 인터페이스

전체 흐름:
  Instruction + Image
       ↓
  [PLAN] Gemini → skill sequence + affordance scores
       ↓ (skill마다 반복)
  [AFFORD] 실행 직전 affordance 재확인
       ↓
  [ACT] Skill 실행 (scripted, 학습 없음)
       ↓
  [OBSERVE] 새 이미지 캡처
       ↓
  [VERIFY] Gemini → 성공 여부 판단
       ↓ 실패 시
  [REPLAN] Gemini → 새 plan 수립
       ↓
  다음 skill

사용법:
  python utils/gemini_saycan.py -i "타이레놀 꺼내서 트레이에 넣어줘"
  python utils/gemini_saycan.py -i "타이레놀 꺼내서 트레이에 넣어줘" --dry-run
"""

import sys, os, time, json, argparse, datetime
from pathlib import Path
from typing import Optional
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.skill_library import SkillLibrary, SkillResult
from utils.task_planner   import TaskPlanner
from utils.calibration    import load_calibration


# ── 실행 로그 ──────────────────────────────────────────────────────

class ExecutionLog:
    def __init__(self, instruction: str):
        self.instruction  = instruction
        self.started_at   = datetime.datetime.now().isoformat()
        self.steps        = []
        self.final_result = None

    def add_step(self, step_num: int, skill: str, params: dict,
                 affordance: float, skill_result: SkillResult,
                 verify_score: int, latency_s: float):
        self.steps.append({
            "step":          step_num,
            "skill":         skill,
            "params":        params,
            "affordance":    affordance,
            "skill_success": skill_result.success,
            "skill_reason":  skill_result.reason,
            "verify_score":  verify_score,
            "latency_s":     latency_s,
        })

    def save(self, path: Optional[str] = None):
        if path is None:
            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"results/saycan/{ts}_execution.json"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vars(self), f, ensure_ascii=False, indent=2, default=str)
        print(f"\n[Log] {path}")
        return path


# ── Mock 카메라 & 로봇 (dry-run용) ────────────────────────────────

class MockCamera:
    camera_matrix = np.array([[615, 0, 320], [0, 615, 240], [0, 0, 1]], dtype=float)

    def read(self):
        rgb   = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
        depth = np.ones((480, 640), dtype=np.float32) * 0.45
        return rgb, depth

    def release(self):
        pass


class MockRobot:
    _stroke = 0

    def movel(self, pose, vel=50, acc=50, wait=True):
        print(f"    [mock movel] {[round(p,1) for p in pose[:3]]} mm")
        time.sleep(0.1)
        return True

    def set_gripper(self, stroke, wait_sec=1.0):
        self._stroke = stroke
        state = "닫힘" if stroke > 350 else "열림"
        print(f"    [mock gripper] stroke={stroke} ({state})")
        time.sleep(0.1)

    def get_gripper_stroke(self):
        # dry-run: pick 시 mock 성공 반환
        return 250 if self._stroke > 350 else 0

    def get_tcp_pose(self):
        return [400.0, 0.0, 300.0, 180.0, 0.0, 0.0]

    def shutdown(self):
        pass


class MockBridge:
    """dry-run용 bridge: 임의 pose 반환."""
    def detect_object_for_pick(self, rgb, depth, object_name, grasp_style):
        return {
            "grasp_pose":    [350.0, 0.0, 80.0, 180.0, 0.0, 0.0],
            "pregrasp_pose": [350.0, 0.0, 200.0, 180.0, 0.0, 0.0],
            "stroke_open":   400,
        }

    def detect_place_location(self, rgb, depth, place_norm):
        return {"place_pose": [300.0, 0.0, 50.0, 180.0, 0.0, 0.0]}


# ── GeminiSayCan 메인 클래스 ───────────────────────────────────────

class GeminiSayCan:
    """
    ReAct 루프 기반 오케스트레이터.

    ReAct = Reasoning + Acting
    매 step:  Reason(Gemini) → Act(Skill) → Observe(Camera) → Reason...
    """

    MAX_RETRIES_PER_SKILL = 2   # skill당 최대 재시도
    MAX_REPLAN_COUNT      = 2   # 전체 replan 최대 횟수
    AFFORDANCE_THRESHOLD  = 0.4  # 이 이하면 skill 건너뜀

    def __init__(self, dry_run: bool = False):
        self.dry_run     = dry_run
        self.skill_lib   = SkillLibrary()
        self.planner     = TaskPlanner(self.skill_lib)
        self.replan_count = 0

        if dry_run:
            self.camera = MockCamera()
            self.robot  = MockRobot()
            self.bridge = MockBridge()
            print("[GeminiSayCan] DRY-RUN 모드")
        else:
            self._init_real()

    def _init_real(self):
        from utils.gemini_bridge   import RealSenseCapture, DoosanCartesianRobot, GeminiBridge

        # 캘리브레이션 로드
        T, intr = load_calibration()
        K = intr.get("camera_matrix")
        if K is None:
            raise RuntimeError("camera_matrix 없음 — 먼저 calibration.py --run 실행")

        self.camera = RealSenseCapture()
        K           = self.camera.camera_matrix   # 실기기 intrinsics 우선
        self.robot  = DoosanCartesianRobot(dry_run=False)
        self.bridge = GeminiBridge(T, K, dry_run=False)
        print("[GeminiSayCan] 실제 로봇 모드")

    # ── 메인 실행 루프 ──────────────────────────────────────────────

    def run(self, instruction: str) -> ExecutionLog:
        log = ExecutionLog(instruction)

        print(f"\n{'='*60}")
        print(f"  GeminiSayCan")
        print(f"  Instruction: {instruction}")
        print(f"{'='*60}")

        # ── Phase 1: PLAN ──────────────────────────────────────────
        # Gemini가 씬 보고 skill sequence + affordance 생성
        rgb, depth = self.camera.read()
        plan_result = self.planner.plan(rgb, instruction)

        if not plan_result.get("plan"):
            print("\n❌ Plan 수립 실패 — 종료")
            log.final_result = {"success": False, "reason": "plan 수립 실패"}
            log.save()
            return log

        plan_steps = plan_result["plan"]
        step_idx   = 0

        # ── Phase 2: ReAct 실행 루프 ───────────────────────────────
        while step_idx < len(plan_steps):
            step      = plan_steps[step_idx]
            skill_name = step["skill"]
            params     = step.get("params", {})
            expected   = step.get("expected_result", f"{skill_name} 완료")
            t_step     = time.time()

            print(f"\n{'─'*50}")
            print(f"  Step {step_idx+1}/{len(plan_steps)}: [{skill_name}] "
                  f"{json.dumps(params, ensure_ascii=False)}")
            print(f"  예상 결과: {expected}")

            # ── AFFORD: 실행 직전 affordance 재확인 ─────────────
            rgb, depth  = self.camera.read()
            affordance  = self.planner.check_affordance(rgb, skill_name, params)
            aff_score   = affordance.get("score", 0.5)

            if aff_score < self.AFFORDANCE_THRESHOLD:
                reason = affordance.get("reason", "affordance 낮음")
                alt    = affordance.get("alternative")
                print(f"  ⚠️  Affordance 낮음({aff_score:.2f}) — {reason}")

                if alt and alt in self.skill_lib.REGISTRY:
                    print(f"  → 대안 skill '{alt}'로 대체")
                    step["skill"]  = alt
                    step["params"] = {}
                    skill_name = alt
                    params     = {}
                else:
                    step_idx += 1
                    continue

            # ── ACT: Skill 실행 ──────────────────────────────────
            skill   = self.skill_lib.get(skill_name)
            retries = 0
            skill_result = SkillResult(False, skill_name, "미실행")

            while retries <= self.MAX_RETRIES_PER_SKILL:
                skill_result = skill.execute(
                    params, self.robot, self.camera, self.bridge)

                # ── OBSERVE & VERIFY: 새 이미지로 결과 확인 ────
                rgb, depth  = self.camera.read()
                verify      = self.planner.verify_skill(
                    rgb, skill_name, params, expected)
                verify_ok   = verify.get("success", False)
                verify_score = verify.get("score", 0)

                if skill_result.success and verify_ok:
                    break   # 성공

                # 실패 처리
                if retries < self.MAX_RETRIES_PER_SKILL:
                    print(f"  🔄 재시도 {retries+1}/{self.MAX_RETRIES_PER_SKILL} "
                          f"— {skill_result.reason}")
                    retries += 1
                    time.sleep(0.5)
                else:
                    break

            # ── 로그 기록 ─────────────────────────────────────────
            log.add_step(
                step_idx + 1, skill_name, params,
                aff_score, skill_result, verify_score,
                round(time.time() - t_step, 1)
            )

            # ── 실패 시 REPLAN ───────────────────────────────────
            if not skill_result.success or not verify_ok:
                failure_reason = skill_result.reason or verify.get("observation", "실패")
                print(f"\n  ❌ Step {step_idx+1} 최종 실패: {failure_reason}")

                if self.replan_count >= self.MAX_REPLAN_COUNT:
                    print("  ⛔ 최대 replan 횟수 초과 — 태스크 중단")
                    break

                rgb, _    = self.camera.read()
                replan    = self.planner.replan(
                    rgb, instruction, plan_steps,
                    step_idx, failure_reason
                )
                self.replan_count += 1

                if replan.get("abort_recommended"):
                    print(f"  ⛔ Gemini 중단 권고: {replan.get('abort_reason')}")
                    break

                # 새 plan으로 교체
                new_steps = replan.get("recovery_plan", [])
                if new_steps:
                    plan_steps = new_steps
                    step_idx   = 0
                    print(f"  🔄 새 plan으로 재시작 ({len(new_steps)}단계)")
                    continue
            else:
                print(f"  ✅ Step {step_idx+1} 완료 (score={verify_score})")
                step_idx += 1

        # ── 결과 정리 ─────────────────────────────────────────────
        success = all(s["skill_success"] and s["verify_score"] >= 60
                      for s in log.steps)
        planner_summary = self.planner.summary()

        log.final_result = {
            "success":        success,
            "total_steps":    len(log.steps),
            "replan_count":   self.replan_count,
            "gemini_queries": planner_summary["total_queries"],
            "gemini_latency_s": planner_summary["total_latency_s"],
        }

        self._print_final(log)
        log.save()
        return log

    def _print_final(self, log: ExecutionLog):
        r = log.final_result
        print(f"\n{'='*60}")
        print(f"  결과: {'✅ 성공' if r['success'] else '❌ 실패/부분성공'}")
        print(f"  실행 단계:      {r['total_steps']}")
        print(f"  Replan 횟수:    {r['replan_count']}")
        print(f"  Gemini 쿼리:    {r['gemini_queries']}회")
        print(f"  Gemini 소요:    {r['gemini_latency_s']}초")
        print(f"{'='*60}")

        if log.steps:
            print("\n  단계별 요약:")
            for s in log.steps:
                mark = "✅" if s["skill_success"] else "❌"
                print(f"    {s['step']}. {mark} {s['skill']} "
                      f"(aff={s['affordance']:.2f}, verify={s['verify_score']}/100, "
                      f"{s['latency_s']}s)")

    def shutdown(self):
        if not self.dry_run:
            if hasattr(self.camera, "release"):
                self.camera.release()
            if hasattr(self.robot, "shutdown"):
                self.robot.shutdown()


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GeminiSayCan — 약국 로봇 태스크 실행")
    p.add_argument("-i", "--instruction",
                   default="타이레놀을 꺼내서 라벨을 확인하고 트레이에 넣어줘",
                   help="자연어 태스크 지시")
    p.add_argument("--dry-run", action="store_true",
                   help="로봇 없이 Gemini 추론만 실행")
    args = p.parse_args()

    saycan = GeminiSayCan(dry_run=args.dry_run)
    try:
        saycan.run(args.instruction)
    finally:
        saycan.shutdown()
