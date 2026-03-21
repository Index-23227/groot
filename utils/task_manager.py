"""
Task Manager: Task Decomposition + History-based Execution

- task decomposition 결과를 JSON으로 저장
- history 폴더에서 최근 step 확인
- 성공 → 다음 action / 실패 → 동일 action 재시도
- 매 action 수행 후 history 업데이트

Usage:
    tm = TaskManager(session_dir="results/sessions")

    # 새 세션 시작 (task decomposition)
    session_id = tm.new_session(instruction, tasks)

    # 다음 실행할 action 가져오기
    action = tm.get_next_action(session_id)

    # action 수행 후 결과 기록
    tm.update_history(session_id, action, success=True, details={...})
"""
import json
import time
from pathlib import Path
from datetime import datetime


class TaskManager:
    def __init__(self, session_dir: str = "results/sessions"):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

    # ── 새 세션 생성 ─────────────────────────────────────────
    def new_session(self, instruction: str, tasks: list[dict],
                    rationale: str = "") -> str:
        """
        Task decomposition 결과로 새 세션 생성.

        Args:
            instruction: 원본 자연어 명령
            tasks: [{"task":"pick","target":"..."}, {"task":"place","target":"...","relation":"..."}]
            rationale: decomposition 근거

        Returns: session_id
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_path = self.session_dir / session_id
        session_path.mkdir(parents=True, exist_ok=True)

        # task.json 저장
        task_data = {
            "session_id": session_id,
            "instruction": instruction,
            "tasks": tasks,
            "rationale": rationale,
            "created_at": datetime.now().isoformat(),
            "total_steps": len(tasks),
        }
        (session_path / "task.json").write_text(
            json.dumps(task_data, indent=2, ensure_ascii=False), encoding="utf-8")

        # history 폴더 생성
        (session_path / "history").mkdir(exist_ok=True)

        print(f"  [TaskManager] 새 세션: {session_id}")
        print(f"    instruction: {instruction}")
        print(f"    steps: {len(tasks)}개")
        for i, t in enumerate(tasks):
            rel = f", {t.get('relation','')}" if t.get('relation') else ""
            print(f"      [{i}] ({t['task']}, {t.get('target','')}{rel})")

        return session_id

    # ── 최근 세션 찾기 ───────────────────────────────────────
    def get_latest_session(self) -> str | None:
        """가장 최근 세션 ID 반환. 없으면 None."""
        sessions = sorted(self.session_dir.iterdir())
        sessions = [s for s in sessions if s.is_dir() and (s / "task.json").exists()]
        return sessions[-1].name if sessions else None

    # ── 세션 로드 ────────────────────────────────────────────
    def load_session(self, session_id: str) -> dict:
        """세션의 task.json 로드"""
        task_path = self.session_dir / session_id / "task.json"
        if not task_path.exists():
            return {}
        return json.loads(task_path.read_text(encoding="utf-8"))

    # ── history 로드 ─────────────────────────────────────────
    def load_history(self, session_id: str) -> list[dict]:
        """세션의 history 전체 로드 (시간순 정렬)"""
        hist_dir = self.session_dir / session_id / "history"
        if not hist_dir.exists():
            return []
        files = sorted(hist_dir.glob("step_*.json"))
        history = []
        for f in files:
            history.append(json.loads(f.read_text(encoding="utf-8")))
        return history

    def get_latest_history(self, session_id: str) -> dict | None:
        """가장 최근 history entry 반환"""
        history = self.load_history(session_id)
        return history[-1] if history else None

    # ── 다음 action 결정 ─────────────────────────────────────
    def get_next_action(self, session_id: str) -> dict | None:
        """
        history를 확인하여 다음 실행할 action을 결정.

        - history 없음 → tasks[0] (첫 번째 action)
        - 최근 history 성공 → tasks[next_step]
        - 최근 history 실패 → tasks[same_step] (재시도)
        - 모든 step 완료 → None

        Returns: {"step_index": int, "task": dict, "is_retry": bool} or None
        """
        task_data = self.load_session(session_id)
        if not task_data:
            print(f"  [TaskManager] 세션 {session_id} 없음")
            return None

        tasks = task_data["tasks"]
        total = len(tasks)
        latest = self.get_latest_history(session_id)

        if latest is None:
            # history 없음 → 첫 번째 action
            print(f"  [TaskManager] history 없음 → step 0 시작")
            return {
                "step_index": 0,
                "task": tasks[0],
                "is_retry": False,
            }

        step_idx = latest["step_index"]
        success = latest["success"]

        if success:
            # 성공 → 다음 step
            next_idx = step_idx + 1
            if next_idx >= total:
                print(f"  [TaskManager] 모든 step 완료! ({total}/{total})")
                return None
            print(f"  [TaskManager] step {step_idx} 성공 → step {next_idx} 진행")
            return {
                "step_index": next_idx,
                "task": tasks[next_idx],
                "is_retry": False,
            }
        else:
            # 실패 → 동일 step 재시도
            retry_count = sum(1 for h in self.load_history(session_id)
                              if h["step_index"] == step_idx and not h["success"])
            print(f"  [TaskManager] step {step_idx} 실패 → 재시도 (attempt #{retry_count+1})")
            return {
                "step_index": step_idx,
                "task": tasks[step_idx],
                "is_retry": True,
            }

    # ── history 업데이트 ─────────────────────────────────────
    def update_history(self, session_id: str, step_index: int,
                       task: dict, success: bool,
                       ee_pick: tuple = None, ee_place: tuple = None,
                       details: dict = None) -> str:
        """
        action 수행 후 history 기록.

        Returns: history 파일 경로
        """
        hist_dir = self.session_dir / session_id / "history"
        hist_dir.mkdir(exist_ok=True)

        # 기존 history 수 → 파일 번호
        existing = sorted(hist_dir.glob("step_*.json"))
        file_idx = len(existing)

        entry = {
            "step_index": step_index,
            "task": task,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "ee_pick": list(ee_pick) if ee_pick else None,
            "ee_place": list(ee_place) if ee_place else None,
            "details": details or {},
        }

        filename = f"step_{file_idx:03d}.json"
        filepath = hist_dir / filename
        filepath.write_text(
            json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")

        status = "SUCCESS" if success else "FAILED"
        rel = f", {task.get('relation','')}" if task.get('relation') else ""
        print(f"  [History] [{status}] step {step_index}: "
              f"({task['task']}, {task.get('target','')}{rel})")
        print(f"    saved: {filepath}")

        return str(filepath)

    # ── 세션 완료 여부 ───────────────────────────────────────
    def is_session_complete(self, session_id: str) -> bool:
        """모든 step이 성공적으로 완료되었는지"""
        task_data = self.load_session(session_id)
        if not task_data:
            return False
        total = len(task_data["tasks"])
        history = self.load_history(session_id)
        completed = len(set(h["step_index"] for h in history if h["success"]))
        return completed >= total

    # ── 세션 상태 요약 ───────────────────────────────────────
    def get_status(self, session_id: str) -> dict:
        """세션의 현재 상태 요약"""
        task_data = self.load_session(session_id)
        history = self.load_history(session_id)

        if not task_data:
            return {"error": "session not found"}

        total = len(task_data["tasks"])
        completed = len(set(h["step_index"] for h in history if h["success"]))
        failed = sum(1 for h in history if not h["success"])
        latest = history[-1] if history else None

        return {
            "session_id": session_id,
            "instruction": task_data["instruction"],
            "total_steps": total,
            "completed_steps": completed,
            "failed_attempts": failed,
            "is_complete": completed >= total,
            "latest_step": latest["step_index"] if latest else -1,
            "latest_success": latest["success"] if latest else None,
        }

    # ── 전체 세션 목록 ───────────────────────────────────────
    def list_sessions(self) -> list[dict]:
        """모든 세션의 요약 목록"""
        sessions = sorted(self.session_dir.iterdir())
        result = []
        for s in sessions:
            if s.is_dir() and (s / "task.json").exists():
                result.append(self.get_status(s.name))
        return result
