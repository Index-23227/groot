"""
Gemini Embodied Reasoning 실시간 시각화 대시보드

test_gemini_embodied.py의 단발 테스트와 달리,
이 모듈은 연속 스트리밍 + 4분할 패널 대시보드를 제공합니다.

사용법:
  python utils/gemini_visualizer.py                     # 기본 대시보드
  python utils/gemini_visualizer.py --interval 3.0      # 3초마다 Gemini 호출
  python utils/gemini_visualizer.py --save-video out.avi
"""

import os, sys, time, json, threading, queue, argparse
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np
import cv2
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
from configs.doosan_e0509_config import CAMERA_WIDTH, CAMERA_HEIGHT
from utils.doosan_recorder import CameraCapture

# ─── 패널 레이아웃 ────────────────────────────────────────────────────────────
# ┌──────────────────┬──────────────────┐
# │  LIVE CAMERA     │  OBJECT DETECT   │
# │  + 오버레이       │  + bbox          │
# ├──────────────────┼──────────────────┤
# │  ACTION PLAN     │  REASONING LOG   │
# │  스텝 시각화      │  텍스트 스크롤    │
# └──────────────────┴──────────────────┘

PANEL_W = CAMERA_WIDTH   # 각 패널 너비
PANEL_H = CAMERA_HEIGHT  # 각 패널 높이
DASH_W = PANEL_W * 2
DASH_H = PANEL_H * 2

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = 0.42
FONT_MED = 0.55
LINE_H = 18  # 텍스트 줄 간격


def put_text_bg(img, text, org, scale=0.45, color=(255, 255, 255), bg=(0, 0, 0)):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + 3), bg, -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def norm_to_px(cx, cy, bw, bh, panel_w, panel_h):
    x1 = int((cx - bw / 2) * panel_w)
    y1 = int((cy - bh / 2) * panel_h)
    x2 = int((cx + bw / 2) * panel_w)
    y2 = int((cy + bh / 2) * panel_h)
    return x1, y1, x2, y2


# ─── 패널 렌더러 ──────────────────────────────────────────────────────────────

class PanelRenderer:
    OBJECT_COLORS = [
        (0, 255, 0), (255, 165, 0), (0, 200, 255),
        (255, 0, 255), (200, 200, 0), (0, 255, 200),
    ]

    def render_live(self, frame_bgr: np.ndarray, status: str, fps: float) -> np.ndarray:
        panel = cv2.resize(frame_bgr, (PANEL_W, PANEL_H))
        put_text_bg(panel, "LIVE CAMERA", (8, 20), scale=FONT_MED, color=(0, 255, 255))
        put_text_bg(panel, f"{fps:.1f} FPS", (8, 42), color=(180, 255, 180))
        put_text_bg(panel, status, (8, PANEL_H - 12), scale=FONT_SMALL, color=(200, 200, 255))
        cv2.rectangle(panel, (0, 0), (PANEL_W - 1, PANEL_H - 1), (0, 255, 255), 1)
        return panel

    def render_detection(self, frame_bgr: np.ndarray, result: dict) -> np.ndarray:
        panel = cv2.resize(frame_bgr, (PANEL_W, PANEL_H))
        objects = result.get("objects", [])

        for i, obj in enumerate(objects):
            color = self.OBJECT_COLORS[i % len(self.OBJECT_COLORS)]
            bbox = obj.get("bbox_norm", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = norm_to_px(*bbox, PANEL_W, PANEL_H)
                cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
                conf = obj.get("confidence", 0)
                label = f"{obj.get('label', '?')} {conf:.2f}"
                put_text_bg(panel, label, (x1, max(y1 - 5, 15)), color=color)

                # 잡을 수 있는 물체: 중심점 마커
                if obj.get("graspable"):
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cv2.drawMarker(panel, (cx, cy), color, cv2.MARKER_CROSS, 12, 2)

        # 씬 설명
        desc = result.get("scene_description", "")
        if desc:
            put_text_bg(panel, desc[:55], (8, PANEL_H - 12), scale=FONT_SMALL)

        latency = result.get("_latency_s", 0)
        header = f"OBJECT DETECTION  {latency:.2f}s"
        put_text_bg(panel, header, (8, 20), scale=FONT_MED, color=(0, 255, 0))
        put_text_bg(panel, f"{len(objects)} objects", (8, 42), color=(200, 255, 200))
        cv2.rectangle(panel, (0, 0), (PANEL_W - 1, PANEL_H - 1), (0, 255, 0), 1)
        return panel

    def render_action_plan(self, frame_bgr: np.ndarray, result: dict) -> np.ndarray:
        panel = cv2.resize(frame_bgr, (PANEL_W, PANEL_H))
        steps = result.get("action_steps", [])
        tgt = result.get("target_object", {})
        dst = result.get("destination", {})

        # 타겟 + 목적지 박스
        if tgt.get("bbox_norm") and len(tgt["bbox_norm"]) == 4:
            x1, y1, x2, y2 = norm_to_px(*tgt["bbox_norm"], PANEL_W, PANEL_H)
            cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 2)
            put_text_bg(panel, "PICK: " + tgt.get("label", "?"), (x1, max(y1 - 5, 15)),
                        color=(0, 255, 0))
        if dst.get("bbox_norm") and len(dst["bbox_norm"]) == 4:
            x1, y1, x2, y2 = norm_to_px(*dst["bbox_norm"], PANEL_W, PANEL_H)
            cv2.rectangle(panel, (x1, y1), (x2, y2), (255, 165, 0), 2)
            put_text_bg(panel, "PLACE: " + dst.get("label", "?"), (x1, max(y1 - 5, 15)),
                        color=(255, 165, 0))

        # 타겟 → 목적지 화살표
        if (tgt.get("bbox_norm") and dst.get("bbox_norm") and
                len(tgt["bbox_norm"]) == 4 and len(dst["bbox_norm"]) == 4):
            t = tgt["bbox_norm"]
            d = dst["bbox_norm"]
            pt1 = (int(t[0] * PANEL_W), int(t[1] * PANEL_H))
            pt2 = (int(d[0] * PANEL_W), int(d[1] * PANEL_H))
            cv2.arrowedLine(panel, pt1, pt2, (255, 255, 0), 2, tipLength=0.2)

        # 스텝 목록 (우측 하단 반투명 배경)
        if steps:
            overlay = panel.copy()
            box_h = min(len(steps), 7) * LINE_H + 12
            cv2.rectangle(overlay, (4, PANEL_H - box_h - 4), (PANEL_W - 4, PANEL_H - 4),
                          (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.6, panel, 0.4, 0, panel)
            for i, step in enumerate(steps[:7]):
                color = (0, 255, 0) if i == 0 else (180, 180, 180)
                text = f"{step['step']}. {step['action']}: {step.get('description','')[:28]}"
                cv2.putText(panel, text,
                            (10, PANEL_H - box_h + 4 + i * LINE_H),
                            FONT, FONT_SMALL, color, 1, cv2.LINE_AA)

        risk = result.get("risk_assessment", "?")
        risk_color = {"low": (0, 255, 0), "medium": (0, 165, 255), "high": (0, 0, 255)}.get(risk, (200, 200, 200))
        latency = result.get("_latency_s", 0)
        put_text_bg(panel, f"ACTION PLAN  {latency:.2f}s", (8, 20), scale=FONT_MED, color=(255, 165, 0))
        put_text_bg(panel, f"Risk: {risk}", (8, 42), color=risk_color)
        cv2.rectangle(panel, (0, 0), (PANEL_W - 1, PANEL_H - 1), (255, 165, 0), 1)
        return panel

    def render_reasoning_log(self, log_lines: list, result: dict) -> np.ndarray:
        panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        panel[:] = (18, 18, 30)  # 어두운 배경

        # 헤더
        cv2.putText(panel, "REASONING LOG", (8, 22), FONT, FONT_MED, (200, 200, 255), 1)
        latency = result.get("_latency_s", 0)
        model = result.get("_model", "")
        cv2.putText(panel, f"{model}  {latency:.2f}s", (8, 40), FONT, FONT_SMALL, (150, 150, 200), 1)
        cv2.line(panel, (0, 46), (PANEL_W, 46), (60, 60, 80), 1)

        # 로그 줄 (최신 N줄)
        max_lines = (PANEL_H - 60) // LINE_H
        visible = log_lines[-max_lines:]
        for i, line in enumerate(visible):
            y = 58 + i * LINE_H
            # 타임스탬프 있으면 색 다르게
            if line.startswith("["):
                color = (100, 200, 255)
            elif line.startswith("  ✅") or line.startswith("  OK"):
                color = (0, 255, 120)
            elif line.startswith("  ❌") or "error" in line.lower():
                color = (80, 80, 255)
            else:
                color = (200, 200, 200)
            cv2.putText(panel, line[:72], (8, y), FONT, FONT_SMALL, color, 1, cv2.LINE_AA)

        # 최신 결과에서 핵심 정보
        objects = result.get("objects", [])
        if objects:
            y = PANEL_H - 30
            names = ", ".join(o.get("label", "?") for o in objects[:4])
            cv2.putText(panel, f"Detected: {names}", (8, y), FONT, FONT_SMALL,
                        (0, 255, 150), 1, cv2.LINE_AA)

        cv2.rectangle(panel, (0, 0), (PANEL_W - 1, PANEL_H - 1), (80, 80, 120), 1)
        return panel


# ─── 대시보드 ─────────────────────────────────────────────────────────────────

class GeminiDashboard:
    def __init__(self, genai, model: str, interval: float = 2.0, save_video: str = None):
        self.genai = genai
        self.model = model
        self.interval = interval

        self.camera = CameraCapture()
        self.renderer = PanelRenderer()

        self.result_queue = queue.Queue(maxsize=1)
        self.log_lines = deque(maxlen=200)
        self.latest_result = {"_latency_s": 0, "_model": model}
        self.last_frame_bgr = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

        self._running = True
        self._fps_counter = deque(maxlen=30)

        self.writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.writer = cv2.VideoWriter(save_video, fourcc, 10.0, (DASH_W, DASH_H))
            print(f"[Video] Saving to {save_video}")

    def _log(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        for line in text.split("\n"):
            if line.strip():
                self.log_lines.append(f"[{ts}] {line}")

    def _gemini_worker(self):
        """별도 스레드에서 주기적으로 Gemini 호출"""
        from google import genai as genai_mod
        from pathlib import Path as _Path
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            token_path = _Path(__file__).parent.parent / "token"
            if token_path.exists():
                api_key = token_path.read_text().strip()
        client = genai_mod.Client(api_key=api_key)
        model_name = self.model

        while self._running:
            t0 = time.time()
            frame_rgb = self.camera.read()
            pil_img = Image.fromarray(frame_rgb.astype(np.uint8))

            self._log(f"Sending frame to Gemini ({self.model})...")
            try:
                combined_prompt = """
이미지를 분석하여 아래 JSON을 반환하세요 (코드블록 없이):
{
  "objects": [
    {
      "label": "...",
      "display_name": "...",
      "bbox_norm": [cx, cy, w, h],
      "graspable": true,
      "confidence": 0.9
    }
  ],
  "scene_description": "씬 요약",
  "target_object": {"label": "...", "bbox_norm": [cx, cy, w, h]},
  "destination": {"label": "...", "bbox_norm": [cx, cy, w, h]},
  "action_steps": [
    {"step": 1, "action": "...", "description": "..."}
  ],
  "risk_assessment": "low",
  "spatial_summary": "물체 공간 관계 요약"
}
약국 Pick & Place 씬 기준으로 분석하세요.
bbox_norm: [cx, cy, w, h] (0~1 정규화)
"""
                import re
                response = client.models.generate_content(model=model_name, contents=[combined_prompt, pil_img])
                text = response.text.strip()
                text = re.sub(r"```(?:json)?\s*", "", text)
                text = re.sub(r"```\s*$", "", text)
                start = text.find("{")
                end = text.rfind("}") + 1
                result = json.loads(text[start:end])
            except json.JSONDecodeError:
                result = {"raw": text[:300], "parse_error": True}
                self._log("  ❌ JSON parse error")
            except Exception as e:
                result = {"error": str(e)}
                self._log(f"  ❌ {e}")

            latency = time.time() - t0
            result["_latency_s"] = round(latency, 3)
            result["_model"] = self.model

            self._log(
                f"  ✅ {len(result.get('objects', []))} objects  "
                f"latency={latency:.2f}s"
            )

            # 결과 큐에 넣기 (가득 차있으면 버리기)
            try:
                self.result_queue.put_nowait((frame_rgb, result))
            except queue.Full:
                pass

            # 다음 호출까지 대기
            elapsed = time.time() - t0
            wait = max(0, self.interval - elapsed)
            for _ in range(int(wait * 10)):
                if not self._running:
                    break
                time.sleep(0.1)

    def run(self):
        worker = threading.Thread(target=self._gemini_worker, daemon=True)
        worker.start()
        self._log(f"Dashboard started | Model: {self.model} | Interval: {self.interval}s")
        self._log("  'q': 종료  |  's': 스크린샷")

        while True:
            t_frame = time.time()
            frame_rgb = self.camera.read()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self.last_frame_bgr = frame_bgr

            # 큐에 새 결과 있으면 업데이트
            try:
                snap_rgb, result = self.result_queue.get_nowait()
                self.latest_result = result
                snap_bgr = cv2.cvtColor(snap_rgb, cv2.COLOR_RGB2BGR)
            except queue.Empty:
                snap_bgr = frame_bgr
                result = self.latest_result

            # 패널 렌더링
            self._fps_counter.append(time.time())
            fps = len(self._fps_counter) / max(
                self._fps_counter[-1] - self._fps_counter[0], 1e-6
            )
            elapsed_since = time.time() - result.get("_ts", time.time())
            status = f"Interval: {self.interval}s | Model: {self.model}"

            p1 = self.renderer.render_live(frame_bgr, status, fps)
            p2 = self.renderer.render_detection(snap_bgr, result)
            p3 = self.renderer.render_action_plan(snap_bgr, result)
            p4 = self.renderer.render_reasoning_log(list(self.log_lines), result)

            # 4분할 조합
            top = np.hstack([p1, p2])
            bot = np.hstack([p3, p4])
            dashboard = np.vstack([top, bot])

            if self.writer:
                self.writer.write(dashboard)

            cv2.imshow("Gemini Embodied Reasoning Dashboard", dashboard)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_dir = Path("./results/gemini_dashboard")
                out_dir.mkdir(parents=True, exist_ok=True)
                path = str(out_dir / f"dashboard_{ts}.png")
                cv2.imwrite(path, dashboard)
                self._log(f"Screenshot saved: {path}")

        self._running = False
        self.camera.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        print("\n[Dashboard] 종료")


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Gemini Embodied Reasoning 대시보드")
    p.add_argument("--model", default="gemini-robotics-er-1.5-preview", help="Gemini 모델 ID")
    p.add_argument("--interval", type=float, default=2.0,
                   help="Gemini 호출 간격(초) (기본: 2.0)")
    p.add_argument("--save-video", default=None, help="영상 저장 경로 (.avi)")
    args = p.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        token_path = Path(__file__).parent.parent / "token"
        if token_path.exists():
            api_key = token_path.read_text().strip()
            os.environ["GOOGLE_API_KEY"] = api_key
        else:
            print("[Error] GOOGLE_API_KEY 환경변수를 설정하거나 groot/token 파일에 키를 저장하세요.")
            sys.exit(1)

    from google import genai
    dashboard = GeminiDashboard(genai, args.model, args.interval, args.save_video)
    dashboard.run()


if __name__ == "__main__":
    main()
