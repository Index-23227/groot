"""
Gemini ER → 두산 E0509 실행 브리지 (Closed-Loop, 다중 검증)

───────────────────────────────────────────────────────
왜 필요한가?
───────────────────────────────────────────────────────
Gemini ER 이 주는 것: 의미(semantic) 수준의 계획
  - bbox_norm, grasp_point_norm → "어떤 물체를 집을 것인가"
  - approach_direction, gripper_opening_mm → "어떻게 집을 것인가"
  - action_steps (pick/lift/move/place) → "어떤 순서로 할 것인가"

로봇이 필요한 것: 물리(physical) 수준의 명령
  - TCP 목표 위치 [x, y, z] (mm, 로봇 베이스 좌표계)
  - TCP 목표 방향 [rx, ry, rz] (deg)
  - 그리퍼 stroke (0~700)
  - movel / movej ROS2 서비스 호출

이 브리지가 하는 일:
  grasp_point_norm → 픽셀 → RealSense depth → 3D camera frame
  → T_cam2base → 로봇 베이스 좌표 → movel 명령

───────────────────────────────────────────────────────
왜 Gemini를 최대한 많이 쿼리하는가?
───────────────────────────────────────────────────────
Gemini ER은 이미지를 보고 실시간으로 판단한다.
한 번 계획하고 눈 감고 실행하면:
  - 물체가 조금 이동하면 잡기 실패
  - 그리퍼 정렬이 미세하게 틀려도 감지 불가
  - 예상치 못한 장애물 처리 불가

각 단계에서 새 이미지를 찍고 Gemini에게 물어보면:
  - 실시간 보정 가능
  - "지금 그리퍼가 물체 위에 제대로 정렬됐는가?" 확인
  - "물체를 잡았는가?" 확인
  - "놓을 위치가 비어있는가?" 확인

속도: 쿼리당 ~46초, 전체 pick-place당 5~6회 = 약 4~5분
데모 용도로 충분히 허용 가능.

───────────────────────────────────────────────────────
Gemini 쿼리 시점 (총 5회)
───────────────────────────────────────────────────────
  1. [초기 분석]   씬 전체 → 물체 감지, grasp point, action plan
  2. [pregrasp 전] 그리퍼가 물체 위에 있는가? 좌표 보정 필요한가?
  3. [grasp 후]    물체를 잡았는가? 그리퍼에 물체가 들어있는가?
  4. [이동 중]     물체를 여전히 들고 있는가? 경로에 장애물 있는가?
  5. [place 후]    태스크가 성공적으로 완료됐는가?

사용법:
  python utils/gemini_bridge.py --instruction "스프라이트 캔을 집어서 오른쪽으로 옮겨줘"
  python utils/gemini_bridge.py --dry-run    # 로봇 없이 변환 결과만 출력
"""

import sys, os, time, json, base64, argparse
from pathlib import Path
from io import BytesIO
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.doosan_e0509_config import *
from utils.calibration import load_calibration, pixel_depth_to_robot, matrix_to_euler_deg
from utils.calibration import _Rx, _Ry, _Rz


# ──────────────────────────────────────────────────────
# Gemini 쿼리 모듈
# ──────────────────────────────────────────────────────

def _load_gemini_key():
    token = Path(__file__).parent.parent / "token"
    if token.exists():
        for line in token.read_text().splitlines():
            line = line.strip()
            if line.startswith("AIza"):
                return line
    key = os.environ.get("GOOGLE_API_KEY", "")
    if key:
        return key
    raise RuntimeError("Google API 키 없음. groot/token 파일에 AIza... 키 입력")


def _pil_to_b64(pil_img, quality=90):
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _query_gemini(pil_img, prompt, model="gemini-robotics-er-1.5-preview"):
    """Gemini ER에 이미지 + 프롬프트 전송 → JSON dict 반환."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_load_gemini_key())
    resp = client.models.generate_content(
        model=model,
        contents=[prompt, pil_img],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    text = resp.text.strip()
    # JSON 파싱
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1:
        return {"raw": text}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {"raw": text}


# ──────────────────────────────────────────────────────
# Gemini 프롬프트 (단계별)
# ──────────────────────────────────────────────────────

def _prompt_initial_analysis(instruction):
    return f"""
당신은 로봇 조작 전문가입니다. 이미지를 보고 다음 태스크를 수행하기 위한 계획을 세우세요.

태스크: {instruction}

로봇 스펙:
- Doosan E0509 6-DOF 로봇 팔
- 그리퍼: ROBOTIS RH-P12-RN-A (최대 열림 120mm)
- 카메라: Intel RealSense D435 (위에서 아래를 내려다보는 각도로 설치)

다음 JSON 형식으로 정확하게 답하세요:
{{
  "target_object": "집을 물체 이름",
  "grasp_point_norm": [cx, cy],       // 이미지에서 물체 무게중심 (0~1 정규화)
  "bbox_norm": [cx, cy, w, h],        // 물체 bounding box (0~1)
  "approach_direction": [dx, dy, dz], // 그리퍼 접근 방향 단위벡터 (카메라 좌표계)
  "gripper_opening_mm": 70,           // 물체 크기에 맞는 그리퍼 열림 (50~120mm)
  "pregrasp_offset_m": 0.08,          // 물체 위 pregrasp 높이 (m)
  "place_region_norm": [cx, cy],      // 놓을 위치 (이미지 좌표 0~1), 없으면 null
  "collision_objects": ["방해물1"],   // 경로상 충돌 위험 물체
  "confidence": 0.9,                  // 전체 계획 신뢰도 0~1
  "reasoning": "계획 근거 설명"
}}
"""


def _prompt_verify_pregrasp(instruction, current_pos_mm):
    return f"""
로봇 그리퍼가 물체 위 pregrasp 위치로 이동했습니다.
현재 그리퍼 TCP 위치: {current_pos_mm} (로봇 베이스 좌표, mm)

태스크: {instruction}

이미지를 보고 다음을 평가하세요:
1. 그리퍼가 물체 바로 위에 정렬되어 있는가?
2. 그리퍼 방향이 물체를 잡기에 적합한가?
3. 좌표 보정이 필요한가?

JSON으로 답하세요:
{{
  "aligned": true/false,              // 정렬이 올바른가
  "correction_needed": true/false,    // 보정 필요 여부
  "grasp_point_norm": [cx, cy],       // 보정된 grasp point (변화 없으면 기존값)
  "approach_direction": [dx, dy, dz], // 보정된 접근 방향
  "issue": "문제점 설명 또는 null",
  "confidence": 0.9
}}
"""


def _prompt_verify_grasp(instruction):
    return f"""
로봇 그리퍼가 물체를 잡으려고 닫혔습니다.
태스크: {instruction}

이미지를 보고 다음을 평가하세요:
1. 그리퍼 안에 물체가 있는가?
2. 물체가 안정적으로 잡혀 있는가?
3. 물체가 기울어지거나 미끄러질 위험이 있는가?

JSON으로 답하세요:
{{
  "grasped": true/false,              // 물체를 성공적으로 잡았는가
  "stable": true/false,               // 안정적인 파지인가
  "object_visible": true/false,       // 그리퍼에 물체가 보이는가
  "retry_grasp": true/false,          // 재시도 필요 여부
  "issue": "문제점 또는 null",
  "confidence": 0.9
}}
"""


def _prompt_verify_in_transit(instruction):
    return f"""
로봇이 물체를 들고 목표 위치로 이동 중입니다.
태스크: {instruction}

이미지를 보고 다음을 평가하세요:
1. 물체가 여전히 그리퍼에 안전하게 있는가?
2. 이동 경로에 새로운 장애물이 생겼는가?
3. 목표 위치(놓을 곳)가 비어있는가?

JSON으로 답하세요:
{{
  "object_secured": true/false,       // 물체를 여전히 들고 있는가
  "path_clear": true/false,           // 경로 장애물 없음
  "place_area_clear": true/false,     // 놓을 위치가 비어있는가
  "place_point_norm": [cx, cy],       // 놓을 위치 (보정 또는 재확인)
  "issue": "문제점 또는 null",
  "confidence": 0.9
}}
"""


def _prompt_verify_completion(instruction):
    return f"""
로봇이 물체를 목표 위치에 놓고 그리퍼를 열었습니다.
태스크: {instruction}

이미지를 보고 최종 평가를 하세요:
1. 태스크가 성공적으로 완료됐는가?
2. 물체가 목표 위치에 안정적으로 놓여 있는가?
3. 다음 태스크가 필요한가?

JSON으로 답하세요:
{{
  "success": true/false,              // 전체 태스크 성공
  "object_placed": true/false,        // 물체가 제자리에 있는가
  "stable": true/false,               // 물체가 안정적인가
  "next_action_needed": true/false,   // 추가 동작 필요 여부
  "summary": "결과 요약",
  "score": 85                         // 성공도 0~100
}}
"""


# ──────────────────────────────────────────────────────
# RealSense Depth + Color 동시 캡처
# ──────────────────────────────────────────────────────

class RealSenseCapture:
    """RGB + Depth 동시 캡처. 기존 CameraCapture는 color만 제공해서 별도 클래스."""

    def __init__(self):
        import pyrealsense2 as rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT,
                          rs.format.rgb8, CAMERA_FPS)
        cfg.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT,
                          rs.format.z16, CAMERA_FPS)
        self.pipeline.start(cfg)

        # 내부 파라미터 (캘리브레이션 파일보다 실기기가 정확)
        profile = self.pipeline.get_active_profile()
        intr = (profile.get_stream(rs.stream.color)
                       .as_video_stream_profile().get_intrinsics())
        self.camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                                       [0, intr.fy, intr.ppy],
                                       [0, 0, 1]])
        # Align depth to color
        import pyrealsense2 as rs2
        self._align = rs2.align(rs2.stream.color)
        print(f"[RealSense] fx={intr.fx:.1f} fy={intr.fy:.1f}")

    def read(self):
        """RGB (H,W,3 uint8), depth (H,W float32 m) 반환."""
        frames = self._align.process(self.pipeline.wait_for_frames())
        rgb   = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data()) * 0.001  # mm→m
        return rgb, depth

    def get_depth_at(self, u, v, depth_frame, radius=3):
        """픽셀 (u,v) 주변 radius 내 유효 depth 중앙값 (m)."""
        u, v = int(u), int(v)
        h, w = depth_frame.shape
        u0, u1 = max(0, u-radius), min(w, u+radius+1)
        v0, v1 = max(0, v-radius), min(h, v+radius+1)
        patch = depth_frame[v0:v1, u0:u1]
        valid = patch[patch > 0.1]  # 10cm 이상만 유효
        if len(valid) == 0:
            return None
        return float(np.median(valid))

    def release(self):
        self.pipeline.stop()


# ──────────────────────────────────────────────────────
# 좌표 변환
# ──────────────────────────────────────────────────────

def approach_dir_to_tcp_euler(approach_dir):
    """
    approach_direction (단위벡터) → TCP RPY (deg).

    Gemini가 주는 approach_direction은 카메라 좌표계 기준으로
    그리퍼가 물체에 접근하는 방향 벡터.

    로봇 TCP z축이 approach_direction과 일치하도록 회전행렬 계산.
    'from above' [0, 0, -1] → TCP가 위에서 아래를 향함 → rx=180, ry=0, rz=0
    """
    d = np.array(approach_dir, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-6:
        return np.array([180.0, 0.0, 0.0])
    d = d / norm

    # TCP z축 = 접근 방향 (음수: 물체 쪽으로)
    z_tcp = -d  # 그리퍼가 d 방향으로 접근 → TCP z는 반대 방향

    # x, y축 임의 선택 (z와 수직)
    ref = np.array([1.0, 0.0, 0.0]) if abs(z_tcp[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_tcp = np.cross(z_tcp, ref)
    x_tcp /= np.linalg.norm(x_tcp)
    y_tcp = np.cross(z_tcp, x_tcp)

    R = np.column_stack([x_tcp, y_tcp, z_tcp])
    return matrix_to_euler_deg(R)


def build_tcp_pose(pos_mm, euler_deg):
    """[x,y,z] mm + [rx,ry,rz] deg → Doosan pose list (6-dim)."""
    return list(pos_mm) + list(euler_deg)


# ──────────────────────────────────────────────────────
# Doosan Cartesian 제어 (movel)
# ──────────────────────────────────────────────────────

class DoosanCartesianRobot:
    """
    ROS2 movel 명령으로 Doosan E0509 Cartesian 제어.

    왜 movel인가?
    - Gemini 출력이 "어디로 이동" (TCP 위치) 이므로
      카르테시안 직선 이동(movel)이 직관적으로 맞음
    - movej는 관절각이 필요 → IK를 직접 구현해야 함
    - Doosan의 내장 IK를 movel이 사용 → 우리가 IK 구현 불필요
    """

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.connected = False
        self._grip_pub = None
        self._movel_client = None
        self._tcp_client = None

        if dry_run:
            print("[Robot] DRY-RUN 모드 (실제 로봇 명령 없음)")
            return

        try:
            import rclpy
            from std_msgs.msg import Int32
            try:
                rclpy.init()
            except RuntimeError:
                pass
            self.node = rclpy.create_node("gemini_bridge")

            # 그리퍼 연속 제어 publisher
            self._grip_pub = self.node.create_publisher(
                Int32, GRIPPER_POSITION_TOPIC, 10)

            # movel 서비스 클라이언트
            try:
                from dsr_msgs2.srv import MoveLine, GetCurrentPosx
                self._movel_client = self.node.create_client(
                    MoveLine, MOVE_LINE_SERVICE)
                self._tcp_client = self.node.create_client(
                    GetCurrentPosx, GET_TCP_SERVICE)
            except Exception as e:
                print(f"[Robot] dsr_msgs2 없음 (mock): {e}")

            self.connected = True
            print("[Robot] ROS2 연결됨")
        except ImportError:
            print("[Robot] ROS2 없음 — MOCK 모드")

    def get_tcp_pose(self):
        """현재 TCP pose [x,y,z,rx,ry,rz] (mm, deg) 반환."""
        if self.dry_run or not self.connected or self._tcp_client is None:
            return [350.0, 0.0, 300.0, 180.0, 0.0, 0.0]  # 기본 home pose

        import rclpy
        from dsr_msgs2.srv import GetCurrentPosx
        req = GetCurrentPosx.Request()
        req.ref = 0  # BASE 좌표계
        future = self._tcp_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=3.0)
        if future.done():
            return list(future.result().pos)
        return [350.0, 0.0, 300.0, 180.0, 0.0, 0.0]

    def movel(self, pose_6d, vel=50.0, acc=50.0, wait=True):
        """
        TCP를 목표 pose로 직선 이동.
        pose_6d: [x,y,z(mm), rx,ry,rz(deg)]
        """
        px, py, pz = pose_6d[:3]
        rx, ry, rz = pose_6d[3:]
        print(f"[movel] → pos=({px:.1f},{py:.1f},{pz:.1f})mm "
              f"orient=({rx:.1f},{ry:.1f},{rz:.1f})deg vel={vel}")

        if self.dry_run or not self.connected or self._movel_client is None:
            time.sleep(0.3)
            return True

        import rclpy
        from dsr_msgs2.srv import MoveLine
        req = MoveLine.Request()
        req.pos  = list(pose_6d)
        req.vel  = [vel, vel]
        req.acc  = [acc, acc]
        req.time = 0.0
        req.radius    = 0.0
        req.ref       = 0   # BASE
        req.mode      = 0   # MOVE_MODE_ABSOLUTE
        req.blendType = 0
        req.syncType  = 1 if wait else 0

        future = self._movel_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=30.0)
        return future.done() and future.result().success

    def set_gripper(self, stroke, wait_sec=1.5):
        """그리퍼 stroke (0=열림, 700=닫힘) 설정."""
        stroke = int(np.clip(stroke, GRIPPER_STROKE_MIN, GRIPPER_STROKE_MAX))
        state  = "닫힘" if stroke > 350 else "열림"
        print(f"[gripper] stroke={stroke} ({state})")

        if self.dry_run:
            time.sleep(0.2)
            return

        if self.connected and self._grip_pub is not None:
            from std_msgs.msg import Int32
            self._grip_pub.publish(Int32(data=stroke))
            time.sleep(wait_sec)

    def shutdown(self):
        if self.connected:
            import rclpy
            self.node.destroy_node()
            rclpy.shutdown()


# ──────────────────────────────────────────────────────
# 핵심 브리지: Gemini 출력 → 로봇 명령
# ──────────────────────────────────────────────────────

class GeminiBridge:
    """
    Gemini ER ↔ Doosan E0509 변환 + 폐루프 실행.

    매 단계마다 새 이미지를 캡처하고 Gemini에게 확인받는 구조.
    """

    def __init__(self, T_cam2base, camera_matrix, dry_run=False):
        self.T    = T_cam2base      # 4×4, 카메라→로봇 변환
        self.K    = camera_matrix   # 3×3, 카메라 내부 파라미터
        self.dry_run = dry_run
        self.robot   = DoosanCartesianRobot(dry_run=dry_run)
        self.query_count = 0
        self.query_log   = []

    def _norm_to_pixel(self, norm_xy):
        cx, cy = norm_xy
        return cx * CAMERA_WIDTH, cy * CAMERA_HEIGHT

    def _gemini_query(self, rgb, prompt, label=""):
        """Gemini 쿼리 + 로그 기록."""
        from PIL import Image
        pil = Image.fromarray(rgb)
        t0  = time.time()
        print(f"\n[Gemini {self.query_count+1}] {label} — 쿼리 중...")
        result = _query_gemini(pil, prompt)
        elapsed = time.time() - t0
        self.query_count += 1
        self.query_log.append({
            "step": label,
            "result": result,
            "latency_s": round(elapsed, 2),
        })
        print(f"  → {elapsed:.1f}s | {json.dumps(result, ensure_ascii=False)[:200]}")
        return result

    def gemini_to_robot_pose(self, gemini_out, depth_frame):
        """
        Gemini 초기 분석 결과 → 로봇 grasp TCP pose.

        grasp_point_norm  → pixel → depth lookup → 3D cam → 로봇 베이스
        approach_direction → TCP orientation
        gripper_opening_mm → stroke
        """
        # 1. grasp point → 픽셀
        gp = gemini_out.get("grasp_point_norm", [0.5, 0.5])
        u, v = self._norm_to_pixel(gp)

        # 2. depth lookup (RealSense)
        if depth_frame is not None:
            h, w = depth_frame.shape
            ui, vi = int(np.clip(u, 0, w-1)), int(np.clip(v, 0, h-1))
            radius = 5
            patch = depth_frame[max(0,vi-radius):vi+radius+1,
                                 max(0,ui-radius):ui+radius+1]
            valid = patch[patch > 0.1]
            depth_m = float(np.median(valid)) if len(valid) > 0 else 0.5
        else:
            depth_m = 0.45  # fallback

        # 3. 3D 카메라 좌표 → 로봇 베이스 좌표
        pos_mm = pixel_depth_to_robot(u, v, depth_m, self.T, self.K)

        # 4. approach_direction → TCP orientation
        approach = gemini_out.get("approach_direction", [0, 0, -1])
        euler_deg = approach_dir_to_tcp_euler(approach)

        # 5. gripper stroke
        grip_mm    = gemini_out.get("gripper_opening_mm", 70)
        grip_ratio = np.clip(grip_mm / 120.0, 0.0, 1.0)
        stroke_open = int(grip_ratio * GRIPPER_STROKE_MAX)  # 열림 크기

        # 6. pregrasp pose (물체 위)
        offset_m   = gemini_out.get("pregrasp_offset_m", 0.08)
        # approach_dir의 반대 방향으로 offset 적용 (위쪽)
        d = np.array(approach, dtype=float)
        if np.linalg.norm(d) > 0:
            d = d / np.linalg.norm(d)
        pregrasp_pos = pos_mm - d * offset_m * 1000.0  # mm

        grasp_pose   = build_tcp_pose(pos_mm,      euler_deg)
        pregrasp_pose = build_tcp_pose(pregrasp_pos, euler_deg)

        return {
            "grasp_pose":    grasp_pose,
            "pregrasp_pose": pregrasp_pose,
            "stroke_open":   stroke_open,
            "pos_mm":        pos_mm,
            "depth_m":       depth_m,
            "pixel":         (u, v),
        }

    def execute(self, instruction, camera):
        """
        메인 실행 루프. Gemini를 5회 쿼리하며 pick-place 수행.

        Args:
            instruction: 자연어 태스크 ("스프라이트 캔을 집어서 오른쪽으로 옮겨줘")
            camera: RealSenseCapture 인스턴스
        """
        print(f"\n{'='*60}")
        print(f"  태스크: {instruction}")
        print(f"{'='*60}")

        # ── Query 1: 초기 씬 분석 ───────────────────────────────
        rgb, depth = camera.read()
        plan = self._gemini_query(
            rgb,
            _prompt_initial_analysis(instruction),
            label="초기 씬 분석"
        )

        if not plan.get("target_object"):
            print("❌ Gemini가 물체를 감지하지 못했습니다.")
            return False

        print(f"\n  대상 물체: {plan.get('target_object')}")
        print(f"  신뢰도: {plan.get('confidence', '?')}")

        robot_cmd = self.gemini_to_robot_pose(plan, depth)
        print(f"\n  grasp 위치: {robot_cmd['pos_mm']} mm")
        print(f"  pregrasp 위치: {robot_cmd['pregrasp_pose'][:3]} mm")

        # ── 그리퍼 열기 & pregrasp로 이동 ─────────────────────
        grip_mm = plan.get("gripper_opening_mm", 70)
        self.robot.set_gripper(grip_to_stroke(0.0))  # 완전 열림
        print(f"\n[이동] pregrasp 위치로...")
        self.robot.movel(robot_cmd["pregrasp_pose"], vel=60, acc=60)

        # ── Query 2: pregrasp 정렬 확인 ────────────────────────
        rgb, depth = camera.read()
        tcp_now = self.robot.get_tcp_pose()
        verify2 = self._gemini_query(
            rgb,
            _prompt_verify_pregrasp(instruction, tcp_now[:3]),
            label="pregrasp 정렬 확인"
        )

        # 보정이 필요하면 grasp point 업데이트
        if verify2.get("correction_needed") and "grasp_point_norm" in verify2:
            print("  [보정] Gemini가 grasp point를 보정했습니다.")
            plan["grasp_point_norm"] = verify2["grasp_point_norm"]
            if "approach_direction" in verify2:
                plan["approach_direction"] = verify2["approach_direction"]
            robot_cmd = self.gemini_to_robot_pose(plan, depth)

        if not verify2.get("aligned", True):
            print(f"  ⚠️  정렬 문제: {verify2.get('issue', '?')}")

        # ── grasp 위치로 이동 & 그리퍼 닫기 ──────────────────
        print(f"\n[이동] grasp 위치로...")
        self.robot.movel(robot_cmd["grasp_pose"], vel=30, acc=30)  # 느리게
        self.robot.set_gripper(grip_to_stroke(1.0), wait_sec=2.0)  # 완전 닫힘

        # ── Query 3: grasp 성공 확인 ───────────────────────────
        rgb, depth = camera.read()
        verify3 = self._gemini_query(
            rgb,
            _prompt_verify_grasp(instruction),
            label="파지(grasp) 성공 확인"
        )

        if not verify3.get("grasped", True):
            print(f"  ⚠️  파지 실패: {verify3.get('issue', '?')}")
            if verify3.get("retry_grasp"):
                print("  → 그리퍼 열고 재시도...")
                self.robot.set_gripper(grip_to_stroke(0.0))
                self.robot.movel(robot_cmd["pregrasp_pose"], vel=40, acc=40)
                self.robot.movel(robot_cmd["grasp_pose"], vel=20, acc=20)
                self.robot.set_gripper(grip_to_stroke(1.0), wait_sec=2.0)

        # ── Lift ──────────────────────────────────────────────
        print("\n[이동] 물체 들어올리는 중...")
        lift_pose = list(robot_cmd["grasp_pose"])
        lift_pose[2] += 150  # Z + 150mm
        self.robot.movel(lift_pose, vel=40, acc=40)

        # ── Query 4: 이동 중 물체 상태 확인 ──────────────────
        rgb, depth = camera.read()
        verify4 = self._gemini_query(
            rgb,
            _prompt_verify_in_transit(instruction),
            label="이동 중 상태 확인"
        )

        if not verify4.get("object_secured", True):
            print(f"  ⚠️  물체 상태 위험: {verify4.get('issue', '?')}")

        # place 위치 결정
        place_norm = verify4.get("place_point_norm") or plan.get("place_region_norm")
        if place_norm:
            place_rgb, place_depth = camera.read()
            tmp_plan = {"grasp_point_norm": place_norm,
                        "approach_direction": [0, 0, -1],
                        "gripper_opening_mm": 70,
                        "pregrasp_offset_m": 0.08}
            place_cmd = self.gemini_to_robot_pose(tmp_plan, place_depth)
            place_pose    = place_cmd["grasp_pose"]
            preplace_pose = place_cmd["pregrasp_pose"]
        else:
            # place 위치를 Gemini가 지정 못했으면 현재 위치에서 옆으로
            place_pose    = list(lift_pose)
            place_pose[1] += 200  # Y + 200mm (오른쪽)
            preplace_pose = list(place_pose)
            preplace_pose[2] += 100

        # ── place 이동 & 그리퍼 열기 ──────────────────────────
        print("\n[이동] place 위치로...")
        self.robot.movel(preplace_pose, vel=50, acc=50)
        self.robot.movel(place_pose,    vel=30, acc=30)
        self.robot.set_gripper(grip_to_stroke(0.0), wait_sec=1.5)

        # ── 후퇴 ──────────────────────────────────────────────
        retreat_pose = list(place_pose)
        retreat_pose[2] += 150
        self.robot.movel(retreat_pose, vel=50, acc=50)

        # ── Query 5: 최종 완료 확인 ───────────────────────────
        rgb, depth = camera.read()
        verify5 = self._gemini_query(
            rgb,
            _prompt_verify_completion(instruction),
            label="태스크 완료 확인"
        )

        success = verify5.get("success", False)
        score   = verify5.get("score", 0)
        summary = verify5.get("summary", "")

        print(f"\n{'='*60}")
        print(f"  결과: {'✅ 성공' if success else '⚠️  부분 성공/실패'}")
        print(f"  점수: {score}/100")
        print(f"  요약: {summary}")
        print(f"  총 Gemini 쿼리: {self.query_count}회")
        total_latency = sum(q["latency_s"] for q in self.query_log)
        print(f"  총 Gemini 소요: {total_latency:.1f}초")
        print(f"{'='*60}")

        return success, self.query_log


# ──────────────────────────────────────────────────────
# 메인 진입점
# ──────────────────────────────────────────────────────

def main(args):
    # 캘리브레이션 로드
    try:
        T_cam2base, intr = load_calibration()
        K = intr.get("camera_matrix")
        if K is None:
            raise ValueError("camera_matrix 없음")
        print(f"[Calib] T_cam2base 로드됨")
    except Exception as e:
        print(f"⚠️  캘리브레이션 없음: {e}")
        if not args.dry_run:
            print("먼저: python utils/calibration.py --run")
            return
        # dry-run용 더미
        T_cam2base = np.eye(4)
        T_cam2base[:3, 3] = [0.5, 0.0, 0.3]  # 카메라가 로봇 앞 50cm 위
        K = np.array([[615, 0, 320], [0, 615, 240], [0, 0, 1]], dtype=float)
        print("[Calib] 더미 캘리브레이션으로 실행 (dry-run)")

    # 카메라
    if args.dry_run:
        camera = None
        class MockCamera:
            camera_matrix = K
            def read(self):
                rgb = np.random.randint(100, 200, (CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                depth = np.ones((CAMERA_HEIGHT, CAMERA_WIDTH), dtype=np.float32) * 0.45
                return rgb, depth
            def release(self): pass
        camera = MockCamera()
    else:
        camera = RealSenseCapture()
        K = camera.camera_matrix  # 실기기 intrinsics 우선

    bridge = GeminiBridge(T_cam2base, K, dry_run=args.dry_run)

    try:
        bridge.execute(args.instruction, camera)
    finally:
        if not args.dry_run and camera:
            camera.release()
        bridge.robot.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Gemini ER → Doosan 실행 브리지")
    p.add_argument("--instruction", "-i",
                   default="스프라이트 캔을 집어서 오른쪽으로 옮겨줘",
                   help="자연어 태스크 지시")
    p.add_argument("--dry-run", action="store_true",
                   help="로봇 없이 변환 결과만 출력 (mock 모드)")
    main(p.parse_args())
