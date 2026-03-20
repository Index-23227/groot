# Track 2: Gemini Robotics-ER 1.5 + cuRobo + RGBD Pipeline

> **역할**: VLA 학습 없이 Zero-shot으로 Pick & Place를 수행하는 Classical+AI 파이프라인
> **담당**: P5 (또는 Gemini-ER 전담 인력)
> **소요 시간**: 약 3~4시간 (셋업~첫 실물 테스트)
> **GPU 사용**: 0장 (Gemini-ER은 클라우드 API, cuRobo는 CPU도 가능)

---

## 0. 아키텍처 개요

```
[사용자 자연어 명령]
    ↓
[Gemini Robotics-ER 1.5]  ← Google AI Studio API (클라우드)
    ├── Task Decomposition: "step 1: pick blue bottle → step 2: place in tray 1"
    ├── Object Pointing: (pixel_x, pixel_y) 2D 좌표
    └── Progress Check: 매 step 후 성공 여부 판단
    ↓
[RGBD 카메라]  ← depth로 Z축 직접 측정
    ├── 2D (pixel_x, pixel_y) + depth → 3D (cam_x, cam_y, cam_z)
    └── Extrinsic calibration → 3D (robot_x, robot_y, robot_z)
    ↓
[cuRobo IK Solver]  ← URDF (e0509_gripper_description) 기반
    ├── 3D target pose → joint angles [j1, j2, j3, j4, j5, j6]
    └── Collision-aware motion planning
    ↓
[두산 E0509 + RH-P12-RN-A]  ← ROS2 MoveJoint + Gripper Service
    ├── servoj로 이동
    ├── gripper open/close
    └── gripper stroke 피드백 → grasp 성공 판별
    ↓
[Gemini-ER Progress Check]  ← 성공 → 다음 step / 실패 → retry
```

### 왜 이 파이프라인이 가치 있는가

1. **VLA 학습 없이 즉시 동작** — 데모 수집, fine-tuning 대기 시간 제로
2. **Gemini-ER의 SOTA reasoning** — 15개 embodied reasoning 벤치마크 1위
3. **Multi-step task 자동 분해** — "약병들을 색깔별로 분류해" 같은 복합 명령 처리 가능
4. **Progress estimation으로 자동 retry** — 실패 감지 + 재시도 로직 내장
5. **발표 임팩트** — "Gemini가 생각하고 로봇이 실행하는 Dual-Brain" 스토리

### Track 1 (VLA)과의 관계

- Track 2는 **VLA 학습이 돌아가는 동안 먼저 실물 테스트 가능**
- VLA가 성공하면: "두 접근법 비교" 발표 (더 강력)
- VLA가 실패하면: Track 2가 **시연 보장 백업**
- 이상적 구조: **Gemini-ER가 task planner, VLA가 executor** (Dual-Brain)

---

## 1. 사전 준비 (해커톤 전)

### 1.1 Gemini API Key 확보

```bash
# Google AI Studio에서 API Key 발급
# https://aistudio.google.com/apikey
# 모델명: gemini-2.0-flash 또는 gemini-robotics-er-1.5 (preview)
export GEMINI_API_KEY="your-api-key-here"
```

> **비용**: Gemini Flash는 무료 티어 존재 (분당 15 requests). 해커톤 기간 내 충분.
> **주의**: Gemini Robotics-ER 1.5는 preview 상태. 접근이 안 되면 `gemini-2.0-flash`로 대체 가능 (pointing 정밀도는 다소 떨어짐).

### 1.2 Python 환경 준비

```bash
# Track 2 전용 가상환경 (VLA 환경과 분리)
conda create -n track2 python=3.10 -y
conda activate track2

pip install google-generativeai opencv-python-headless numpy scipy
pip install pyrealsense2  # Intel RealSense 사용 시
# pip install open3d     # point cloud 시각화 (선택)
```

### 1.3 cuRobo 설치

```bash
# 방법 A: pip install (standalone, Isaac Sim 불필요)
pip install curobo

# 방법 B: source build (더 안정적)
git clone https://github.com/NVlabs/curobo.git
cd curobo
pip install -e .
```

> **⚠️ cuRobo 셋업이 안 될 경우 대체**: `ikpy` (순수 Python IK solver)
> ```bash
> pip install ikpy
> ```
> ikpy는 cuRobo보다 느리고 collision-aware가 아니지만, 셋업이 확실히 됨.

### 1.4 로봇 description 패키지 클론

```bash
cd ~/doosan_ws/src
git clone https://github.com/fhekwn549/e0509_gripper_description.git
git clone -b humble https://github.com/fhekwn549/doosan-robot2.git
git clone https://github.com/ROBOTIS-GIT/RH-P12-RN-A.git

cd ~/doosan_ws
colcon build --symlink-install
source install/setup.bash
```

---

## 2. 하드웨어 정보 (확정)

### 2.1 두산 E0509

| 항목 | 값 |
|------|-----|
| DOF | 6 |
| Joint names | `joint_1` ~ `joint_6` |
| Joint unit | **degree** (⚠️ VLA는 radian, 변환 필수) |
| Controller IP | `192.168.137.100` (현장 확인) |
| Controller port | `12345` |
| MoveJoint service | `/dsr01/motion/move_joint` |
| Joint state topic | `/joint_states` (10개: arm 6 + gripper 4) |

### 2.2 ROBOTIS RH-P12-RN-A 그리퍼

| 항목 | 값 |
|------|-----|
| DOF | 1 (open/close) |
| Joint names | `gripper_rh_r1`, `gripper_rh_r2`, `gripper_rh_l1`, `gripper_rh_l2` (4개 동일 값) |
| 통신 | Modbus RTU via Tool Flange Serial |
| Stroke 범위 | 0 (열림) ~ 700 (완전 닫힘) |
| Open service | `/dsr01/gripper/open` (std_srvs/Trigger) |
| Close service | `/dsr01/gripper/close` (std_srvs/Trigger) |
| Position topic | `/dsr01/gripper/position_cmd` (std_msgs/Int32, 0~700) |
| Stroke feedback | `/dsr01/gripper/stroke` (std_msgs/Int32) |

### 2.3 RGBD 카메라

| 항목 | 값 |
|------|-----|
| 설치 방식 | Eye-to-Hand (외부 고정) |
| 뷰 | Front view (global view 1대) |
| 해상도 | 640×480 (확인 필요) |
| 출력 | RGB + Depth |

#### 카메라 배치 가이드

```
        [카메라] ← 약 50-70cm 높이, 45° 하향
          / 
         /
        ▼
  [작업대 + 물체들]
  [  두산 로봇   ]
```

- 로봇 workspace 전체 + 그리퍼가 한 프레임에 들어오게
- 물체들이 서로 겹치지 않는 각도
- 물체가 이미지의 ~20-30% 차지하도록
- **한 번 고정하면 절대 움직이지 않기**

---

## 3. 단계별 구현 가이드

### Phase 1: 카메라 + Extrinsic Calibration (30분~1시간)

#### 3.1 카메라 스트림 확인

```python
# camera_test.py
import cv2
import numpy as np

# OpenCV 카메라 (일반 USB)
cap = cv2.VideoCapture(0)

# Intel RealSense인 경우:
# import pyrealsense2 as rs
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# pipeline.start(config)

while True:
    ret, frame = cap.read()
    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

#### 3.2 Camera Intrinsics 확인

```python
# RealSense의 경우 자동 획득:
# profile = pipeline.get_active_profile()
# intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
# fx, fy = intr.fx, intr.fy
# cx, cy = intr.ppx, intr.ppy

# 수동 설정 (카메라 스펙에서):
CAMERA_INTRINSICS = {
    "fx": 615.0,   # focal length x (pixel)
    "fy": 615.0,   # focal length y (pixel)
    "cx": 320.0,   # principal point x (pixel)
    "cy": 240.0,   # principal point y (pixel)
}
```

#### 3.3 Extrinsic Calibration (카메라→로봇 변환)

**방법: 4점 수동 매핑 (가장 빠름, ~10분)**

```python
# calibration.py
import numpy as np
from scipy.spatial.transform import Rotation

def calibrate_4point(pixel_points, depth_values, robot_points, intrinsics):
    """
    4개 점의 (pixel, depth) ↔ (robot xyz) 매핑으로 T_cam2base 계산
    
    사용법:
    1. 로봇 TCP를 작업대 위 4개 점으로 이동 (teach pendant)
    2. 각 점에서:
       - robot_points에 TCP xyz (mm) 기록
       - 카메라 이미지에서 해당 점의 pixel (x, y) 기록
       - depth 이미지에서 depth 값 (mm) 기록
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    
    # Pixel + depth → camera 3D
    cam_points = []
    for (px, py), d in zip(pixel_points, depth_values):
        x = (px - cx) * d / fx
        y = (py - cy) * d / fy
        z = d
        cam_points.append([x, y, z])
    
    cam_points = np.array(cam_points)  # (4, 3) in camera frame
    robot_points = np.array(robot_points)  # (4, 3) in robot base frame
    
    # Rigid transform: robot = R @ cam + t
    cam_centroid = cam_points.mean(axis=0)
    rob_centroid = robot_points.mean(axis=0)
    
    H = (cam_points - cam_centroid).T @ (robot_points - rob_centroid)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    t = rob_centroid - R @ cam_centroid
    
    # 4x4 변환 행렬
    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = R
    T_cam2base[:3, 3] = t
    
    # 검증: 4개 점의 변환 오차 출력
    for i in range(4):
        transformed = R @ cam_points[i] + t
        error = np.linalg.norm(transformed - robot_points[i])
        print(f"  Point {i}: error = {error:.1f} mm")
    
    return T_cam2base


def pixel_depth_to_robot(px, py, depth, intrinsics, T_cam2base):
    """단일 픽셀 + depth → 로봇 베이스 프레임 3D 좌표"""
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    
    # Pixel → camera 3D
    cam_x = (px - cx) * depth / fx
    cam_y = (py - cy) * depth / fy
    cam_z = depth
    cam_point = np.array([cam_x, cam_y, cam_z, 1.0])
    
    # Camera → robot base
    robot_point = T_cam2base @ cam_point
    return robot_point[:3]  # [x, y, z] in robot base frame (mm)


# ===== 현장에서 실행 =====
if __name__ == "__main__":
    # 1. 로봇 TCP를 4개 점으로 이동하며 기록
    #    (teach pendant에서 TCP 좌표 읽기)
    pixel_points = [
        (150, 120),  # 왼쪽 위
        (490, 120),  # 오른쪽 위
        (150, 360),  # 왼쪽 아래
        (490, 360),  # 오른쪽 아래
    ]
    
    depth_values = [
        650,  # mm (depth 이미지에서 읽기)
        655,
        648,
        652,
    ]
    
    robot_points = [
        [300, -200, 50],   # mm (teach pendant에서 TCP xyz 읽기)
        [300, 200, 50],
        [500, -200, 50],
        [500, 200, 50],
    ]
    
    intrinsics = CAMERA_INTRINSICS
    
    T = calibrate_4point(pixel_points, depth_values, robot_points, intrinsics)
    print("\nT_cam2base:")
    print(T)
    
    # 저장
    np.save("T_cam2base.npy", T)
    print("\nSaved to T_cam2base.npy")
```

**캘리브레이션 절차 (현장):**

1. 로봇을 bringup: `ros2 launch e0509_gripper_description bringup.launch.py mode:=real host:=<IP>`
2. Teach pendant로 TCP를 작업대 위 4개 꼭짓점으로 이동
3. 각 점에서: TCP 좌표 기록 + 카메라 이미지의 pixel 좌표 기록 + depth 값 기록
4. `calibration.py` 실행 → `T_cam2base.npy` 저장
5. 검증: 각 점의 변환 오차가 **10mm 이내**면 OK

---

### Phase 2: Gemini-ER 연동 (30분)

#### 3.4 Gemini-ER Pointing (물체 2D 좌표)

```python
# gemini_er_client.py
import google.generativeai as genai
import json
import re
import cv2
import numpy as np
from PIL import Image
import io

genai.configure(api_key="YOUR_GEMINI_API_KEY")

# Gemini Robotics-ER 1.5 (preview 접근 가능 시)
# model = genai.GenerativeModel("gemini-robotics-er-1.5")

# 대체: Gemini 2.5 Flash (항상 접근 가능, pointing도 가능)
model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")


def get_pick_point(image_bgr, instruction):
    """
    이미지에서 물체의 pick point (pixel x, y)를 반환.
    
    Args:
        image_bgr: OpenCV BGR 이미지 (numpy array)
        instruction: 자연어 명령 ("파란 약병을 집어")
    
    Returns:
        (x, y) pixel 좌표 또는 None
    """
    # BGR → RGB → PIL
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    prompt = f"""You are a robot vision system. 
Given the image, find the best grasp point for the following instruction:
"{instruction}"

Return ONLY a JSON object with the pixel coordinates of the grasp point:
{{"x": <int>, "y": <int>}}

The grasp point should be at the CENTER TOP of the object where a parallel gripper can grip it.
Image resolution is {image_bgr.shape[1]}x{image_bgr.shape[0]}.
Do not include any other text."""

    response = model.generate_content([pil_image, prompt])
    
    try:
        # JSON 추출
        text = response.text.strip()
        # ```json ... ``` 블록 제거
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        result = json.loads(text)
        return (int(result["x"]), int(result["y"]))
    except Exception as e:
        print(f"[Gemini] Parsing error: {e}, raw: {response.text}")
        return None


def get_place_point(image_bgr, instruction):
    """place 목표 위치의 pixel 좌표 반환"""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    prompt = f"""You are a robot vision system.
Given the image, find the target PLACE location for:
"{instruction}"

Return ONLY a JSON object with the pixel coordinates:
{{"x": <int>, "y": <int>}}

The point should be where the object should be placed down.
Image resolution is {image_bgr.shape[1]}x{image_bgr.shape[0]}."""

    response = model.generate_content([pil_image, prompt])
    
    try:
        text = response.text.strip()
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        result = json.loads(text)
        return (int(result["x"]), int(result["y"]))
    except Exception as e:
        print(f"[Gemini] Parsing error: {e}")
        return None


def decompose_task(image_bgr, instruction):
    """복합 명령을 단계별로 분해"""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    prompt = f"""You are a robot task planner. A 6-DOF robot arm with a parallel gripper is in the scene.

Instruction: "{instruction}"

Break this down into sequential pick-and-place steps. Each step should be:
- "pick <object description>" 
- "place <location description>"

Return ONLY a JSON array:
[
  {{"action": "pick", "target": "description"}},
  {{"action": "place", "target": "description"}},
  ...
]"""

    response = model.generate_content([pil_image, prompt])
    
    try:
        text = response.text.strip()
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        return json.loads(text)
    except Exception as e:
        print(f"[Gemini] Task decomposition error: {e}")
        return None


def check_progress(image_bgr, step_description):
    """현재 step이 완료되었는지 판단"""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    prompt = f"""You are monitoring a robot performing a task.
The current step is: "{step_description}"

Look at the image and determine:
1. Is this step COMPLETE or NOT COMPLETE?
2. Brief reason.

Return ONLY JSON:
{{"complete": true/false, "reason": "brief explanation"}}"""

    response = model.generate_content([pil_image, prompt])
    
    try:
        text = response.text.strip()
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        return json.loads(text)
    except Exception as e:
        print(f"[Gemini] Progress check error: {e}")
        return {"complete": False, "reason": "parsing error"}
```

---

### Phase 3: IK Solver (30분~1시간)

#### 3.5 cuRobo IK (권장)

```python
# ik_solver.py
import numpy as np

# ==== 방법 A: cuRobo (권장, collision-aware) ====
try:
    from curobo.types.robot import RobotConfig
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    
    CUROBO_AVAILABLE = True
    
    def setup_curobo(urdf_path):
        """cuRobo IK solver 초기화"""
        robot_cfg = RobotConfig.from_basic(
            urdf_path=urdf_path,
            base_link="base_link",
            ee_link="gripper_rh_p12_rn_base",  # URDF에서 확인 필요
            tensor_args={"device": "cpu"},  # GPU 없어도 됨
        )
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            num_seeds=20,
        )
        return IKSolver(ik_config)
    
    def solve_ik_curobo(solver, target_xyz, target_rot=None):
        """3D target → joint angles"""
        from curobo.types.math import Pose
        
        if target_rot is None:
            # 기본: 아래를 향하는 자세 (pick용)
            target_rot = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0]  # rotation matrix flat
        
        pose = Pose(
            position=target_xyz / 1000.0,  # mm → m
            quaternion=None,  # rotation matrix 사용 시
        )
        result = solver.solve_single(pose)
        
        if result.success:
            return np.rad2deg(result.solution.cpu().numpy().flatten())  # degree
        else:
            print("[IK] No solution found!")
            return None

except ImportError:
    CUROBO_AVAILABLE = False
    print("[IK] cuRobo not available, using ikpy fallback")


# ==== 방법 B: ikpy 대체 (확실히 동작) ====
try:
    import ikpy.chain
    
    IKPY_AVAILABLE = True
    
    def setup_ikpy(urdf_path):
        """ikpy IK solver 초기화"""
        chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path,
            active_links_mask=[False, True, True, True, True, True, True, False]
            # base_link=False, joint_1~6=True, gripper=False
        )
        return chain
    
    def solve_ik_ikpy(chain, target_xyz):
        """3D target → joint angles (degree)"""
        target_m = np.array(target_xyz) / 1000.0  # mm → m
        
        ik_solution = chain.inverse_kinematics(
            target_position=target_m,
            orientation_mode="X",  # gripper pointing down
        )
        
        # active joints만 추출 (joint_1~6)
        joint_angles_rad = ik_solution[1:7]
        joint_angles_deg = np.rad2deg(joint_angles_rad)
        
        return joint_angles_deg

except ImportError:
    IKPY_AVAILABLE = False
    print("[IK] ikpy not available either!")


# ==== 통합 인터페이스 ====
class IKSolverWrapper:
    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        self.solver = None
        self.backend = None
        
        if CUROBO_AVAILABLE:
            self.solver = setup_curobo(urdf_path)
            self.backend = "curobo"
            print("[IK] Using cuRobo (collision-aware)")
        elif IKPY_AVAILABLE:
            self.solver = setup_ikpy(urdf_path)
            self.backend = "ikpy"
            print("[IK] Using ikpy (no collision check)")
        else:
            raise RuntimeError("No IK solver available! Install curobo or ikpy.")
    
    def solve(self, target_xyz_mm):
        """
        Args:
            target_xyz_mm: [x, y, z] in robot base frame (mm)
        Returns:
            joint_angles_deg: [j1, j2, j3, j4, j5, j6] in degrees
            또는 None (IK 실패)
        """
        if self.backend == "curobo":
            return solve_ik_curobo(self.solver, np.array(target_xyz_mm))
        elif self.backend == "ikpy":
            return solve_ik_ikpy(self.solver, target_xyz_mm)
        return None
```

---

### Phase 4: 통합 파이프라인 (1시간)

#### 3.6 메인 실행 코드

```python
# track2_main.py
"""
Track 2: Gemini Robotics-ER 1.5 + cuRobo + RGBD
Zero-shot Pick & Place Pipeline

Usage:
  python track2_main.py --instruction "파란 약병을 조제함에 넣어"
  python track2_main.py --instruction "약병들을 색깔별로 분류해"  # multi-step
"""

import argparse
import time
import cv2
import numpy as np
import json

from calibration import pixel_depth_to_robot, CAMERA_INTRINSICS
from gemini_er_client import (
    get_pick_point, get_place_point, 
    decompose_task, check_progress
)
from ik_solver import IKSolverWrapper

# ===== Configuration =====
ROBOT_IP = "192.168.137.100"
URDF_PATH = "~/doosan_ws/src/e0509_gripper_description/urdf/e0509_with_gripper.urdf.xacro"
CALIBRATION_PATH = "T_cam2base.npy"

# Safety
APPROACH_HEIGHT_MM = 100    # 물체 위 10cm에서 접근
RETREAT_HEIGHT_MM = 150     # pick 후 15cm 들어올리기
GRIPPER_CLOSE_WAIT = 1.5   # 그리퍼 닫힘 대기 (초)
MAX_RETRIES = 3

# ===== Robot Interface =====
class DoosanInterface:
    """ROS2 서비스를 통한 두산 로봇 제어"""
    
    def __init__(self):
        import rclpy
        from dsr_msgs2.srv import MoveJoint
        from std_srvs.srv import Trigger
        from std_msgs.msg import Int32
        from sensor_msgs.msg import JointState
        
        try:
            rclpy.init()
        except RuntimeError:
            pass
        
        self.node = rclpy.create_node("track2_controller")
        
        # 서비스 클라이언트
        self.move_joint_cli = self.node.create_client(
            MoveJoint, "/dsr01/motion/move_joint")
        self.gripper_open_cli = self.node.create_client(
            Trigger, "/dsr01/gripper/open")
        self.gripper_close_cli = self.node.create_client(
            Trigger, "/dsr01/gripper/close")
        
        # 그리퍼 stroke 토픽
        self.gripper_stroke = 0
        self.node.create_subscription(
            Int32, "/dsr01/gripper/stroke", 
            self._stroke_cb, 10)
        
        # Joint state
        self.joint_positions = np.zeros(6)
        self.node.create_subscription(
            JointState, "/joint_states",
            self._joint_cb, 10)
        
        print("[Robot] Interface initialized")
    
    def _stroke_cb(self, msg):
        self.gripper_stroke = msg.data
    
    def _joint_cb(self, msg):
        if len(msg.position) >= 6:
            self.joint_positions = np.array(msg.position[:6])
    
    def move_joint(self, joint_deg, vel=30.0, acc=30.0):
        """Joint 이동 (degree 단위)"""
        from dsr_msgs2.srv import MoveJoint
        import rclpy
        
        req = MoveJoint.Request()
        req.pos = list(joint_deg)
        req.vel = vel
        req.acc = acc
        
        future = self.move_joint_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=30.0)
        
        if future.result() is not None:
            print(f"[Robot] MoveJoint → [{', '.join(f'{d:.1f}' for d in joint_deg)}]")
            return True
        print("[Robot] MoveJoint FAILED")
        return False
    
    def gripper_open(self):
        from std_srvs.srv import Trigger
        import rclpy
        req = Trigger.Request()
        future = self.gripper_open_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)
        time.sleep(1.0)
        print("[Robot] Gripper OPEN")
    
    def gripper_close(self):
        from std_srvs.srv import Trigger
        import rclpy
        req = Trigger.Request()
        future = self.gripper_close_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)
        time.sleep(GRIPPER_CLOSE_WAIT)
        print(f"[Robot] Gripper CLOSE (stroke={self.gripper_stroke})")
    
    def is_object_grasped(self, threshold=50):
        """그리퍼 stroke 기반 grasp 판별 (0=열림, 700=닫힘)"""
        # 완전히 닫혔으면(700에 가까우면) 물체 없음
        # 중간에 멈추면(50~650) 물체 있음
        import rclpy
        rclpy.spin_once(self.node, timeout_sec=0.1)
        
        grasped = threshold < self.gripper_stroke < 650
        print(f"[Robot] Grasp check: stroke={self.gripper_stroke}, grasped={grasped}")
        return grasped
    
    def get_home_position(self):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# ===== Camera Interface =====
class RGBDCamera:
    """RGBD 카메라 인터페이스"""
    
    def __init__(self, camera_index=0):
        # 실제 RGBD 카메라에 맞게 수정 필요
        # RealSense, Azure Kinect 등
        self.cap = cv2.VideoCapture(camera_index)
        self.depth = None
        print(f"[Camera] Opened index {camera_index}")
    
    def read(self):
        """RGB + Depth 반환"""
        ret, color = self.cap.read()
        # TODO: 실제 depth 카메라에서 depth 읽기
        # self.depth = read_depth_from_sensor()
        return color, self.depth
    
    def get_depth_at(self, x, y):
        """특정 픽셀의 depth 값 (mm)"""
        if self.depth is not None:
            # 5x5 영역 median (noise 제거)
            region = self.depth[max(0,y-2):y+3, max(0,x-2):x+3]
            valid = region[region > 0]
            if len(valid) > 0:
                return float(np.median(valid))
        return None
    
    def release(self):
        self.cap.release()


# ===== Main Pipeline =====
class Track2Pipeline:
    def __init__(self, args):
        self.camera = RGBDCamera(args.camera_index)
        self.robot = DoosanInterface()
        self.ik = IKSolverWrapper(URDF_PATH)
        self.T_cam2base = np.load(CALIBRATION_PATH)
        self.intrinsics = CAMERA_INTRINSICS
        
        print("\n" + "="*60)
        print("  Track 2: Gemini-ER + cuRobo + RGBD Pipeline")
        print("="*60 + "\n")
    
    def pixel_to_robot_3d(self, px, py, depth_mm):
        """픽셀 좌표 + depth → 로봇 좌표계 3D"""
        return pixel_depth_to_robot(
            px, py, depth_mm, self.intrinsics, self.T_cam2base
        )
    
    def move_to_3d(self, target_xyz_mm, approach=False):
        """3D 좌표로 이동 (approach=True면 위에서 접근)"""
        if approach:
            # 먼저 target 위로 이동
            above = target_xyz_mm.copy()
            above[2] += APPROACH_HEIGHT_MM
            joints_above = self.ik.solve(above)
            if joints_above is not None:
                self.robot.move_joint(joints_above, vel=20.0, acc=20.0)
                time.sleep(0.5)
        
        # 목표 위치로 이동
        joints = self.ik.solve(target_xyz_mm)
        if joints is not None:
            self.robot.move_joint(joints, vel=10.0, acc=10.0)
            time.sleep(0.5)
            return True
        return False
    
    def pick(self, instruction):
        """단일 pick 동작"""
        color, depth = self.camera.read()
        
        # 1. Gemini-ER로 물체 위치 찾기
        print(f"\n[Gemini] Finding: {instruction}")
        point = get_pick_point(color, instruction)
        if point is None:
            print("[Gemini] Object not found!")
            return False
        
        px, py = point
        print(f"[Gemini] Object at pixel ({px}, {py})")
        
        # 2. Depth 읽기
        depth_mm = self.camera.get_depth_at(px, py)
        if depth_mm is None:
            print("[Camera] No depth at target pixel!")
            return False
        print(f"[Camera] Depth = {depth_mm:.0f} mm")
        
        # 3. 2D+depth → 로봇 3D
        target = self.pixel_to_robot_3d(px, py, depth_mm)
        print(f"[Calib] Robot 3D = [{target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}] mm")
        
        # 4. Gripper 열기
        self.robot.gripper_open()
        
        # 5. 위에서 접근 → 내려가기
        if not self.move_to_3d(target, approach=True):
            print("[IK] Failed to reach target!")
            return False
        
        # 6. Gripper 닫기
        self.robot.gripper_close()
        
        # 7. Grasp 성공 확인
        if not self.robot.is_object_grasped():
            print("[Grasp] FAILED — object not detected in gripper")
            return False
        
        # 8. 들어올리기
        lift = target.copy()
        lift[2] += RETREAT_HEIGHT_MM
        self.move_to_3d(lift)
        
        print("[Pick] SUCCESS")
        return True
    
    def place(self, instruction):
        """단일 place 동작"""
        color, depth = self.camera.read()
        
        # 1. Gemini-ER로 place 위치 찾기
        print(f"\n[Gemini] Finding place location: {instruction}")
        point = get_place_point(color, instruction)
        if point is None:
            print("[Gemini] Place location not found!")
            return False
        
        px, py = point
        depth_mm = self.camera.get_depth_at(px, py)
        if depth_mm is None:
            return False
        
        target = self.pixel_to_robot_3d(px, py, depth_mm)
        # Place는 표면보다 약간 위에서 놓기
        target[2] += 30  # 3cm 위에서 release
        
        # 2. 위에서 접근
        if not self.move_to_3d(target, approach=True):
            return False
        
        # 3. Gripper 열기
        self.robot.gripper_open()
        time.sleep(0.5)
        
        # 4. 후퇴
        retreat = target.copy()
        retreat[2] += RETREAT_HEIGHT_MM
        self.move_to_3d(retreat)
        
        print("[Place] SUCCESS")
        return True
    
    def pick_and_place_with_retry(self, pick_instruction, place_instruction):
        """pick & place + retry"""
        for attempt in range(MAX_RETRIES):
            print(f"\n{'='*40}")
            print(f"  Attempt {attempt+1}/{MAX_RETRIES}")
            print(f"{'='*40}")
            
            # Pick
            if not self.pick(pick_instruction):
                print(f"[Retry] Pick failed, retrying...")
                self.robot.gripper_open()
                self.robot.move_joint(self.robot.get_home_position())
                time.sleep(1.0)
                continue
            
            # Place
            if self.place(place_instruction):
                # Gemini-ER로 최종 확인
                color, _ = self.camera.read()
                progress = check_progress(color, place_instruction)
                if progress.get("complete", False):
                    print(f"\n✅ Task complete! Reason: {progress.get('reason', '')}")
                    return True
                else:
                    print(f"\n⚠️ Gemini says not complete: {progress.get('reason', '')}")
            
            # 실패 → home으로 복귀 후 retry
            self.robot.gripper_open()
            self.robot.move_joint(self.robot.get_home_position())
            time.sleep(1.0)
        
        print(f"\n❌ Failed after {MAX_RETRIES} attempts")
        return False
    
    def run_single_task(self, instruction):
        """단일 pick & place 명령 실행"""
        print(f"\n📋 Instruction: {instruction}\n")
        
        # 단순 pick & place로 처리
        pick_instr = f"pick: {instruction}"
        place_instr = f"place: {instruction}"
        
        return self.pick_and_place_with_retry(pick_instr, place_instr)
    
    def run_multi_step(self, instruction):
        """복합 명령 → Gemini-ER가 분해 → 순차 실행"""
        print(f"\n📋 Complex instruction: {instruction}\n")
        
        color, _ = self.camera.read()
        steps = decompose_task(color, instruction)
        
        if steps is None:
            print("[Gemini] Failed to decompose task")
            return False
        
        print(f"[Gemini] Decomposed into {len(steps)} steps:")
        for i, step in enumerate(steps):
            print(f"  Step {i+1}: {step['action']} → {step['target']}")
        
        # 순차 실행
        i = 0
        while i < len(steps):
            step = steps[i]
            print(f"\n{'='*50}")
            print(f"  Executing Step {i+1}/{len(steps)}: {step['action']} {step['target']}")
            print(f"{'='*50}")
            
            if step["action"] == "pick":
                success = self.pick(step["target"])
            elif step["action"] == "place":
                success = self.place(step["target"])
            else:
                print(f"[Unknown action: {step['action']}]")
                success = False
            
            if success:
                i += 1
            else:
                print(f"[Retry] Step {i+1} failed")
                self.robot.gripper_open()
                self.robot.move_joint(self.robot.get_home_position())
                time.sleep(1.0)
                # 같은 step retry (최대 3회는 pick_and_place_with_retry 내부에서)
                # 여기서는 단순히 다시 시도
        
        print(f"\n✅ All {len(steps)} steps completed!")
        return True
    
    def cleanup(self):
        self.camera.release()
        self.robot.move_joint(self.robot.get_home_position())


# ===== Entry Point =====
def main():
    parser = argparse.ArgumentParser(description="Track 2: Gemini-ER + cuRobo Pipeline")
    parser.add_argument("--instruction", required=True, help="자연어 명령")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--mode", choices=["single", "multi"], default="single",
                        help="single: 단일 pick&place, multi: 복합 명령 자동 분해")
    args = parser.parse_args()
    
    pipeline = Track2Pipeline(args)
    
    try:
        if args.mode == "multi":
            pipeline.run_multi_step(args.instruction)
        else:
            pipeline.run_single_task(args.instruction)
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
```

---

## 4. 현장 실행 체크리스트

### Phase 0: 도착 직후 (0~30분)

```
□ Wi-Fi 연결 확인 (Gemini API 호출에 필수)
□ Gemini API key 동작 확인:
    python -c "import google.generativeai as genai; genai.configure(api_key='KEY'); print('OK')"
□ 로봇 IP 확인 + ping 테스트
□ 카메라 연결 + RGB/Depth 스트림 확인
□ e0509_gripper_description 빌드 + bringup 테스트
```

### Phase 1: 캘리브레이션 (30분~1시간)

```
□ 카메라 고정 (절대 움직이지 않기!)
□ Camera intrinsics 확인/설정
□ 4점 캘리브레이션 실행 → T_cam2base.npy 저장
□ 검증: 각 점 오차 < 10mm
□ 추가 테스트: 5번째 점으로 검증 (캘리브레이션에 사용 안 한 점)
```

### Phase 2: 개별 모듈 테스트 (30분)

```
□ Gemini-ER pointing 테스트:
    이미지 저장 → get_pick_point() → 결과 확인 (이미지에 점 그려보기)
□ IK solver 테스트:
    알려진 3D 좌표 → IK → robot move → 실제 위치 확인
□ 그리퍼 테스트:
    open → close → stroke 값 확인 → is_object_grasped() 테스트
□ 2D→3D 변환 테스트:
    이미지에서 물체 클릭 → 3D 좌표 → IK → 로봇이 해당 위치로 이동
```

### Phase 3: 통합 테스트 (30분)

```
□ 단일 pick 테스트: 물체 하나 집기
□ 단일 place 테스트: 물체 하나 내려놓기
□ pick & place 전체 사이클 테스트
□ retry 로직 테스트: 일부러 실패 유도 → 재시도 동작 확인
□ multi-step 테스트: "약병 2개를 분류해" 같은 복합 명령
```

### Phase 4: 튜닝 (필요 시)

```
□ Gemini pointing이 부정확하면: prompt 수정 (물체 묘사 더 구체적으로)
□ IK가 실패하면: approach height 조절, target orientation 수정
□ Grasp 실패가 많으면: GRIPPER_CLOSE_WAIT 늘리기, threshold 조절
□ 속도 최적화: robot vel/acc 올리기 (안전 범위 내)
```

---

## 5. Dual-Brain 모드 (VLA + Gemini-ER 통합)

VLA (Track 1)가 동작하면, Gemini-ER를 **task planner + progress checker**로 결합:

```python
# dual_brain.py (VLA가 동작할 때만)
"""
Gemini-ER: 무엇을 할지 결정 (task decomposition)
VLA:       어떻게 할지 실행 (end-to-end manipulation)
Gemini-ER: 잘 됐는지 확인 (progress estimation)
"""

def dual_brain_pipeline(instruction, vla_controller, camera):
    # 1. Gemini-ER로 task 분해
    color, _ = camera.read()
    steps = decompose_task(color, instruction)
    
    for step in steps:
        # 2. 각 step을 VLA에게 자연어로 전달
        step_instruction = f"{step['action']} {step['target']}"
        print(f"[Gemini→VLA] Executing: {step_instruction}")
        
        vla_controller.run_episode(step_instruction)  # VLA가 실행
        
        # 3. Gemini-ER로 진행 확인
        color, _ = camera.read()
        progress = check_progress(color, step_instruction)
        
        if not progress["complete"]:
            print(f"[Gemini] Not complete: {progress['reason']}")
            print("[Gemini] Retrying with VLA...")
            vla_controller.run_episode(step_instruction)  # retry
```

---

## 6. 발표 전략

### 슬라이드 구성 제안

```
1. 문제 정의: 고령 사회 + 돌봄 인력 부족
2. 해결 접근: "Gemini가 생각하고, 로봇이 행동하는 Dual-Brain"
3. 아키텍처: Gemini-ER (reasoning) + cuRobo (motion) + RGBD (perception)
4. 시연 영상: 약병 pick & place
5. (선택) VLA vs Classical 비교: 같은 태스크를 두 가지 방법으로
6. 한계 & 향후 방향
```

### 시연 시나리오

```
시나리오: "약국 조제 보조"
1. 사람: "빨간 약병을 1번 조제함에 넣어줘"
   → Gemini-ER가 물체 인식 → cuRobo가 pick → place → 완료 확인
2. 사람: "나머지 약병도 색깔별로 분류해"
   → Gemini-ER가 multi-step 분해 → 순차 실행 → 전부 완료
```

---

## 7. 트러블슈팅

| 문제 | 원인 | 해결 |
|------|------|------|
| Gemini API 호출 실패 | 네트워크/키 문제 | Wi-Fi 확인, 키 재발급 |
| Gemini pointing 부정확 | Prompt가 모호 | 물체 색/크기/위치 구체적 명시 |
| Depth 값이 0/None | 반사/투명 물체 | 물체 위치 약간 이동, 5×5 median 필터 |
| IK solution 없음 | Reach 범위 밖 | 물체를 로봇에 가깝게 배치, approach height 줄이기 |
| 캘리브레이션 오차 > 2cm | 4점 불충분 | 6~8점으로 늘리기, depth 이상치 확인 |
| Grasp 실패 (빈 그리퍼) | 위치 오차 | 캘리브레이션 재확인, approach 속도 낮추기 |
| cuRobo 설치 실패 | CUDA 의존성 | ikpy로 fallback |
| 로봇 MoveJoint 무응답 | 서비스 미실행 | bringup 재실행, 로봇 IP 확인 |

---

## 8. 파일 구조

```
track2/
├── README.md                    ← 이 파일
├── calibration.py              ← 4점 extrinsic calibration
├── gemini_er_client.py         ← Gemini-ER API 래퍼
├── ik_solver.py                ← cuRobo / ikpy IK solver
├── track2_main.py              ← 메인 파이프라인
├── dual_brain.py               ← VLA + Gemini-ER 통합 (선택)
├── camera_test.py              ← 카메라 스트림 확인
├── T_cam2base.npy              ← 캘리브레이션 결과 (현장 생성)
└── requirements.txt            ← 의존성
```

```
# requirements.txt
google-generativeai>=0.8.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
scipy>=1.10.0
Pillow>=9.0.0
ikpy>=3.3.0
# curobo  # 별도 설치 필요
# pyrealsense2  # Intel RealSense 사용 시
```
