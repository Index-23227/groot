"""카메라 RGB + Depth 스트림 테스트"""

import cv2
import numpy as np
import sys


def test_opencv(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"❌ Camera index {index} failed")
        return
    print(f"✅ Camera index {index} opened")
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        print(f"   Resolution: {w}x{h}")
        cv2.imwrite("camera_test_rgb.jpg", frame)
        print("   Saved camera_test_rgb.jpg")
    cap.release()


def test_realsense():
    try:
        import pyrealsense2 as rs
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = pipe.start(cfg)

        # Intrinsics
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        print(f"✅ RealSense opened")
        print(f"   fx={intr.fx:.1f}, fy={intr.fy:.1f}, cx={intr.ppx:.1f}, cy={intr.ppy:.1f}")

        frames = pipe.wait_for_frames()
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data())

        cv2.imwrite("camera_test_rgb.jpg", color)
        np.save("camera_test_depth.npy", depth)
        print(f"   RGB shape: {color.shape}, Depth shape: {depth.shape}")
        print(f"   Depth range: {depth[depth>0].min()}~{depth.max()} mm")
        print("   Saved camera_test_rgb.jpg + camera_test_depth.npy")

        pipe.stop()
    except ImportError:
        print("❌ pyrealsense2 not installed")
    except Exception as e:
        print(f"❌ RealSense error: {e}")


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    test_opencv(idx)
    test_realsense()
