#!/usr/bin/env python3
"""현재 리얼센스 장면을 캡처하여 RGB + numpy 저장"""
import pyrealsense2 as rs
import numpy as np
import cv2
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

# 몇 프레임 버리기 (auto exposure 안정화)
for _ in range(30):
    pipeline.wait_for_frames()

frames = pipeline.wait_for_frames()
aligned = align.process(frames)
depth_frame = aligned.get_depth_frame()
color_frame = aligned.get_color_frame()

color_image = np.asanyarray(color_frame.get_data())
depth_image = np.asanyarray(depth_frame.get_data())

# 저장
cv2.imwrite(os.path.expanduser("~/scene_rgb.png"), color_image)
np.save(os.path.expanduser("~/scene_rgb.npy"), color_image)
np.save(os.path.expanduser("~/scene_depth.npy"), depth_image)

# intrinsics도 저장
intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
intr_data = {
    "fx": intrinsics.fx, "fy": intrinsics.fy,
    "ppx": intrinsics.ppx, "ppy": intrinsics.ppy,
    "width": intrinsics.width, "height": intrinsics.height,
    "coeffs": intrinsics.coeffs,
}
np.save(os.path.expanduser("~/scene_intrinsics.npy"), intr_data)

pipeline.stop()

print(f"저장 완료:")
print(f"  ~/scene_rgb.png (BGR {color_image.shape})")
print(f"  ~/scene_rgb.npy (BGR {color_image.shape})")
print(f"  ~/scene_depth.npy (uint16 {depth_image.shape})")
print(f"  ~/scene_intrinsics.npy")
