#!/usr/bin/env python3
"""
리얼센스 카메라로 클릭한 위치의 3D 좌표를 표시
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

clicked_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

window_name = "RealSense - Click to get 3D coords"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(window_name, mouse_callback)

last_text = ""

print("카메라 시작! 물체를 클릭하세요. q=종료")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data()).copy()
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        # 클릭 처리
        while clicked_points:
            px, py = clicked_points.pop(0)
            dist = depth_frame.get_distance(px, py)
            if dist > 0:
                pt = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], dist)
                x_mm = pt[0] * 1000
                y_mm = pt[1] * 1000
                z_mm = pt[2] * 1000
                last_text = f"x={x_mm:.1f} y={y_mm:.1f} z={z_mm:.1f}mm"
                print(f"카메라 좌표: x={x_mm:.1f}mm, y={y_mm:.1f}mm, z={z_mm:.1f}mm (거리: {z_mm:.1f}mm)")
                cv2.circle(img, (px, py), 8, (0, 255, 0), 2)
            else:
                print(f"depth 없음 ({px}, {py})")

        if last_text:
            cv2.putText(img, last_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
