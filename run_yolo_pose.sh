#!/bin/bash
CAMERA_DEVICE=/dev/video0

docker run \
    -it \
    --rm \
    --net=host \
    --device "$CAMERA_DEVICE" \
    --device /dev/hailo0 \
    -e ROS_DOMAIN_ID=0 \
    -v "$PWD/ws:/ws" \
    ghcr.io/horilaboratory/robot_project_ai/ros2-hailort:latest \
    python3 yolov8m_pose_publisher.py \
    --ros-args \
    -p "device:=$CAMERA_DEVICE" \
    -p hef_path:=/ws/yolov8m_pose.hef
