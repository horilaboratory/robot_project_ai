#!/bin/bash
CAMERA_DEVICE=/dev/video8

docker run \
    -it \
    --rm \
    --net=host \
    --device "$CAMERA_DEVICE" \
    --device /dev/hailo0 \
    -e ROS_DOMAIN_ID=0 \
    -v "$PWD/ws:/ws" \
    hailort-ros2-humble:latest \
    python3 yolov8m_pose_publisher.py \
    --ros-args \
    -p "device:=$CAMERA_DEVICE" \
    -p hef_path:=/ws/yolov8m_pose.hef
