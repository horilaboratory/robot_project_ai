#!/usr/bin/env python3
"""USB webカメラの画像を sensor_msgs/Image (bgr8) として Publish するノード。

cv_bridge に依存せず、OpenCV + numpy だけで動作する。
使い方:
    python3 usb_cam_publisher.py --ros-args -p device:=/dev/video8 -p fps:=15.0
"""
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image


class UsbCamPublisher(Node):
    def __init__(self):
        super().__init__("usb_cam_publisher")
        self.declare_parameter("device", "/dev/video8")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 15.0)
        self.declare_parameter("frame_id", "usb_cam")
        self.declare_parameter("topic", "image_raw")
        self.declare_parameter("publish_raw", True)
        self.declare_parameter("publish_compressed", True)
        self.declare_parameter("jpeg_quality", 80)

        device = self.get_parameter("device").value
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        fps = float(self.get_parameter("fps").value)
        self.frame_id = self.get_parameter("frame_id").value

        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"カメラを開けません: {device}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # 別PCの image_raw 購読(sensor_data QoS = BEST_EFFORT)と互換
        self.pub = self.create_publisher(
            Image, self.get_parameter("topic").value, qos_profile_sensor_data
        )
        self.publish_raw = bool(self.get_parameter("publish_raw").value)
        self.publish_compressed = bool(self.get_parameter("publish_compressed").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        # Wi-Fi 越しでは圧縮画像 (<topic>/compressed) の利用を推奨
        self.pub_c = self.create_publisher(
            CompressedImage,
            self.get_parameter("topic").value + "/compressed",
            qos_profile_sensor_data,
        )
        self.timer = self.create_timer(1.0 / fps, self.on_timer)
        self.get_logger().info(
            f"{device} を開きました ({width}x{height} @ {fps}fps) -> "
            f"{self.pub.topic_name}"
        )

    def on_timer(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warning("フレームの取得に失敗しました")
            return
        stamp = self.get_clock().now().to_msg()
        if self.publish_raw:
            msg = Image()
            msg.header.stamp = stamp
            msg.header.frame_id = self.frame_id
            msg.height, msg.width = frame.shape[:2]
            msg.encoding = "bgr8"
            msg.is_bigendian = 0
            msg.step = msg.width * 3
            msg.data = frame.tobytes()
            self.pub.publish(msg)
        if self.publish_compressed:
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
            if ok:
                cmsg = CompressedImage()
                cmsg.header.stamp = stamp
                cmsg.header.frame_id = self.frame_id
                cmsg.format = "jpeg"
                cmsg.data = buf.tobytes()
                self.pub_c.publish(cmsg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = UsbCamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
