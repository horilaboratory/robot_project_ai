#!/usr/bin/env python3
"""USBカメラ画像を HailoRT (YOLO, NMS_BY_CLASS 出力の HEF) で物体認識し、結果を Publish するノード。

Publish するトピック (名前は ~ 相対):
    image_annotated             sensor_msgs/Image           バウンディングボックス付き画像 (bgr8)
    image_annotated/compressed  sensor_msgs/CompressedImage 同上 (jpeg)
    detections                  vision_msgs/Detection2DArray 検出結果 (中心座標・サイズは元画像のピクセル)

使い方:
    python3 yolo_hailo_publisher.py --ros-args -p hef_path:=/ws/yolov8m.hef -p device:=/dev/video8
"""
import time

import cv2
import numpy as np
import rclpy
from hailo_platform import HailoSchedulingAlgorithm, VDevice
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


class YoloHailoPublisher(Node):
    def __init__(self):
        super().__init__("yolo_hailo_publisher")
        self.declare_parameter("hef_path", "yolov8m.hef")
        self.declare_parameter("device", "/dev/video0")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("cam_fps", 15.0)
        self.declare_parameter("score_threshold", 0.5)
        self.declare_parameter("frame_id", "usb_cam")
        self.declare_parameter("publish_raw", True)
        self.declare_parameter("publish_compressed", True)
        self.declare_parameter("jpeg_quality", 80)
        self.declare_parameter("timeout_ms", 1000)

        self.frame_id = self.get_parameter("frame_id").value
        self.threshold = float(self.get_parameter("score_threshold").value)
        self.publish_raw = bool(self.get_parameter("publish_raw").value)
        self.publish_compressed = bool(self.get_parameter("publish_compressed").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.timeout_ms = int(self.get_parameter("timeout_ms").value)

        device = self.get_parameter("device").value
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"カメラを開けません: {device}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.get_parameter("width").value))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.get_parameter("height").value))
        self.cap.set(cv2.CAP_PROP_FPS, float(self.get_parameter("cam_fps").value))

        # Hailo の初期化 (VDevice は終了時に release する必要がある)
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.vdevice = VDevice(params)
        infer_model = self.vdevice.create_infer_model(self.get_parameter("hef_path").value)
        self.infer_model = infer_model
        self.configured_ctx = infer_model.configure()
        self.configured = self.configured_ctx.__enter__()
        self.bindings = self.configured.create_bindings()
        self.out_buffer = np.empty(infer_model.output().shape, dtype=np.float32)
        self.bindings.output().set_buffer(self.out_buffer)
        # 入力形状は (H, W, C)
        self.in_h, self.in_w = infer_model.input().shape[:2]

        # 別PCの購読 (sensor_data QoS = BEST_EFFORT) と互換
        self.pub_img = self.create_publisher(Image, "image_annotated", qos_profile_sensor_data)
        self.pub_comp = self.create_publisher(
            CompressedImage, "image_annotated/compressed", qos_profile_sensor_data
        )
        self.pub_det = self.create_publisher(Detection2DArray, "detections", 10)

        self.get_logger().info(
            f"HEF 読み込み完了 (入力 {self.in_w}x{self.in_h}) / カメラ {device}"
        )
        self.timer = self.create_timer(0.0, self.on_timer)

    def infer(self, frame_bgr):
        """推論して [(x1, y1, x2, y2, score, class_id), ...] (元画像のピクセル座標) を返す。"""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (self.in_w, self.in_h)), cv2.COLOR_BGR2RGB)
        self.bindings.input().set_buffer(np.expand_dims(rgb, axis=0).astype(np.uint8))
        self.configured.run([self.bindings], self.timeout_ms)
        result = self.bindings.output().get_buffer()

        dets = []
        # NMS_BY_CLASS: クラスごとの配列 [y_min, x_min, y_max, x_max, score] (0-1 に正規化)
        for cls, arr in enumerate(result):
            arr = np.asarray(arr)
            if arr.size == 0:
                continue
            for y1, x1, y2, x2, score in arr.reshape(-1, 5):
                if score < self.threshold:
                    continue
                dets.append((
                    float(np.clip(x1, 0, 1) * w), float(np.clip(y1, 0, 1) * h),
                    float(np.clip(x2, 0, 1) * w), float(np.clip(y2, 0, 1) * h),
                    float(score), cls,
                ))
        return dets

    @staticmethod
    def draw(frame, dets, fps):
        for x1, y1, x2, y2, score, cls in dets:
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
            name = COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else str(cls)
            cv2.putText(frame, f"{name}: {score:.2f}", (p1[0] + 3, max(p1[1] + 16, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    def on_timer(self):
        start = time.time()
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warning("フレームの取得に失敗しました")
            return
        stamp = self.get_clock().now().to_msg()
        dets = self.infer(frame)
        fps = 1.0 / max(time.time() - start, 1e-6)

        arr = Detection2DArray()
        arr.header.stamp = stamp
        arr.header.frame_id = self.frame_id
        for x1, y1, x2, y2, score, cls in dets:
            d = Detection2D()
            d.header = arr.header
            d.bbox.center.position.x = (x1 + x2) / 2.0
            d.bbox.center.position.y = (y1 + y2) / 2.0
            d.bbox.size_x = x2 - x1
            d.bbox.size_y = y2 - y1
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else str(cls)
            hyp.hypothesis.score = score
            d.results.append(hyp)
            arr.detections.append(d)
        self.pub_det.publish(arr)

        self.draw(frame, dets, fps)
        if self.publish_raw:
            msg = Image()
            msg.header.stamp = stamp
            msg.header.frame_id = self.frame_id
            msg.height, msg.width = frame.shape[:2]
            msg.encoding = "bgr8"
            msg.is_bigendian = 0
            msg.step = msg.width * 3
            msg.data = frame.tobytes()
            self.pub_img.publish(msg)
        if self.publish_compressed:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                cmsg = CompressedImage()
                cmsg.header.stamp = stamp
                cmsg.header.frame_id = self.frame_id
                cmsg.format = "jpeg"
                cmsg.data = buf.tobytes()
                self.pub_comp.publish(cmsg)

    def destroy_node(self):
        self.cap.release()
        self.configured_ctx.__exit__(None, None, None)
        self.vdevice.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = YoloHailoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
