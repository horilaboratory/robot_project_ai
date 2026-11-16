#!/usr/bin/env python3
"""USB カメラ画像から YOLOv8m で姿勢を推定し、結果と描画画像を ROS 2 に配信する。"""

import time
from contextlib import ExitStack

import cv2
import numpy as np
import rclpy
from hailo_platform import FormatType, HailoSchedulingAlgorithm, VDevice
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


# COCO の 17 関節の順序。SKELETON の数字はこの配列のインデックス。
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
)


def letterbox(frame, width, height):
    """縦横比を保って画像をリサイズし、余白と座標復元用の変換情報を返す。"""
    src_h, src_w = frame.shape[:2]
    scale = min(width / src_w, height / src_h)
    resized_w = max(1, min(width, round(src_w * scale)))
    resized_h = max(1, min(height, round(src_h * scale)))
    left = (width - resized_w) // 2
    top = (height - resized_h) // 2
    resized = cv2.resize(frame, (resized_w, resized_h))
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    canvas[top:top + resized_h, left:left + resized_w] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), (
        resized_w / src_w, resized_h / src_h, left, top
    )


def output_layout(infer_model):
    """ストリーム名や順序に依存せず、9 個の出力を形状で識別する。"""
    # 各解像度のチャンネル数は人物スコア 1、関節 51、矩形 64。
    expected = {(size, size, channels) for size in (20, 40, 80)
                for channels in (1, 51, 64)}
    layout = {}
    for output in infer_model.outputs:
        shape = tuple(output.shape)
        if shape not in expected or shape in layout:
            raise ValueError(f"非対応または重複した HEF 出力: {output.name} {shape}")
        layout[shape] = output.name
    if set(layout) != expected:
        raise ValueError(f"YOLOv8m pose HEF の出力が不足しています: {expected - set(layout)}")
    return layout


def decode_pose(outputs, layout, transform, image_shape, score_threshold,
                iou_threshold, max_detections):
    """モデルの生出力を人物矩形と関節に変換し、重複検出を除く。"""
    boxes, scores, joints = [], [], []
    for size in (20, 40, 80):
        score_map = outputs[layout[(size, size, 1)]].reshape(size, size)
        rows, cols = np.nonzero(score_map > score_threshold)
        if not len(rows):
            continue
        stride = 640 / size
        # 矩形の各辺に対応する 16 分割の DFL 分布を距離の期待値に変換する。
        logits = outputs[layout[(size, size, 64)]][rows, cols].reshape(-1, 4, 16)
        weights = np.exp(logits - logits.max(axis=2, keepdims=True))
        distances = (weights @ np.arange(16, dtype=np.float32)) / weights.sum(axis=2)
        distances *= stride
        centers = (np.stack((cols, rows), axis=1) + 0.5) * stride
        box = np.concatenate((centers - distances[:, :2], centers + distances[:, 2:]), axis=1)
        # 関節の x, y はグリッド座標から入力画像の座標へ、信頼度は sigmoid で変換する。
        keypoints = outputs[layout[(size, size, 51)]][rows, cols].reshape(-1, 17, 3).copy()
        keypoints[:, :, :2] = (
            2 * keypoints[:, :, :2] + np.stack((cols, rows), axis=1)[:, None, :]
        ) * stride
        keypoints[:, :, 2] = 1.0 / (1.0 + np.exp(-np.clip(keypoints[:, :, 2], -80, 80)))
        boxes.append(box)
        scores.append(score_map[rows, cols])
        joints.append(keypoints)

    if not boxes:
        return []
    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    joints = np.concatenate(joints)
    order = np.argsort(-scores, kind="stable")
    selected = []
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    # スコア順に選び、IoU が閾値を超える矩形を除外する (NMS)。
    while len(order) and len(selected) < max_detections:
        index = order[0]
        selected.append(index)
        rest = order[1:]
        intersection = np.maximum(
            np.minimum(boxes[index, 2:], boxes[rest, 2:])
            - np.maximum(boxes[index, :2], boxes[rest, :2]), 0
        ).prod(axis=1)
        union = areas[index] + areas[rest] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = rest[iou <= iou_threshold]

    # letterbox の余白と拡大率を戻し、元画像のピクセル座標に変換する。
    scale_x, scale_y, left, top = transform
    height, width = image_shape[:2]
    results = []
    for index in selected:
        box = boxes[index].copy()
        box[[0, 2]] = np.clip((box[[0, 2]] - left) / scale_x, 0, width)
        box[[1, 3]] = np.clip((box[[1, 3]] - top) / scale_y, 0, height)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        keypoints = joints[index].copy()
        keypoints[:, 0] = (keypoints[:, 0] - left) / scale_x
        keypoints[:, 1] = (keypoints[:, 1] - top) / scale_y
        results.append((box, float(scores[index]), keypoints))
    return results


def make_detection(header, person_id, class_id, score, x, y, width=0.0, height=0.0):
    """人物矩形またはサイズ 0 の関節点を Detection2D に格納する。"""
    detection = Detection2D()
    detection.header = header
    detection.id = person_id
    detection.bbox.center.position.x = float(x)
    detection.bbox.center.position.y = float(y)
    detection.bbox.size_x = float(width)
    detection.bbox.size_y = float(height)
    hypothesis = ObjectHypothesisWithPose()
    hypothesis.hypothesis.class_id = class_id
    hypothesis.hypothesis.score = float(score)
    detection.results.append(hypothesis)
    return detection


class YoloPosePublisher(Node):
    def __init__(self):
        super().__init__("yolov8m_pose_publisher")
        # 初期化途中の失敗時も、確保済みのカメラと Hailo リソースを解放する。
        self.resources = ExitStack()
        try:
            self.setup()
        except Exception:
            self.resources.close()
            super().destroy_node()
            raise

    def setup(self):
        defaults = {
            "hef_path": "/ws/yolov8m_pose.hef", "device": "/dev/video0",
            "width": 640, "height": 480, "cam_fps": 15.0,
            "score_threshold": 0.5, "keypoint_threshold": 0.5,
            "iou_threshold": 0.45, "max_detections": 100,
            "frame_id": "usb_cam", "publish_raw": True, "publish_compressed": True,
            "jpeg_quality": 80, "timeout_ms": 1000,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        param = lambda name: self.get_parameter(name).value
        self.frame_id = param("frame_id")
        self.score_threshold = float(param("score_threshold"))
        self.keypoint_threshold = float(param("keypoint_threshold"))
        self.iou_threshold = float(param("iou_threshold"))
        self.max_detections = int(param("max_detections"))
        self.publish_raw = bool(param("publish_raw"))
        self.publish_compressed = bool(param("publish_compressed"))
        self.jpeg_quality = int(param("jpeg_quality"))
        self.timeout_ms = int(param("timeout_ms"))
        width, height, fps = int(param("width")), int(param("height")), float(param("cam_fps"))
        if (not all(0 <= value <= 1 for value in
                    (self.score_threshold, self.keypoint_threshold, self.iou_threshold))
                or width <= 0 or height <= 0 or not 0 < fps < float("inf")
                or self.max_detections <= 0 or self.timeout_ms <= 0
                or not 1 <= self.jpeg_quality <= 100):
            raise ValueError("カメラまたは推論パラメータが不正です")

        device = param("device")
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.resources.callback(self.cap.release)
        if not self.cap.isOpened():
            raise RuntimeError(f"カメラを開けません: {device}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # HailoRT の各出力を float32 に変換して受け取り、CPU 側で後処理する。
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.vdevice = VDevice(params)
        self.resources.callback(self.vdevice.release)
        self.infer_model = self.vdevice.create_infer_model(param("hef_path"))
        if tuple(self.infer_model.input().shape) != (640, 640, 3):
            raise ValueError(f"非対応の HEF 入力形状: {self.infer_model.input().shape}")
        self.layout = output_layout(self.infer_model)
        for output in self.infer_model.outputs:
            output.set_format_type(FormatType.FLOAT32)
        self.configured = self.resources.enter_context(self.infer_model.configure())
        self.buffers = {name: np.empty(shape, dtype=np.float32)
                        for shape, name in self.layout.items()}
        self.bindings = self.configured.create_bindings(output_buffers=self.buffers)

        # 人物と関節は Detection2DArray、描画画像は生画像と JPEG で配信する。
        self.pub_people = self.create_publisher(Detection2DArray, "pose/detections", 10)
        self.pub_keypoints = self.create_publisher(Detection2DArray, "pose/keypoints", 10)
        self.pub_image = self.create_publisher(Image, "image_annotated", qos_profile_sensor_data)
        self.pub_compressed = self.create_publisher(
            CompressedImage, "image_annotated/compressed", qos_profile_sensor_data
        )
        self.timer = self.create_timer(0.0, self.on_timer)
        self.get_logger().info(f"姿勢推定 HEF を読み込みました / カメラ {device}")

    def on_timer(self):
        start = time.monotonic()
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warning("フレームの取得に失敗しました")
            return
        # 同じフレーム由来の人物・関節・画像には共通の時刻を付ける。
        stamp = self.get_clock().now().to_msg()
        rgb, transform = letterbox(frame, 640, 640)
        self.bindings.input().set_buffer(rgb)
        self.configured.run([self.bindings], self.timeout_ms)
        poses = decode_pose(self.buffers, self.layout, transform, frame.shape,
                            self.score_threshold, self.iou_threshold, self.max_detections)

        people = Detection2DArray()
        people.header.stamp = stamp
        people.header.frame_id = self.frame_id
        keypoints = Detection2DArray()
        keypoints.header = people.header
        for index, (box, score, joints) in enumerate(poses):
            x1, y1, x2, y2 = box
            # ID は同一フレーム内の人物と関節の対応用で、フレーム間の追跡 ID ではない。
            person_id = str(index)
            people.detections.append(make_detection(
                people.header, person_id, "person", score,
                (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            ))
            for name, (x, y, confidence) in zip(KEYPOINT_NAMES, joints):
                keypoints.detections.append(make_detection(
                    people.header, person_id, name, confidence, x, y
                ))
            self.draw_pose(frame, box, score, joints)
        self.pub_people.publish(people)
        self.pub_keypoints.publish(keypoints)

        fps = 1.0 / max(time.monotonic() - start, 1e-6)
        cv2.putText(frame, f"FPS: {fps:.1f}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        if self.publish_raw:
            msg = Image()
            msg.header = people.header
            msg.height, msg.width = frame.shape[:2]
            msg.encoding = "bgr8"
            msg.is_bigendian = 0
            msg.step = msg.width * 3
            msg.data = frame.tobytes()
            self.pub_image.publish(msg)
        if self.publish_compressed:
            ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                self.get_logger().error("JPEG 画像のエンコードに失敗しました")
            else:
                msg = CompressedImage()
                msg.header = people.header
                msg.format = "jpeg"
                msg.data = data.tobytes()
                self.pub_compressed.publish(msg)

    def draw_pose(self, frame, box, score, joints):
        x1, y1, x2, y2 = (int(value) for value in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"person: {score:.2f}", (x1 + 3, max(y1 + 16, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        height, width = frame.shape[:2]
        # 閾値以上かつ画像内の関節だけを描画する。配信結果には全関節を含める。
        visible = [(confidence >= self.keypoint_threshold and 0 <= x < width and 0 <= y < height)
                   for x, y, confidence in joints]
        for a, b in SKELETON:
            if visible[a] and visible[b]:
                cv2.line(frame, tuple(joints[a, :2].astype(int)),
                         tuple(joints[b, :2].astype(int)), (0, 255, 0), 2)
        for joint, show in zip(joints, visible):
            if show:
                cv2.circle(frame, tuple(joint[:2].astype(int)), 3, (0, 255, 255), -1)

    def destroy_node(self):
        try:
            self.resources.close()
        finally:
            super().destroy_node()


def main():
    rclpy.init()
    node = None
    try:
        node = YoloPosePublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()