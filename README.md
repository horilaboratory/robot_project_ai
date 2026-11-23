README
========

本プロジェクトは、ROS 2 上で Raspberry Pi AI HAT+ を使用するためのものである。

## 実行環境
本プロジェクトを実行するためには，Raspberry Pi 5（AI HAT+搭載）と以下の環境が必要である．
- Raspberry Pi OS (64 bit) trixie
- Docker
- Hailoドライバ（Firmware Version: 4.23.0）

実行環境であるHailoRTとROS2はDockerコンテナ内で実行する．コンテナイメージは次のコマンドで取得できる。

```bash
docker pull ghcr.io/horilaboratory/robot_project_ai/ros2-hailort:latest
```

コンテナの環境は以下の通りである．
- OS: Ubuntu 22.04
- ROS 2: Humble
- HailoRT: 4.23.0 (Pythonバインディング含む)

USB camera test
---------------

[ws/usb_cam_publisher.py](ws/usb_cam_publisher.py) は USB Web カメラの画像を Publish する。

- `/image_raw` (`sensor_msgs/Image`, bgr8)
- `/image_raw/compressed` (`sensor_msgs/CompressedImage`, jpeg)

パラメータ: `device` (既定 `/dev/video8`), `width`, `height`, `fps`, `frame_id`, `topic`,
`publish_raw`, `publish_compressed`, `jpeg_quality`

Raspberry Pi 上 (コンテナ):

    docker run -it --rm --net=host --device /dev/video8 \
        -e ROS_DOMAIN_ID=0 -v $PWD/ws:/ws ghcr.io/horilaboratory/robot_project_ai/ros2-hailort:latest \
        python3 usb_cam_publisher.py

- カメラのデバイスは `v4l2-ctl --list-devices` で確認する。Raspberry Pi 5 では
  `/dev/video0〜7` は CSI、`20〜35` は ISP なので USB カメラではない。
  USB カメラ配下の 2 つのうち、Video Capture のほう (メタデータ用ではない) を指定する。
- `--net=host` が必須 (DDS 探索のため)。
- Wi-Fi では生画像 (約 0.9MB/フレーム) は欠落しやすいので、`-p publish_raw:=false` で止めて
  圧縮トピックを使うとよい。

YOLO 物体認識 (HailoRT)
-----------------------

[ws/yolo_hailo_publisher.py](ws/yolo_hailo_publisher.py) は USB カメラ画像を Hailo-8 で YOLO 推論し、結果を Publish する
(NMS 後処理込みの HEF、COCO 80 クラス想定。カメラは本ノードが直接開くので `usb_cam_publisher.py` とは同時に使用できない)。

- `/image_annotated` (`sensor_msgs/Image`, bgr8): バウンディングボックス付き画像
- `/image_annotated/compressed` (`sensor_msgs/CompressedImage`, jpeg)
- `/detections` (`vision_msgs/Detection2DArray`): 元画像のピクセル座標、`class_id` はクラス名

パラメータ: `hef_path`, `device` (既定 `/dev/video0`), `width`, `height`, `cam_fps`,
`score_threshold` (既定 0.5),
`frame_id`, `publish_raw`, `publish_compressed`, `jpeg_quality`, `timeout_ms`

    docker run -it --rm --net=host --device /dev/video8 --device /dev/hailo0 \
        -e ROS_DOMAIN_ID=0 -v $PWD/ws:/ws ghcr.io/horilaboratory/robot_project_ai/ros2-hailort:latest \
        python3 yolo_hailo_publisher.py --ros-args \
        -p hef_path:=/ws/yolov8m.hef -p device:=/dev/video8

カメラを変更する場合は、`--device` で Docker に渡すデバイスと、ROS パラメータ
`-p device:=/dev/videoN` の両方を指定する。Python ノード単体で実行する場合も
`python3 yolo_hailo_publisher.py --ros-args -p device:=/dev/videoN` と指定できる。

YOLOv8m 姿勢推定 (HailoRT)
--------------------------

[ws/yolov8m_pose_publisher.py](ws/yolov8m_pose_publisher.py) は、USB カメラ画像から
複数人の COCO 17 関節を推定する。カメラは本ノードが直接開くため、同じカメラを
`usb_cam_publisher.py` や物体認識ノードと同時に使用できない。

[Hailo Model Zoo の Hailo-8 用姿勢推定モデル一覧](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_pose_estimation.rst)
から **yolov8m_pose の RGB HEF** をダウンロードし、`ws/yolov8m_pose.hef` に置く。
Hailo-8 用 HEF と互換性のある Hailo デバイス・HailoRT が必要。
640x640 入力と 9 個の生出力を持つ Model Zoo の HEF を想定し、後処理は CPU で行う。

```sh
docker run -it --rm --net=host --device /dev/video8 --device /dev/hailo0 \
    -e ROS_DOMAIN_ID=0 -v "$PWD/ws:/ws" ghcr.io/horilaboratory/robot_project_ai/ros2-hailort:latest \
    python3 yolov8m_pose_publisher.py --ros-args \
    -p device:=/dev/video8 -p hef_path:=/ws/yolov8m_pose.hef
```

トピック名はノードの名前空間に対する相対名。以下はルート名前空間の場合。

- `/pose/detections` (`vision_msgs/Detection2DArray`): 人物ごとの矩形、`class_id=person`、
  人物検出の信頼度。
- `/pose/keypoints` (`vision_msgs/Detection2DArray`): 各人物の 17 関節。関節の
  `bbox.center.position` は元画像上のピクセル座標、サイズは 0、
  `results[0].hypothesis.class_id` は COCO 関節名、`score` は関節の信頼度。
  信頼度に関わらず全関節を出力し、画像外に予測された座標もそのまま出力する。
- `/image_annotated` (`sensor_msgs/Image`, bgr8): 人物矩形と骨格の描画画像。
- `/image_annotated/compressed` (`sensor_msgs/CompressedImage`, jpeg): 同じ描画画像の圧縮版。

ROS 2 Humble の標準メッセージには複数人の 2D 骨格を表す専用型がないため、
`vision_msgs/Detection2DArray` を関節点にも使用する。この表現は本ノード固有であり、
専用の姿勢推定メッセージを期待するノードには変換が必要。両配列の `Detection2D.id`
が一致する関節と人物が対応する。ID はフレーム内の番号であり、追跡 ID ではない。
全トピックのヘッダーに同じ撮影直後の時刻と `frame_id` が入るため、
異なるフレームはヘッダーで区別する。検出なしの場合も空の配列を配信する。
関節名は `nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear`,
`left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`,
`right_wrist`, `left_hip`, `right_hip`, `left_knee`, `right_knee`,
`left_ankle`, `right_ankle` の順。

パラメータ: `hef_path` (既定 `/ws/yolov8m_pose.hef`), `device` (既定 `/dev/video0`),
`width`, `height`, `cam_fps`, `score_threshold` (人物検出、既定 0.5),
`keypoint_threshold` (描画のみ、既定 0.5), `iou_threshold` (NMS、既定 0.45),
`max_detections` (既定 100), `frame_id`, `publish_raw`, `publish_compressed`,
`jpeg_quality`, `timeout_ms`。画像トピックは sensor data QoS を使用する。
