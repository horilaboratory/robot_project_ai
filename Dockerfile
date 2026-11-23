# ROS 2 Humble + HailoRT (libhailort / pyhailort) for Raspberry Pi 5 + AI HAT+ (Hailo-8)
# docker build 時に HailoRT をビルドしてインストール
FROM ros:humble

ARG HAILORT_VERSION=4.23.0
ENV DEBIAN_FRONTEND=noninteractive

# ビルド依存パッケージとランタイム依存パッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake python3-dev curl ca-certificates \
        python3-opencv ros-humble-vision-msgs \
    && rm -rf /var/lib/apt/lists/*

# HailoRT をクローンしてビルド
RUN cd /opt \
    && git clone --branch v${HAILORT_VERSION} --depth 1 https://github.com/hailo-ai/hailort.git \
    && cd hailort \
    && cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config release --target install -j4 \
    && ldconfig

# uv をインストール
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# 推論用 venv をセットアップ (rclpy などを使うため system-site-packages を有効化)
RUN uv venv --system-site-packages --python /usr/bin/python3.10 /opt/hailo_venv \
    && . /opt/hailo_venv/bin/activate \
    && uv pip install setuptools wheel "numpy<2"

# HailoRT Python バインディングをビルドしてインストール
RUN . /opt/hailo_venv/bin/activate \
    && cd /opt/hailort/hailort/libhailort/bindings/python/platform \
    && python setup.py bdist_wheel \
    && uv pip install dist/hailort-${HAILORT_VERSION}-*.whl \
    && cd / \
    && rm -rf /opt/hailort /root/.cache/uv

# venv を常に有効化 (ROS 環境は ros:humble 標準の /ros_entrypoint.sh が source する)
ENV VIRTUAL_ENV=/opt/hailo_venv
ENV PATH=/opt/hailo_venv/bin:${PATH}

WORKDIR /ws
CMD ["bash"]