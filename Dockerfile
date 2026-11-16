# ROS 2 Humble + HailoRT (libhailort / pyhailort) for Raspberry Pi 5 + AI HAT+ (Hailo-8)
# ビルドコンテキストの pkgs/ に以下を置いておくこと:
#   hailort_${HAILORT_VERSION}_arm64.deb
#   hailort-${HAILORT_VERSION}-cp310-cp310-linux_aarch64.whl
FROM ros:humble

ARG HAILORT_VERSION=4.23.0
ENV DEBIAN_FRONTEND=noninteractive

COPY pkgs/hailort_${HAILORT_VERSION}_arm64.deb \
     pkgs/hailort-${HAILORT_VERSION}-cp310-cp310-linux_aarch64.whl \
     /tmp/pkgs/

# HailoRT 本体 (curl は uv の取得用)
# コンテナには systemd が無く postinst の systemctl 呼び出しで失敗するため、
# インストール中だけ何もしないダミーの systemctl を置く
RUN touch /.dockerenv \
    && apt-get update && apt-get install -y --no-install-recommends \
        /tmp/pkgs/hailort_${HAILORT_VERSION}_arm64.deb \
        curl ca-certificates python3-opencv ros-humble-vision-msgs \
    && rm /.dockerenv \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

# uv と推論用 venv (rclpy などを使うため system-site-packages を有効化)
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && uv venv --system-site-packages --python /usr/bin/python3.10 /opt/hailo_venv \
    && uv pip install --python /opt/hailo_venv/bin/python \
        /tmp/pkgs/hailort-${HAILORT_VERSION}-cp310-cp310-linux_aarch64.whl \
        "numpy<2" \
    && rm -rf /tmp/pkgs /root/.cache/uv

# venv を常に有効化 (ROS 環境は ros:humble 標準の /ros_entrypoint.sh が source する)
ENV VIRTUAL_ENV=/opt/hailo_venv
ENV PATH=/opt/hailo_venv/bin:${PATH}

WORKDIR /ws
CMD ["bash"]