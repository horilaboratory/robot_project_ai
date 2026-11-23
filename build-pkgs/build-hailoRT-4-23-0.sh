#!/bin/bash

apt update
apt install -y git build-essential cmake python3-dev curl

cd /opt
git clone --branch v4.23.0 --depth 1 https://github.com/hailo-ai/hailort.git
cd hailort

# Build and install HailoRT
cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release
cmake --build build --config release --target install -j4
ldconfig

# Install uv and make virtual environment for Python
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

source /opt/ros/humble/setup.bash
uv venv --system-site-packages --python /usr/bin/python3.10 /opt/hailo_venv
source /opt/hailo_venv/bin/activate
uv pip install setuptools wheel "numpy<2"

# Install HailoRT Python bindings
cd /opt/hailort/hailort/libhailort/bindings/python/platform
python setup.py bdist_wheel
uv pip install dist/hailort-4.23.0-*.whl