#!/bin/bash

# 1. 更新 pip
pip install --upgrade pip

# 2. 安装依赖 (使用清华源加速，或者直接用默认源)
echo "Installing dependencies..."
pip install -r requirements.txt

# 3. 登录 HuggingFace (如果模型是私有的，Qwen2.5 是公开的，通常不需要这一步)
# huggingface-cli login --token YOUR_TOKEN

# 4. 运行仿真
echo "Starting Simulation..."
python main.py

# 5. (可选) 如果你想运行 Web UI
# echo "Starting Web UI..."
# python app.py