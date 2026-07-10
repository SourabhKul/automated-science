#!/bin/bash
# Phase 7: Cognitive Expansion [Environment & Cache Setup]

VLLM_DIR="/Users/sourabh/Documents/Projects/vllm-mlx"

echo "Initializing vLLM-MLX Server Environment..."

if [ ! -d "$VLLM_DIR" ]; then
    echo "Cloning official vLLM repo into target vllm-mlx directory..."
    git clone https://github.com/vllm-project/vllm.git $VLLM_DIR
fi

cd $VLLM_DIR

if [ ! -d "$VLLM_DIR/.venv" ]; then
    echo "Building Apple Silicon inference environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    
    # Install vLLM forcing the macos metal backend configurations
    pip install -e .
else
    source .venv/bin/activate
fi

echo "Pre-fetching the massive 122B Model Array via HuggingFace Hub to prevent runtime timeouts..."
pip install huggingface_hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx')"

echo "=========================================================="
echo "Download & Environment Setup Complete!"
echo "Server has NOT been launched to preserve memory for IL-6."
echo "Execute 'start_122B_server.sh' when ready to load models!"
echo "=========================================================="
