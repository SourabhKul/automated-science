#!/bin/bash
# Phase 7: Cognitive Expansion [MLX Direct Server Launch]

echo "Bypassing Python 3.14 C-Extension Compilation Failures..."
echo "Binding previously downloaded Qwen122B natively into Apple Metal via MLX endpoint (Port 1235)..."

source /Users/sourabh/Documents/Projects/workspace/automated_science/.venv/bin/activate

# --- SERVER RUNTIME CONFIGURATION ---
# The target Qwen matrix is inherently formatted as an MXFP4 tensor structure.
# Instead of forcing it through buggy 3rd party vLLM C++ wrappers, we utilize the native Apple MLX pipeline.

python3 -m mlx_lm.server \
    --model "nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx" \
    --port 1235 \
    --chat-template "chatml" \
    --max-tokens 4096 \
    --prompt-cache-bytes 1073741824 \
    --trust-remote-code
