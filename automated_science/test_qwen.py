import requests, re
prompt = "We are fitting a mathematical model for ecology...\nOutput ONLY a Python code block with dynamics(t, y, args) and metadata.\nUse jnp.clip(..., -1e6, 1e6)"
payload = {
    "model": "qwen/qwen3-coder-next", 
    "messages": [
        {"role": "system", "content": "You are a mathematical AI. Output ONLY a Python code block with `dynamics(t, y, args)` and `metadata`. No prose."}, 
        {"role": "user", "content": prompt}
    ], 
    "temperature": 0.2, 
    "max_tokens": 8192
}
r = requests.post("http://localhost:1234/v1/chat/completions", json=payload).json()
content = r["choices"][0]["message"]["content"]
print("LLM OUTPUT:\n", content)
