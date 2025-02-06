#Teste DeepSeek - ChatGPT falou pra criar um servidor

# wsl pra abrir o terminal do linux no windows

# Comando pra rodar o servidor DeepSeek
# vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager
# vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 1 --max-model-len 8192 --swap-space 2 --enforce-eager
# vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 1 --max-model-len 8192 --swap-space 2 --dtype=half --enforce-eager
# vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 1 --max-model-len 8192 --swap-space 2 --dtype=half --cpu-offload-gb 2 --enforce-eager

# Modelo DeepSeek-Coder - menor(?)
# vllm serve deepseek-ai/deepseek-coder-1.3b --tensor-parallel-size 1 --max-model-len 4096 --swap-space 2 --dtype=half --enforce-eager
# vllm serve deepseek-ai/deepseek-coder-1.3b --tensor-parallel-size 1 --max-model-len 4096 --swap-space 2 --dtype=half --enforce-eager --trust-remote-code

import requests

# URL do servidor do DeepSeek-R1 rodando no vLLM
url = "http://localhost:8000/generate"

# Parâmetros da requisição
data = {
    "prompt": "Tell me a joke.",
    "max_tokens": 100
}

# Enviar a requisição para o servidor
response = requests.post(url, json=data)

# Exibir a resposta do modelo
print(response.json())
