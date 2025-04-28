# Ollama

## Installation

### Linux

```shell
podman pull docker.io/ollama/ollama
unset HTTP_PROXY
unset HTTPS_PROXY
unset http_proxy
unset https_proxy
podman run -d --device nvidia.com/gpu=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
brew install ollama # For client
```

## Models

- llama3.1
- llama3.2
- qwen2.5
- qwen2.5-coder
- llama3.2-vision
- bge-m3
