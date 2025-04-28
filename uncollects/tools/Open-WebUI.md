# Open-WebUI

```shell
podman run -d -p 8080:8080 -e HF_ENDPOINT=https://hf-mirror.com -e 'OLLAMA_BASE_URL=http://host.containers.internal:11434' -e 'WEBUI_URL=http://localhost:8080' -e 'WEBUI_AUTH=False' -e 'OFFLINE_MODE=True' -e 'VECTOR_DB=pgvector' -e 'RAG_EMBEDDING_ENGINE=ollama' -e 'RAG_EMBEDDING_MODEL=bge-m3:latest' -e 'RAG_EMBEDDING_BATCH_SIZE=10' -e 'RAG_OLLAMA_BASE_URL=http://host.containers.internal:11434' -e 'ENABLE_RAG_WEB_SEARCH=True' -e 'RAG_WEB_SEARCH_ENGINE=duckduckgo' -e 'DATABASE_URL=postgresql://openwebui:openwebui@host.containers.internal:5432/openwebui' -e 'DATABASE_POOL_SIZE=10' -e 'http_proxy=http://host.containers.internal:1081' -e 'https_proxy=http://host.containers.internal:1081' -e 'WHISPER_MODEL=turbo' --device nvidia.com/gpu=all -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:cuda
```

```shell
podman run -d -p 8080:8080 -e HF_ENDPOINT=https://hf-mirror.com -e 'OLLAMA_BASE_URL=http://host.containers.internal:11434' -e 'WEBUI_URL=http://localhost:8080' -e 'WEBUI_AUTH=False' -e 'ENABLE_OPENAI_API=False' -e 'OFFLINE_MODE=True' -e 'DATABASE_URL=postgresql://openwebui:openwebui@host.containers.internal:5432/openwebui' -e 'DATABASE_POOL_SIZE=10' -e 'http_proxy=' -e 'https_proxy=' -e 'WHISPER_MODEL=turbo' --device nvidia.com/gpu=all -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:cuda
```

```shell
podman run -d -p 8080:8080 -e HF_ENDPOINT=https://hf-mirror.com -e 'OLLAMA_BASE_URL=http://host.containers.internal:11434' -e 'WEBUI_URL=http://localhost:8080' -e 'WEBUI_AUTH=False' -e 'OFFLINE_MODE=True' -e 'VECTOR_DB=pgvector' -e 'RAG_EMBEDDING_ENGINE=ollama' -e 'RAG_EMBEDDING_MODEL=bge-m3:latest' -e 'RAG_EMBEDDING_BATCH_SIZE=10' -e 'RAG_OLLAMA_BASE_URL=http://host.containers.internal:11434' -e 'ENABLE_RAG_WEB_SEARCH=True' -e 'RAG_WEB_SEARCH_ENGINE=duckduckgo' -e 'DATABASE_URL=postgresql://openwebui:openwebui@host.containers.internal:5432/openwebui' -e 'DATABASE_POOL_SIZE=10' -e 'http_proxy=http://host.containers.internal:1081' -e 'https_proxy=http://host.containers.internal:1081' -e 'no_proxy=host.containers.internal' -e 'WHISPER_MODEL=turbo' --device nvidia.com/gpu=all -v open-webui:/app/backend/data --name open-webui open-webui:cuda-fix
```
