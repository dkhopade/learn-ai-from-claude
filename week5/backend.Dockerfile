FROM python:3.13-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY week5/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# pre-cache the embedding model into the image so pods start fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# the Day 1 backend app
COPY week5/ .

# the Day 2 agent team (analyst + sql specialist) + its database builder
COPY day2/agents /app/agents
COPY day2/build_db.py /app/build_db.py

ENV LLM_BACKEND=vllm
ENV LLM_BASE_URL=http://vllm:8000
ENV LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
ENV GATEWAY_URL=http://ai-gateway:8080

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
