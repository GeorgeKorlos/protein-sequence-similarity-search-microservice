FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Bake model weights at build time 
RUN python -c "\
from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D'); \
AutoModel.from_pretrained('facebook/esm2_t33_650M_UR50D')"

COPY src/ ./src/

# Corpus and ID file needed at runtime by CorpusStore
COPY data/swissprot_clean.csv ./data/
COPY data/swissprot_ids.txt ./data/
COPY data/embedding_config.json ./data/

# Index is NOT baked in — mounted at runtime via docker-compose or Cloud Run
# data/indexes/ is provided via volume mount

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.service.main:app", "--host", "0.0.0.0", "--port", "8000"]