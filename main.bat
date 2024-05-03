call conda activate reranker-api
uvicorn main:app --reload --host 0.0.0.0 --port 7965