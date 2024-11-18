# Embeddings API

## Install

```
conda create -n embeddings-api python=3.11 -y
conda activate embeddings-api
```

```
pip install -U pip
pip install -r requirements.txt

# pip install flash-attn --no-build-isolation
# pip list --format=freeze > requirements.txt
```

## Run

```
uvicorn main:app --reload --host 0.0.0.0 --port 7998

# or on windows
./main.bat

# or on linux
./main.sh
```

## Use

```
http://localhost:7965/v1/embed
```

## Supported Models

- text-embedding-3-large
- text-embedding-3-small
- text-embedding-ada-002
- multilingual-e5-large-instruct