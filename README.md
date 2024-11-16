# embedding-api
Embedding API

## Prepare

```
conda create -n embedding-api python=3.11 -y
conda activate embedding-api
```

```
pip install -U pip
pip install -r requirements.txt

pip install flash-attn==2.5.8
# pip list --format=freeze > requirements.txt
```

## Run

```
uvicorn main:app --reload --host 0.0.0.0 --port 7998
or on windows
./main.bat
or on linux
./main.sh
```

## Use

```
http://localhost:7965/v1/embed
```


## Supported Models
- text-embedding-ada-002
- bce-embedding-base_v1
- zpoint_large_embedding_zh
- 