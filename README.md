# embedding-api
Embedding API

## Prepare

```
conda create -n embedding-api python=3.11 -y
conda activate embedding-api
```

```
pip install -r requirements.txt
```

## Run

```
uvicorn main:app --reload --host 0.0.0.0 --port 7998
or on windows
./main.bat
or on linux
./main.sh
```