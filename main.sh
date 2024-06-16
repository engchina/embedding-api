#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate embedding-api
uvicorn openai_api_main:app --reload --host 0.0.0.0 --port 7965