#!/bin/bash
CUDA_VISIBLE_DEVICES=3,1
eval "$(conda shell.bash hook)"
conda activate embedding-api
uvicorn main:app --reload --host 0.0.0.0 --port 7965