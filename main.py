import numpy as np
import os
import tiktoken
import torch
from BCEmbedding import EmbeddingModel
from FlagEmbedding import BGEM3FlagModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any

from sentence_transformers import SentenceTransformer


def serialize_ndarray(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class EmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    engine: Optional[str] = None
    input: Union[str, List[Any]]
    user: Optional[str] = None
    encoding_format: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: UsageInfo


WORKER_API_EMBEDDING_BATCH_SIZE = int(os.getenv("FASTCHAT_WORKER_API_EMBEDDING_BATCH_SIZE", 4))

# conan_embedding_v1 = EmbeddingModel(model_name_or_path="TencentBAC/Conan-embedding-v1", device='cuda',
#                                     trust_remote_code=True)

# xiaobu_embedding_v2 = EmbeddingModel(model_name_or_path="lier007/xiaobu-embedding-v2", device='cuda',
#                                      trust_remote_code=True)

#
# bce_embedding_base_v1 = EmbeddingModel(model_name_or_path="maidalun1020/bce-embedding-base_v1", device='cuda',
#                                        trust_remote_code=True)

bge_m3 = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cuda')

# bge_multilingual_gemma2 = SentenceTransformer("BAAI/bge-multilingual-gemma2",
#                                               model_kwargs={"torch_dtype": torch.float16})

e5_large_instruct = SentenceTransformer('intfloat/multilingual-e5-large-instruct', trust_remote_code=True,
                                        device='cuda')

# e5_large = SentenceTransformer('intfloat/multilingual-e5-large', trust_remote_code=True,
#                                device='cuda')

# qte_qwen = SentenceTransformer("Alibaba-NLP/gte-Qwen1.5-7B-instruct", trust_remote_code=True, device='cuda')


# zpoint_large_embedding_zh = SentenceTransformer('iampanda/zpoint_large_embedding_zh', device='cuda',
#                                                 trust_remote_code=True)

app = FastAPI()


def get_embedding_model(model_name: str):
    print(f"{model_name=}")
    # if model_name == 'text-embedding-3-small' or model_name == 'text-embedding-ada-002' or model_name == 'TencentBAC/Conan-embedding-v1' or model_name == 'TencentBAC/Conan-embedding-v1':
    #     return conan_embedding_v1
    # if model_name == 'text-embedding-3-small' or model_name == 'text-embedding-ada-002' or model_name == 'lier007/xiaobu-embedding-v2' or model_name == 'xiaobu-embedding-v2':
    #     return xiaobu_embedding_v2
    # elif model_name == 'lier007/xiaobu-embedding-v2' or model_name == 'xiaobu-embedding-v2':
    #     return xiaobu_embedding_v2
    # elif model_name == 'maidalun1020/bce-embedding-base_v1' or model_name == 'bce-embedding-base_v1':
    #     return bce_embedding_base_v1
    # elif model_name == 'iampanda/zpoint_large_embedding_zh' or model_name == 'zpoint_large_embedding_zh':
    #     return zpoint_large_embedding_zh
    if model_name == 'BAAI/bge-m3' or model_name == 'bge-m3':
        return bge_m3
    # elif model_name == 'BAAI/bge-multilingual-gemma2' or model_name == 'bge-multilingual-gemma2':
    #     return bge_multilingual_gemma2
    if model_name == "gpt-4" or model_name == 'intfloat/multilingual-e5-large-instruct' or model_name == 'multilingual-e5-large-instruct':
        return e5_large_instruct
    # if model_name == 'intfloat/multilingual-e5-large' or model_name == 'multilingual-e5-large':
    #     return e5_large
    # elif model_name == 'Alibaba-NLP/gte-Qwen1.5-7B-instruct':
    #     return qte_qwen
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def process_input(model_name, inp):
    if isinstance(inp, str):
        inp = [inp]
    elif isinstance(inp, list):
        if isinstance(inp[0], int):
            try:
                decoding = tiktoken.model.encoding_for_model(model_name)
            except KeyError:
                model = "cl100k_base"
                decoding = tiktoken.get_encoding(model)
            inp = [decoding.decode(inp)]
        elif isinstance(inp[0], list):
            try:
                decoding = tiktoken.model.encoding_for_model(model_name)
            except KeyError:
                model = "cl100k_base"
                decoding = tiktoken.get_encoding(model)
            inp = [decoding.decode(text) for text in inp]
    return inp


@app.post("/v1/embeddings")
@app.post("/v1/engines/{model_name}/embeddings")
async def create_embeddings(request: EmbeddingsRequest, model_name: str = None) -> dict:
    print(f"{request.model=}")
    print(f"{model_name=}")
    if request.model is None:
        request.model = model_name
    try:
        embedding_model = get_embedding_model(request.model)
        print(f"{embedding_model=}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    request.input = process_input(request.model, request.input)
    data = []
    token_num = 0
    batch_size = WORKER_API_EMBEDDING_BATCH_SIZE
    batches = [
        request.input[i: min(i + batch_size, len(request.input))]
        for i in range(0, len(request.input), batch_size)
    ]

    for num_batch, batch in enumerate(batches):
        if request.model == 'BAAI/bge-m3' or request.model == 'bge-m3':
            embeddings = [embedding_model.encode(sentence)['dense_vecs'] for sentence in batch]
        elif request.model == 'BAAI/bge-multilingual-gemma2' or request.model == 'bge-multilingual-gemma2':
            for sentence in batch:
                print(f"{sentence=}")
            embeddings = [embedding_model.encode(sentence) for sentence in batch]
        elif request.model == 'lier007/xiaobu-embedding-v2' or request.model == 'xiaobu-embedding-v2':
            embeddings = embedding_model.encode(batch, normalize_embeddings=True)
        elif request.model == 'text-embedding-3-small' or request.model == 'text-embedding-ada-002' or request.model == 'xiaobu-embedding-v2':
            embeddings = embedding_model.encode(batch, normalize_embeddings=True)
        else:
            embeddings = embedding_model.encode(batch)
        data += [
            {
                "object": "embedding",
                "embedding": serialize_ndarray(emb),
                "index": num_batch * batch_size + i,
            }
            for i, emb in enumerate(embeddings)
        ]
        token_num += sum(len(sent) for sent in batch)

    usage = UsageInfo(
        prompt_tokens=token_num,
        total_tokens=token_num,
        completion_tokens=None,
    )

    response = EmbeddingsResponse(
        data=data,
        model=request.model,
        usage=usage,
    )

    response_dump = response.model_dump(exclude_none=True)
    return response_dump


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7965)
