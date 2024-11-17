import os
from typing import List, Optional, Union, Dict, Any

import numpy as np
import tiktoken
# from BCEmbedding import EmbeddingModel
# from FlagEmbedding import BGEM3FlagModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# Helper to serialize NumPy arrays
def serialize_ndarray(value: Any) -> Any:
    return value.tolist() if isinstance(value, np.ndarray) else value


# Pydantic models for request and response
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


# Constants and model initialization
WORKER_API_EMBEDDING_BATCH_SIZE = int(os.getenv("FASTCHAT_WORKER_API_EMBEDDING_BATCH_SIZE", 4))

MODELS = {
    "text-embedding-3-large": SentenceTransformer(
        "intfloat/multilingual-e5-large-instruct", trust_remote_code=True, device="cuda"
    ),
    # "text-embedding-3-small": SentenceTransformer(
    #     "intfloat/multilingual-e5-large-instruct", trust_remote_code=True, device="cuda"
    # ),
    # "text-embedding-ada-002": SentenceTransformer(
    #     "intfloat/multilingual-e5-large-instruct", trust_remote_code=True, device="cuda"
    # ),
    # "multilingual-e5-large-instruct": SentenceTransformer(
    #     "intfloat/multilingual-e5-large-instruct", trust_remote_code=True, device="cuda"
    # ),
    # "multilingual-e5-large": SentenceTransformer(
    #     "intfloat/multilingual-e5-large", trust_remote_code=True, device="cuda"
    # ),
    # "Conan-embedding-v1": EmbeddingModel(
    #     model_name_or_path="TencentBAC/Conan-embedding-v1", device="cuda", trust_remote_code=True
    # ),
    # "xiaobu-embedding-v2": EmbeddingModel(
    #     model_name_or_path="lier007/xiaobu-embedding-v2", device="cuda", trust_remote_code=True
    # ),
    # "bce-embedding-base_v1": EmbeddingModel(
    #     model_name_or_path="maidalun1020/bce-embedding-base_v1", device="cuda", trust_remote_code=True
    # ),
    # "bge-m3": BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda"),
    # "bge-multilingual-gemma2": SentenceTransformer(
    #     "BAAI/bge-multilingual-gemma2", model_kwargs={"torch_dtype": torch.float16}
    # ),
    # "gte-Qwen1.5-7B-instruct": SentenceTransformer(
    #     "Alibaba-NLP/gte-Qwen1.5-7B-instruct", trust_remote_code=True, device="cuda"
    # ),
    # "zpoint_large_embedding_zh": SentenceTransformer(
    #     "iampanda/zpoint_large_embedding_zh", device="cuda", trust_remote_code=True
    # ),
}

app = FastAPI()


def get_embedding_model(model_name: str):
    """Fetch the embedding model by name."""
    model = MODELS.get(model_name)
    if not model:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def process_input(model_name: str, inp: Union[str, List[Any]]):
    """Preprocess the input data."""
    if isinstance(inp, str):
        return [inp]
    if isinstance(inp, list):
        try:
            encoding = tiktoken.model.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return [encoding.decode(item) if isinstance(item, (list, int)) else item for item in inp]
    return inp


@app.post("/v1/embeddings")
@app.post("/v1/engines/{model_name}/embeddings")
async def create_embeddings(request: EmbeddingsRequest, model_name: str = None) -> dict:
    """Endpoint to generate embeddings."""
    request.model = request.model or model_name
    try:
        embedding_model = get_embedding_model(request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    inputs = process_input(request.model, request.input)
    data, token_num = [], 0
    batch_size = WORKER_API_EMBEDDING_BATCH_SIZE

    # Process inputs in batches
    for num_batch, batch in enumerate(
            [inputs[i: i + batch_size] for i in range(0, len(inputs), batch_size)]
    ):
        embeddings = embedding_model.encode(batch)
        data.extend(
            {
                "object": "embedding",
                "embedding": serialize_ndarray(emb),
                "index": num_batch * batch_size + i,
            }
            for i, emb in enumerate(embeddings)
        )
        token_num += sum(len(sent) for sent in batch)

    usage = UsageInfo(prompt_tokens=token_num, total_tokens=token_num)
    response = EmbeddingsResponse(data=data, model=request.model, usage=usage)
    return response.model_dump(exclude_none=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7965)
