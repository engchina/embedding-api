import datetime as dt
import os
import typing
from typing import List, Optional, Union, Dict, Any

import numpy as np
import tiktoken
import torch
# from BCEmbedding import EmbeddingModel
# from FlagEmbedding import BGEM3FlagModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity

from core.datetime_utils import serialize_datetime
from core.pydantic_utilities import pydantic_v1
from core.unchecked_base_model import UncheckedBaseModel


# 优化点 1：提升 NumPy 数组序列化的效率
def serialize_ndarray(value: Any) -> Any:
    return value.tolist() if isinstance(value, np.ndarray) else value


# Pydantic 模型定义
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


class ClassifyExample(UncheckedBaseModel):
    text: Optional[str] = None
    label: Optional[str] = None

    def json(self, **kwargs: typing.Any) -> str:
        return super().json(by_alias=True, exclude_unset=True, **kwargs)

    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        return super().dict(by_alias=True, exclude_unset=True, **kwargs)

    class Config:
        frozen = True
        smart_union = True
        extra = pydantic_v1.Extra.allow
        json_encoders = {dt.datetime: serialize_datetime}


class DocumentClassifyManager(BaseModel):
    classify_model: str
    query_text: str
    example_docs: List[Dict]


# 优化点 2：减少常量加载时间
WORKER_API_EMBEDDING_BATCH_SIZE = int(os.getenv("FASTCHAT_WORKER_API_EMBEDDING_BATCH_SIZE", 4))

MODELS = {
    # "text-embedding-3-large": SentenceTransformer(
    #     model_name_or_path="intfloat/multilingual-e5-large-instruct", trust_remote_code=True, device="cuda"
    # ),
    "text-embedding-3-large": SentenceTransformer(
        model_name_or_path="jinaai/jina-embeddings-v3", trust_remote_code=True, device="cuda",
        model_kwargs={'default_task': 'retrieval.query'}
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
    # "jina-embeddings-v3": SentenceTransformer(
    #     model_name_or_path="jinaai/jina-embeddings-v3", trust_remote_code=True, device="cuda",
    #     model_kwargs={'default_task': 'retrieval.query'}
    # ),
}

app = FastAPI()


def get_embedding_model(model_name: str):
    """获取嵌入模型实例。"""
    model = MODELS.get(model_name)
    if not model:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


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
async def create_embeddings(request: EmbeddingsRequest, model_name: str = "text-embedding-3-large") -> dict:
    """生成嵌入向量的 API 接口。"""
    request.model = request.model or model_name
    try:
        embedding_model = get_embedding_model(request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    inputs = process_input(request.model, request.input)
    data, token_num = [], 0
    batch_size = WORKER_API_EMBEDDING_BATCH_SIZE

    # 优化点 3：使用生成器处理批量数据，减少内存开销
    def batched_data(inputs, batch_size):
        for i in range(0, len(inputs), batch_size):
            yield inputs[i:i + batch_size]

    for num_batch, batch in enumerate(batched_data(inputs, batch_size)):
        # embeddings = embedding_model.encode(batch, convert_to_tensor=True)
        embeddings = embedding_model.encode(batch)
        data.extend(
            {
                "object": "embedding",
                # "embedding": serialize_ndarray(emb.cpu().float().numpy()),
                "embedding": serialize_ndarray(emb),
                "index": num_batch * batch_size + i,
            }
            for i, emb in enumerate(embeddings)
        )
        token_num += sum(len(text) for text in batch)

    usage = UsageInfo(prompt_tokens=token_num, total_tokens=token_num)
    response = EmbeddingsResponse(data=data, model=request.model, usage=usage)
    return response.model_dump(exclude_none=True)


@app.post("/v1/classify")
async def classify_query(manager: DocumentClassifyManager, model_name: str = "text-embedding-3-large") -> str:
    """
    使用嵌入模型对查询进行分类。
    """
    classify_model = manager.classify_model or model_name
    query_text = manager.query_text
    example_docs = manager.example_docs

    examples = [ClassifyExample(**doc) for doc in example_docs]
    query_embeddings = await get_embeddings(classify_model, [query_text])
    example_embeddings = await get_embeddings(classify_model, [example.text for example in examples])

    similarities = calculate_similarities(query_embeddings[0], example_embeddings, examples)

    label_similarities = {}
    for similarity in similarities:
        label = similarity["label"]
        label_similarities.setdefault(label, []).append(similarity["similarity"])

    return max(label_similarities, key=lambda label: sum(label_similarities[label]) / len(label_similarities[label]))


async def get_embeddings(model_name: str, inputs: List[str]) -> List[List[float]]:
    """
    获取嵌入向量。
    """
    request = EmbeddingsRequest(model=model_name, input=inputs)
    response = await create_embeddings(request)
    return [item["embedding"] for item in response["data"]]


def calculate_similarities(query_embeddings: List[float], example_embeddings: List[List[float]],
                           examples: List[ClassifyExample]) -> List[Dict[str, float]]:
    """
    计算余弦相似度。
    """
    query_tensor = torch.tensor(query_embeddings).unsqueeze(0).to("cuda")
    example_tensors = torch.tensor(example_embeddings).to("cuda")

    similarities = cosine_similarity(query_tensor, example_tensors).squeeze(0).cpu().numpy()
    return [{"label": example.label, "similarity": similarity} for example, similarity in zip(examples, similarities)]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7965)
