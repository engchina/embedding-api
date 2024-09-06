from BCEmbedding import EmbeddingModel
from FlagEmbedding import BGEM3FlagModel

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
bce_embedding_base_v1 = EmbeddingModel(model_name_or_path="maidalun1020/bce-embedding-base_v1", device='cuda')
bge_m3 = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cuda')
# e5_large = SentenceTransformer('intfloat/multilingual-e5-large', trust_remote_code=True, device='cuda')
e5_large_instruct = SentenceTransformer('intfloat/multilingual-e5-large-instruct', trust_remote_code=True, device='cuda')
qte_qwen = SentenceTransformer("Alibaba-NLP/gte-Qwen1.5-7B-instruct", trust_remote_code=True, device='cuda')


class DocumentRankerManager(BaseModel):
    sentences: list
    embedding_model: str


@app.post("/v1/embed")
def embed_docs(manager: DocumentRankerManager) -> list:
    embedding_model = manager.embedding_model
    sentences = manager.sentences
    print(f"{sentences=}")

    if embedding_model == 'maidalun1020/bce-embedding-base_v1':
        embeddings = bce_embedding_base_v1.encode(sentences)
    elif embedding_model == 'BAAI/bge-m3':
        embeddings = [bge_m3.encode(sentence)['dense_vecs'] for sentence in sentences]
    # elif embedding_model == 'intfloat/multilingual-e5-large':
    #     embeddings = e5_large.encode(sentences, normalize_embeddings=True)
    elif embedding_model == 'intfloat/multilingual-e5-large-instruct':
        embeddings = e5_large_instruct.encode(sentences, normalize_embeddings=True)
    elif embedding_model == 'Alibaba-NLP/gte-Qwen1.5-7B-instruct':
        embeddings = qte_qwen.encode(sentences)

    else:
        embeddings = e5_large_instruct.encode(sentences, normalize_embeddings=True)
    return [list(map(float, e)) for e in embeddings]
