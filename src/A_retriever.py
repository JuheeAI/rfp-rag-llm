import faiss
import pickle
import numpy as np
from A_embedding import load_embedding_model
from langchain_core.documents import Document

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def load_metadata(meta_path):
    with open(meta_path, "rb") as f:
        return pickle.load(f)

def embed_query(query, model):
    vec = model.encode([query], convert_to_numpy=True)
    return vec.astype(np.float32)

def retrieve_top_k(query_vec, index, top_k=3):
    if isinstance(index, faiss.IndexFlat):
        print("IndexFlat 방식으로 검색을 수행합니다.")
    D, I = index.search(query_vec, top_k)
    return I[0], D[0]

def display_results(indices, scores, metadata):
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        if idx == -1:
            continue
        meta = metadata.get(idx, {})
        print(f"\n[결과 {rank}]")
        print(f"▶ 유사도 점수: {score:.4f}")
        print(f"▶ 관련 문장: {meta.get('text', 'N/A')}")
        print(f"▶ 메타정보: {meta}")

from langchain.vectorstores.faiss import FAISS
from langchain_core.embeddings import Embeddings

def load_langchain_retriever(base_path: str, model_key: str, embedding_model: Embeddings):
    full_path = f"{base_path}/{model_key}"
    return FAISS.load_local(full_path, embedding_model, allow_dangerous_deserialization=True).as_retriever()