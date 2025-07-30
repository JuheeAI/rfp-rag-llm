import argparse
import faiss
import pickle
import numpy as np
from A_embedding_model import load_embedding_model

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="질의 문장")
    parser.add_argument("--model_key", type=str, default="kr-sbert", help="임베딩 모델 키")
    parser.add_argument("--index_path", type=str, required=True, help="FAISS 인덱스 경로")
    parser.add_argument("--meta_path", type=str, required=True, help="메타데이터 pkl 경로")
    parser.add_argument("--top_k", type=int, default=3, help="검색할 문서 수")
    args = parser.parse_args()

    model = load_embedding_model(args.model_key)
    index = load_faiss_index(args.index_path)
    if isinstance(index, faiss.IndexIVF):
        index.nprobe = 16  # IVF 기반 인덱스의 검색 정확도를 위해 nprobe 설정
    metadata = load_metadata(args.meta_path)

    query_vec = embed_query(args.query, model)
    indices, scores = retrieve_top_k(query_vec, index, top_k=args.top_k)

    display_results(indices, scores, metadata)