import pickle, argparse, faiss, os, json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from typing import List, Tuple
from A_embedding_model import load_embedding_model, extract_texts_from_json
from A_indexing import load_faiss_index
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# 쿼리 임베딩
def get_query_embedding(query: str, model) -> np.ndarray:
    return model.encode([query]).reshape(1, -1)

# 유사한 chunk 검색
def retrieve_top_k_chunks(query: str, model, index: faiss.IndexFlatL2, corpus: List[str], k: int = 5) -> List[Tuple[int, float, str]]:
    query_vec = get_query_embedding(query, model)
    D, I = index.search(query_vec, k) 
    results = []
    for idx, dist in zip(I[0], D[0]):
        results.append((idx, dist, corpus[idx]))
    return results

# TF-IDF 기반 검색
def retrieve_tfidf_top_k(query: str, vectorizer: TfidfVectorizer, tfidf_matrix: csr_matrix, corpus: List[str], k: int = 5) -> List[Tuple[int, float, str]]:
    query_vec = vectorizer.transform([query])
    scores = (query_vec @ tfidf_matrix.T).toarray().squeeze()
    topk_idx = scores.argsort()[::-1][:k]
    return [(i, scores[i], corpus[i]) for i in topk_idx]

# 하이브리드 검색
def retrieve_hybrid_top_k(query: str, model, vectorizer: TfidfVectorizer, tfidf_matrix: csr_matrix, corpus: List[str], k: int = 5) -> List[Tuple[int, float, str]]:
    print("[DEBUG] corpus 샘플:")
    for i, text in enumerate(corpus[:5]):
        print(f"  - 문서 {i}: {text}")

    candidates = retrieve_tfidf_top_k(query, vectorizer, tfidf_matrix, corpus, k=20)
    print("[DEBUG] TF-IDF 후보군:")
    for _, score, text in candidates:
        print(f"  - 점수: {score:.8f}, 문장: {text['text'][:50]}")
    query_vec = normalize(get_query_embedding(query, model))  # shape (1, dim)
    text_vecs = normalize(model.encode([c[2]["text"] for c in candidates]))  # shape (k, dim)
    print("[DEBUG] query_vec[:5]:", query_vec[0][:5])
    print("[DEBUG] text_vecs[0][:5]:", text_vecs[0][:5])
    sim_scores = cosine_similarity(query_vec, text_vecs)[0]  # shape (k,)
    reranked = sorted(zip([c[0] for c in candidates], sim_scores, [c[2] for c in candidates]), key=lambda x: x[1], reverse=True)

    print("[DEBUG] Vectorizer 단어 수:", len(vectorizer.vocabulary_))
    sample_keywords = ["지체상금", "이미지", "위약금", "책임", "포함"]
    for kw in sample_keywords:
        in_vocab = kw in vectorizer.vocabulary_
        print(f"[DEBUG] '{kw}' in vocab? {in_vocab} -> {vectorizer.vocabulary_.get(kw, '없음')}")

    return reranked[:k]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, required=True, help="FAISS index 파일 경로 (.index)")
    parser.add_argument("--meta_path", type=str, required=True, help="메타데이터 JSON 경로")
    parser.add_argument("--query", type=str, required=True, help="검색할 사용자 쿼리")
    parser.add_argument("--model_key", type=str, default="kr-sbert", help="임베딩 모델 키 (기본: kr-sbert)")
    parser.add_argument("--tfidf_path", type=str, help="TF-IDF 인덱스 경로 (.pkl)")
    parser.add_argument("--mode", type=str, default="dense", choices=["dense", "sparse", "hybrid"], help="retrieval 방식 선택")

    args = parser.parse_args()

    model = load_embedding_model(args.model_key)
    index = load_faiss_index(args.index_path)

    with open(args.meta_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    if args.mode in ["sparse", "hybrid"]:
        with open(args.tfidf_path, "rb") as f:
            vectorizer, tfidf_matrix = pickle.load(f)

    print(f"\n[사용자 질의] {args.query}")
    if args.mode == "dense":
        top_k_chunks = retrieve_top_k_chunks(args.query, model, index, corpus, k=5)
    elif args.mode == "sparse":
        top_k_chunks = retrieve_tfidf_top_k(args.query, vectorizer, tfidf_matrix, corpus, k=5)
    elif args.mode == "hybrid":
        top_k_chunks = retrieve_hybrid_top_k(args.query, model, vectorizer, tfidf_matrix, corpus, k=5)

    print("\n[검색 결과]")
    for idx, score, _ in top_k_chunks:
        item = corpus[idx]
        print(f"\n[Index: {idx}, 점수: {score:.8f}]")
        print(f"[파일명: {item['파일명']}]")
        print(f"[사업명: {item['사업명']}]")
        print(f"본문: {item['text']}")
