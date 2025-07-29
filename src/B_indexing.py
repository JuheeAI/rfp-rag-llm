import os, json, faiss, argparse
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from B_embedding_model import extract_texts_from_json, load_embedding_model

def build_faiss_index(json_folder_path: str, model_key: str = "kr-sbert", save_path: str = "faiss_index"):
    model = load_embedding_model(model_key)
    index = None
    metadata = []

    all_texts_for_tfidf = []

    for filename in os.listdir(json_folder_path):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(json_folder_path, filename)

        texts = extract_texts_from_json(json_data)
        all_texts_for_tfidf.extend(texts)
        if not texts:
            print(f"[WARN] {filename}에서 추출된 텍스트가 없습니다.")
            continue

        embeddings = model.encode(texts, convert_to_numpy=True)
        if index is None:
            index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        metadata.extend([
            {
                "파일명": json_data.get("파일명", filename),
                "공고번호": json_data.get("공고번호", ""),
                "사업명": json_data.get("사업명", ""),
                "text": text,
                "text_idx": i
            }
            for i, text in enumerate(texts)
        ])

    faiss.write_index(index, f"{save_path}.index")
    with open(f"{save_path}_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts_for_tfidf)
    # print(f"[DEBUG] TF-IDF 단어 수: {len(vectorizer.vocabulary_)}개")
    # print(f"[DEBUG] 상위 단어 샘플: {list(vectorizer.vocabulary_.keys())[:10]}")
    with open(f"{save_path}_tfidf.pkl", "wb") as f:
        pickle.dump((vectorizer, tfidf_matrix), f)
        # print(f"[DEBUG] TF-IDF 벡터 크기: {tfidf_matrix.shape}")
        # print("[DEBUG] TF-IDF vectorizer 저장 완료")

    print(f"저장 완료: {save_path}.index / {save_path}_meta.json / {save_path}_tfidf.pkl")

def load_faiss_index(index_path: str) -> faiss.Index:
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index 파일을 찾을 수 없습니다: {index_path}")
    return faiss.read_index(index_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAISS 인덱스 생성 스크립트")
    parser.add_argument("--json_dir", type=str, required=True, help="JSON 파일들이 저장된 폴더 경로")
    parser.add_argument("--model_key", type=str, default="kr-sbert", help="임베딩에 사용할 모델 키 (kr-sbert | ko-sbert | kosimcse)")
    parser.add_argument("--save_path", type=str, default="faiss_index", help="인덱스 저장 경로 prefix")

    args = parser.parse_args()
    build_faiss_index(args.json_dir, model_key=args.model_key, save_path=args.save_path)