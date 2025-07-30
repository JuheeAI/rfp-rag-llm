

import os
import json
import faiss
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from A_embedding_model import extract_texts_from_json, load_embedding_model, embed_texts

def index_documents(json_dir, model_key, output_path):
    model = load_embedding_model(model_key)
    all_embeddings = []
    all_metadatas = []
    id2doc = {}

    for file_name in tqdm(os.listdir(json_dir), desc="Indexing JSON files"):
        if not file_name.endswith(".json"):
            continue
        file_path = os.path.join(json_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = extract_texts_from_json(data)
        if not chunks:
            continue
        embeddings = embed_texts(model, chunks, for_faiss=True)
        all_embeddings.append(embeddings)

        for i, chunk in enumerate(chunks):
            meta = {
                "파일명": file_name,
                "청크번호": i,
                "공고번호": data.get("공고번호", "N/A"),
                "사업명": data.get("사업명", "N/A"),
                "본문": chunk,
            }
            all_metadatas.append(meta)
            id2doc[len(id2doc)] = meta

    # FAISS 인덱스 생성
    all_embeddings = np.vstack(all_embeddings).astype(np.float32)
    dim = all_embeddings.shape[1]
    if len(all_embeddings) < 40:
        print("데이터 수가 적어 IVF+PQ 훈련이 어려워 IndexFlatL2로 대체합니다.")
        index = faiss.IndexFlatL2(dim)
        index.add(all_embeddings)
    else:
        quantizer = faiss.IndexFlatL2(dim)
        nlist = min(100, len(all_embeddings))
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, 8, 8)
        index.train(all_embeddings)
        assert index.is_trained, "FAISS 인덱스가 훈련되지 않았습니다."
        index.add(all_embeddings)

    # 저장
    faiss.write_index(index, os.path.join(output_path, "faiss.index"))
    with open(os.path.join(output_path, "faiss_meta.pkl"), "wb") as f:
        pickle.dump(id2doc, f)

    print(f"FAISS 인덱싱 완료: 총 문서 수 = {len(id2doc)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--model_key", type=str, default="kr-sbert")
    parser.add_argument("--output_path", type=str, default="./faiss_db")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    index_documents(args.json_dir, args.model_key, args.output_path)