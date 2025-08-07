import os
import json
import argparse
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from A_embedding import extract_texts_from_json, load_embedding_model

def index_documents(json_dir, model_key, output_path):
    embedding_model = load_embedding_model(model_key)
    docs = []

    for file_name in tqdm(os.listdir(json_dir), desc="Indexing JSON files"):
        if not file_name.endswith(".json"):
            continue
        file_path = os.path.join(json_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = extract_texts_from_json(data)
        for i, chunk in enumerate(chunks):
            metadata = {
                "파일명": file_name,
                "청크번호": i,
                "공고번호": data.get("공고번호", "N/A"),
                "사업명": data.get("사업명", "N/A"),
            }
            docs.append(Document(page_content=chunk, metadata=metadata))

    vectorstore = FAISS.from_documents(docs, embedding_model)
    model_output_path = os.path.join(output_path, model_key)
    os.makedirs(model_output_path, exist_ok=True)
    vectorstore.save_local(model_output_path, index_name="faiss")
    print(f"FAISS 저장 완료: 총 문서 수 = {len(docs)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--model_key", type=str, default="kr-sbert")
    parser.add_argument("--output_path", type=str, default="./faiss_db")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    index_documents(args.json_dir, args.model_key, args.output_path)