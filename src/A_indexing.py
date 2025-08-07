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

        if "pdf_data" in data:
            data["페이지별_데이터"] = data.pop("pdf_data")

        metadata_source = data.get("csv_metadata", {})
        chunks = extract_texts_from_json(data)

        # 검색 성능을 높이기 위해 중요한 메타데이터 헤더 추가
        metadata_header = (
            f"[문서 요약 정보]\n"
            f"- 사업명: {metadata_source.get('사업명', '정보 없음')}\n"
            f"- 사업 금액: {metadata_source.get('사업 금액', '정보 없음')}\n"
            f"- 발주 기관: {metadata_source.get('발주 기관', '정보 없음')}\n"
            f"- 파일명: {metadata_source.get('파일명', '정보 없음')}\n\n"
            f"- 사업 요약: {metadata_source.get('사업 요약', '정보 없음')}\n\n"
        )

        for i, chunk in enumerate(chunks):
            chunk_with_metadata = metadata_header + chunk
            
            metadata = {
                "사업명": metadata_source.get("사업명", ""),
                "공고번호": metadata_source.get("공고 번호", ""),
                "공고차수": metadata_source.get("공고 차수", ""),
                "사업금액": metadata_source.get("사업 금액", ""),
                "발주기관": metadata_source.get("발주 기관", ""),
                "입찰참여시작일": metadata_source.get("입찰 참여 시작일", ""),
                "입찰참여마감일": metadata_source.get("입찰 참여 마감일", ""),
                "사업요약": metadata_source.get("사업 요약", ""),
                "파일명": metadata_source.get("파일명", ""),
                "source": file_name,
                "청크번호": i, # A 모델에만 필요
            }
            docs.append(Document(page_content=chunk_with_metadata, metadata=metadata))

    vectorstore = FAISS.from_documents(docs, embedding_model)
    model_output_path = os.path.join(output_path, model_key)
    os.makedirs(model_output_path, exist_ok=True)
    vectorstore.save_local(model_output_path, index_name="index")
    print(f"FAISS 저장 완료: 총 문서 수 = {len(docs)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True, default="/home/data/preprocess/json")
    parser.add_argument("--model_key", type=str, default="kr-sbert")
    parser.add_argument("--output_path", type=str, default="/home/data/A_faiss_db")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    index_documents(args.json_dir, args.model_key, args.output_path)
    # python src/A_indexing.py --json_dir /home/data/preprocess/json --model_key kr-sbert --output_path /home/data/A_faiss_db