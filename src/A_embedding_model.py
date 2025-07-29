import argparse
from sentence_transformers import SentenceTransformer
from typing import List, Union
import torch, json

# 모델 목록
SUPPORTED_MODELS = {
    "kr-sbert": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    "ko-sbert": "jhgan/ko-sbert-sts",
    "kosimcse": "BM-K/KoSimCSE-roberta-multitask"
}

# 임베딩 벡터 인코딩 함수
def load_embedding_model(model_key: str) -> SentenceTransformer:
    if model_key not in SUPPORTED_MODELS:
        raise ValueError(f"지원하지 않는 모델입니다: {model_key}")
    model_name = SUPPORTED_MODELS[model_key]
    return SentenceTransformer(model_name)

# 텍스트 인코딩 함수
def extract_texts_from_json(json_data: dict) -> List[str]:
    if not isinstance(json_data, dict):
        raise ValueError("json_data는 딕셔너리여야 합니다.")
    
    pages = json_data.get("페이지별데이터", [])
    all_texts = []

    for page in pages:
        if "text" in page and page["text"].strip():
            all_texts.append(page["text"].strip())
        if "ocr_text" in page and page["ocr_text"].strip():
            all_texts.append(page["ocr_text"].strip())
        if "tables" in page:
            for table in page["tables"]:
                for row in table:
                    all_texts.append(" ".join(cell.strip() for cell in row if cell))
        if "images" in page and page["images"]:
            image_note = f"[페이지 {page['page']}] 이미지 포함 - 도식이나 다이어그램일 수 있으니 원문 문서를 참조하세요."
            all_texts.append(image_note)

    return all_texts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="JSON 파일 경로")
    parser.add_argument("--model_key", type=str, default="kr-sbert", help="임베딩 모델 키")
    args = parser.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    texts = extract_texts_from_json(json_data)
    print(f"\n총 추출된 텍스트 개수: {len(texts)}")
    for i, t in enumerate(texts, 1):
        print(f"[{i}] {t}")

    model = load_embedding_model(args.model_key)
    embeddings = model.encode(texts)
    print(f"\n임베딩 벡터 shape: {embeddings.shape if isinstance(embeddings, torch.Tensor) else len(embeddings)}")
