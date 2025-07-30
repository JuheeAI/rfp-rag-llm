import numpy as np
import torch, json, kss, argparse
from typing import List, Union
from transformers import AutoTokenizer
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# 모델 목록
SUPPORTED_MODELS = {
    "kr-sbert": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    "ko-sbert": "jhgan/ko-sbert-sts",
    "kosimcse": "BM-K/KoSimCSE-roberta-multitask"
}

# 임베딩 벡터 인코딩 함수
def load_embedding_model(model_key: str) -> Embeddings:
    if model_key not in SUPPORTED_MODELS:
        raise ValueError(f"지원하지 않는 모델입니다: {model_key}")
    model_name = SUPPORTED_MODELS[model_key]
    return HuggingFaceEmbeddings(model_name=model_name)

# 텍스트 인코딩 함수
def extract_texts_from_json(json_data: dict) -> List[str]:
    if not isinstance(json_data, dict):
        raise ValueError("json_data는 딕셔너리여야 합니다.")
    
    pages = json_data.get("페이지별데이터", [])
    raw_texts = []

    for page in pages:
        if "text" in page and page["text"].strip():
            raw_texts.append(page["text"].strip())
        if "ocr_text" in page and page["ocr_text"].strip():
            raw_texts.append(page["ocr_text"].strip())
        if "tables" in page:
            for table in page["tables"]:
                for row in table:
                    row_text = " ".join(cell.strip() for cell in row if cell)
                    if row_text:
                        raw_texts.append("[표] " + row_text)
        if "images" in page and page["images"]:
            image_note = f"[이미지] 페이지 {page.get('page', '?')}에 이미지 포함"
            raw_texts.append(image_note)

    # 전체 문단 이어 붙이기
    merged_text = "\n".join(raw_texts)

    # 토큰 기반 청크 생성
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
    tokens = tokenizer(merged_text, return_offsets_mapping=True, return_attention_mask=False, return_token_type_ids=False)
    offset_mapping = tokens["offset_mapping"]
    input_ids = tokens["input_ids"]

    max_tokens = 512
    stride = 64

    # 유효한 토큰 위치만 필터링
    valid_offsets = [(i, off) for i, off in enumerate(offset_mapping) if off != (0, 0)]
    if not valid_offsets:
        return []

    max_offset_idx = valid_offsets[-1][0]

    if len(input_ids) <= max_tokens:
        chunks = [merged_text.strip()]
        return chunks
    else:
        chunks = []
        for start_idx in range(0, max_offset_idx, stride):
            end_idx = start_idx + max_tokens
            start_char = offset_mapping[start_idx][0]
            end_char = offset_mapping[min(end_idx, len(offset_mapping) - 1)][1]
            chunk_text = merged_text[start_char:end_char].strip()
            if len(chunks) == 0 and start_idx == 0 and len(merged_text.strip()) <= max_tokens:
                continue  
            if chunk_text:
                chunks.append(chunk_text)

    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="JSON 파일 경로")
    parser.add_argument("--model_key", type=str, default="kr-sbert", help="임베딩 모델 키")
    args = parser.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    texts = extract_texts_from_json(json_data)

    model = load_embedding_model(args.model_key)
    embeddings = model.embed_documents(texts)
    print(np.array(embeddings).shape)
    pass