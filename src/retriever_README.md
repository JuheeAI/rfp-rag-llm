# Retriever 사용 가이드 (ChromaDB / FAISS)

이 문서는 RAG 시스템의 Retriever 구성 요소로서 ChromaDB 또는 FAISS를 사용하여 생성(GPT 등) 모듈과 연동하는 방법을 설명합니다.

---

## 1. ChromaDB 기반 Retriever → Generation 연동 예시

```python
from chromadb import PersistentClient
from chromadb.config import Settings
from A_embedding_model import load_embedding_model
from A_retriever_chroma import query_documents

# (1) ChromaDB 클라이언트 로드
client = PersistentClient(
    path="/home/data/chromadb/kr-sbert",
    settings=Settings()
)

# (2) 임베딩 모델 로드
model = load_embedding_model("kr-sbert")

# (3) 관련 청크 검색
results = query_documents(
    chroma_client=client,
    collection_name="rfp_chunks",
    query_text="스마트캠퍼스 구축 계획은 어떻게 되나요?",
    model=model,
    top_k=3
)

# (4) Generation 모듈 연동
retrieved_texts = [item['text'] for item in results]
prompt = "다음 정보를 바탕으로 질문에 답변하세요:\n" + "\n".join(retrieved_texts)
```

---

## 2. FAISS 기반 Retriever → Generation 연동 예시

```python
import pickle
import faiss
import numpy as np
from A_embedding_model import load_embedding_model
from A_retriever_faiss import embed_query, search_index

# (1) 임베딩 모델 로드
model = load_embedding_model("kr-sbert")

# (2) FAISS 인덱스 및 메타데이터 로드
index = faiss.read_index("/home/data/faiss_db/faiss.index")
with open("/home/data/faiss_db/faiss_meta.pkl", "rb") as f:
    metadata = pickle.load(f)

# (3) 질의 임베딩 후 유사 청크 검색
query = "스마트캠퍼스 구축 계획은 어떻게 되나요?"
query_vec = embed_query(model, query)
top_k = 3
results = search_index(index, query_vec, metadata, top_k)

# (4) Generation 모듈 연동
retrieved_texts = [item["text"] for item in results]
prompt = "다음 정보를 바탕으로 질문에 답변하세요:\n" + "\n".join(retrieved_texts)
```

---

## 3. 스크립트 실행 예시 (__main__ 블록이 있을 경우)

### A_retriever_chroma.py 실행

```bash
python A_retriever_chroma.py \
  --query "스마트캠퍼스 구축 계획은 어떻게 되나요?" \
  --model_key kr-sbert \
  --persist_path /home/data/chromadb/kr-sbert \
  --collection_name rfp_chunks \
  --top_k 3
```

### A_retriever_faiss.py 실행

```bash
python A_retriever_faiss.py \
  --query "스마트캠퍼스 구축 계획은 어떻게 되나요?" \
  --model_key kr-sbert \
  --index_path /home/data/faiss_db/faiss.index \
  --meta_path /home/data/faiss_db/faiss_meta.pkl \
  --top_k 3
```

---

## 4. 최적화 및 구현 주요 사항

Retriever 구성은 실제 서비스 수준의 성능과 유지보수 고려
고급 설정과 개선 사항을 반영

### ChromaDB 최적화

### FAISS 최적화
- **IVF+PQ 인덱스 구조**를 사용하여 대규모 문서셋에 대비한 검색 속도 향상 구조 설계
- 단일 문서만 있을 경우 자동으로 **IndexFlatL2**로 fallback 처리
- 검색 유사도 score와 메타정보를 함께 반환하여 이후 filtering 또는 scoring 로직 유연하게 확장 가능
- FAISS 인덱스는 `faiss.index`, 메타데이터는 `faiss_meta.pkl`로 분리 저장 → 인덱스 갱신/재훈련이 용이함

---