## 1. Overview
이 문서는 ChromaDB 기반 Retriever 구성 요소와 실행 방법을 안내합니다. 아래 파일들을 사용하여 JSON 문서에서 텍스트를 추출하고, 임베딩을 생성하고, 질의에 따라 관련 청크를 검색할 수 있습니다.

## 2. 파일 구성
| 파일명                 | 역할                                                   |
|-----------------------|--------------------------------------------------------|
| A_embedding_model.py   | JSON 문서 텍스트 추출 및 임베딩 모델 로드             |
| A_indexing_chroma.py   | 전체 문서 임베딩 및 ChromaDB에 저장                    |
| A_retriever_chroma.py  | 질의어에 대해 Top-k 유사 청크를 ChromaDB에서 검색     |

## 3. 실행 방법

```bash
# (1) indexing - 문서들을 ChromaDB에 저장
python A_indexing_chroma.py \
  --json_dir /home/juhee/experiment/sample_jsons \
  --model_key kr-sbert \
  --persist_path /home/data/chromadb/kr-sbert \
  --collection_name rfp_chunks
```

```bash
# (2) retriever - ChromaDB에서 관련 청크 검색
python A_retriever_chroma.py \
  --query "스마트캠퍼스 구축 계획은 어떻게 되나요?" \
  --model_key kr-sbert \
  --persist_path /home/data/chromadb/kr-sbert \
  --collection_name rfp_chunks \
  --top_k 3
```

## 4. 모델 키 목록
| 모델 키     | 모델 설명                                    |
|-------------|-----------------------------------------------|
| kr-sbert    | snunlp/KR-SBERT-V40K-klueNLI-augSTS           |
| ko-sbert    | jhgan/ko-sbert-sts                            |
| kosimcse    | BM-K/KoSimCSE-roberta-multitask              |

## 5. 출력 예시

[사용자 질의] 스마트캠퍼스 구축 계획은 어떻게 되나요?

[검색 결과]

[결과 1]
▶ 관련 문장: 이 문서는 스마트캠퍼스 구축 제안서입니다. AI 기반 교육 시스템 도입...
▶ 메타정보: {'공고번호': '12345', '사업명': '스마트캠퍼스 구축', '파일명': 'example_smartcampus.json', '청크번호': 0}

## 6. Generation 연동 방법

retriever의 출력 결과는 generation 모듈에서 답변 생성을 위한 입력으로 사용됩니다. 아래는 예시 코드입니다.

```python
from chromadb import PersistentClient
from chromadb.config import Settings
from A_embedding_model import load_embedding_model
from A_retriever_chroma import query_documents

# ChromaDB 클라이언트 로드
client = PersistentClient(
    path="/home/data/chromadb/kr-sbert",
    settings=Settings()
)

# 임베딩 모델 로드
model = load_embedding_model("kr-sbert")

# 질의어를 기반으로 top-k 관련 문서 검색
results = query_documents(
    chroma_client=client,
    collection_name="rfp_chunks",
    query_text="스마트캠퍼스 구축 계획은 어떻게 되나요?",
    model=model,
    top_k=3
)

# generation에 전달
retrieved_texts = [item['text'] for item in results]
prompt = "다음 정보를 바탕으로 질문에 답변하세요:\n" + "\n".join(retrieved_texts)
```

- generation 모듈에서는 `prompt`를 LLM의 입력으로 사용하여 자연어 응답을 생성합니다.