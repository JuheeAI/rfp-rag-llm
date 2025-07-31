## 임베딩 및 인덱스 생성

단일 JSON 또는 디렉토리 내 모든 JSON을 임베딩하고 FAISS 인덱스로 저장합니다.

### 1. 임베딩 확인 (테스트용)

```bash
python src/A_embedding_model.py \
  --json_path /home/juhee/experiment/sample_jsons/example_smartcampus.json \
  --model_key kr-sbert
```

### 2. FAISS 인덱싱 (디렉토리 내 JSON 일괄 처리)

```bash
python src/A_indexing_faiss.py \
  --json_dir /home/juhee/experiment/sample_jsons \
  --model_key kr-sbert \
  --output_path /home/data/faiss_db
```

- 저장 경로 
  - `/home/data/faiss_db/kr-sbert/faiss.faiss`
  - `/home/data/faiss_db/kr-sbert/faiss.pkl`

---

## RAG 검색 및 생성

FAISS 인덱스를 기반으로 사용자의 질문에 답변을 생성합니다.

### 실행 명령어

```bash
python src/A_generation_faiss.py
```

- 설정은 `A_generation_faiss.py` 내부 코드에서 직접 지정합니다.

### 실행 예시

```bash
>>>>> Start RAG Chain <<<<<
질문: 스마트캠퍼스 시스템의 주요 기능은?
답변: 스마트캠퍼스 시스템은 ...
```

---

## 지원 Generation 모델

| 모델 키 | 모델 경로|
|---------|-----------------------------------|
| `midm-mini` | `/home/models/llm/Midm-2.0-Mini-Instruct` |
| 기타 | HuggingFace 호환 LLM이면 모두 지원 가능 (단, 시스템 요구사항 충족 필요) |

> 로컬 모델 디렉토리를 사용하는 경우 `--llm_model_dir` 인자에 경로를 명시하세요.

---

## 인덱스 저장 구조

- 루트: `/home/data/faiss_db/{model_key}/`
  - `faiss.faiss`: FAISS 벡터 인덱스
  - `faiss.pkl`: 메타데이터 파일

---

## 지원 Embedding 모델 키

- `kr-sbert` : `snunlp/KR-SBERT-V40K-klueNLI-augSTS`
- `ko-sbert` : `jhgan/ko-sbert-sts`,
- `kosimcse` : `BM-K/KoSimCSE-roberta-multitask`

---

## 주요 파일 설명

| 파일명 | 설명 |
|--------|------|
| `A_embedding_model.py` | 단일 JSON 문서 임베딩 |
| `A_indexing_faiss.py` | JSON 디렉토리 일괄 인덱싱 |
| `A_retriever_faiss.py` | FAISS 기반 벡터 검색기 정의 |
| `A_generation_faiss.py` | 검색 + 생성 RAG 파이프라인 실행 |

---