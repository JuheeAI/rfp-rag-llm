# HuggingFace 기반 RAG
## 임베딩 및 인덱스 생성

단일 JSON 또는 디렉토리 내 모든 JSON을 임베딩하고 FAISS 인덱스로 저장합니다.

### 실행 명령어

```bash
python A_indexing_faiss.py   \
    --json_dir /home/data/preprocess/json \
    --model_key kr-sbert   \
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

# OpenAI API를 기반 RAG
## 임베딩 및 인덱스 생성

OpenAIEmbeddings를 이용해 JSON 문서를 벡터화하고 FAISS 인덱스를 저장합니다.

**실행 명령어**
```bash
python -m src.B_generation
```
* .env에 OPENAI_API_KEY가 설정되어 있어야 합니다.
* 최초 실행 시 약 1~2분간 임베딩 및 인덱스 생성 시간이 소요됩니다.

## 질의 기반 검색 및 응답 생성
실행 후 사용자 질문을 입력하면, FAISS 검색기를 통해 관련 문서를 검색한 후 GPT 모델이 응답을 생성합니다.

**예시**
```bash
모델 선택 (nano/mini): mini
질문: 부산 국제 영화제 관련 불공정 조항이 있나요?
답변: ## 0순위: 불공정 조항 식별 🔍

- **불공정 조항 여부:** 없음  
- **근거:**  
  - 문맥 내에서
...
```

## 지원 모델

| 분류       | 모델 키      | 설명                             |
|------------|---------------|----------------------------------|
| 임베딩     | openai-embed  | text-embedding-3-small (OpenAI) |
| 생성 (API) | gpt-4.1-nano  | 초경량 GPT-4.1 기반 모델 (OpenAI) |
| 생성 (API) | gpt-4.1-mini  | GPT-4.1 기반 경량 모델          |

## 주요 파일 설명

| 파일명            | 설명                                      |
|-------------------|-------------------------------------------|
| `B_retriever.py`  | JSON 로딩, 임베딩, FAISS 인덱스 생성 및 로딩 |
| `B_generation.py` | LLM 체인 구성 및 사용자 질의 응답 실행       |