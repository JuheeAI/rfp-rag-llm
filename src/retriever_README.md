## 1. Overview
Retriever는 FAISS 기반 구성 파일 (B_embedding_model.py, B_indexing.py, B_retriever.py)에 대한 설명과 실행 방법을 안내합니다.
총 100개의 문서에 대해 벡터 인덱스를 생성하고 유사한 청크를 거리 기반으로 검색해 generation 시스템의 입력으로 활용할 수 있도록 합니다.

## 2. 파일 구성
| 파일명               | 역할                                                  |
|---------------------|-------------------------------------------------------|
| B_embedding_model.py | 텍스트 추출 + 문장 임베딩 생성                        |
| B_indexing.py        | JSON 문서 전체를 벡터화하여 `.index` 및 `_meta.json` 생성 |
| B_retriever.py       | 질의 입력 시 Top-k 유사 청크 검색 결과 반환           |

## 3. 실행 방법

```bash
# (1) indexing - 모든 문서 임베딩 및 저장
python B_indexing.py \
  --json_dir /home/juhee/experiment/sample_jsons \
  --model_key kr-sbert \
  --save_path /home/juhee/experiment/outputs/smartcampus_kr-sbert_20250729
```
### → 다음 파일들이 생성됩니다 (파일 이름과 폴더 구조는 예시입니다.)
### - 벡터 인덱스 파일 예시: smartcampus_kr-sbert_20250729.index
### - 메타 정보 JSON: smartcampus_kr-sbert_20250729_meta.json

```bash
# (2) retriever - 질의어에 대해 Top-k 유사 청크 반환 (hybrid 방식)
python B_retriever.py \
  --index_path /home/juhee/experiment/outputs/smartcampus_kr-sbert_20250729.index \
  --meta_path /home/juhee/experiment/outputs/smartcampus_kr-sbert_20250729_meta.json \
  --tfidf_path /home/juhee/experiment/outputs/smartcampus_kr-sbert_20250729_tfidf.pkl \
  --query "이미지가 포함된 문서가 있나요?" \
  --model_key kr-sbert \
  --mode hybrid
```

## 4. 모델 키 목록
| 모델 키     | 모델 설명                                    |
|-------------|-----------------------------------------------|
| kr-sbert    | snunlp/KR-SBERT-V40K-klueNLI-augSTS           |
| ko-sbert    | jhgan/ko-sbert-sts                            |
| kosimcse    | BM-K/KoSimCSE-roberta-multitask              |

## 5. 출력 예시

[사용자 질의] 위약금 조항이 있나요?

[검색 결과]

[Index: 6, 거리: 476.9672]
[파일명: 스마트캠퍼스제안서.pdf]
[사업명: 스마트캠퍼스 구축]
본문: [페이지 1] 납기 미준수 시 위약금 조항이 포함되어 있습니다.