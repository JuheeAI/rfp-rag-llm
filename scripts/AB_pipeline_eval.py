# make_eval_files.py
import os, json, pandas as pd
from typing import List
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ----- A 파이프라인 (FAISS 리트리버 + 수동 컨텍스트 합성 RAG 체인) -----
from A_retriever_faiss import load_langchain_retriever as A_load_retriever
from A_embedding_model import load_embedding_model as A_load_embedding
from A_generation_faiss import rag_chain as A_rag_chain, extract_context as A_extract_context

# ----- B 파이프라인 (내장 컨텍스트 추출 체인) -----
from B_retriever import get_retriever as B_get_retriever
from B_generation import create_generation_chain as B_create_chain, extract_context as B_extract_context

QUESTIONS_CSV = "/home/juhee/rfp-rag-llm/data/retrieval_questions.csv"
A_DB = "/home/data/A_faiss_db"
B_DB = "/home/data/B_faiss_db"
OUTDIR = "./eval_out"
TOP_K = 5

os.makedirs(OUTDIR, exist_ok=True)

def _id_from_meta_A(meta: dict) -> str:
    # A 인덱싱 시 저장했던 키들에 맞춰 최대한 식별자 구성
    doc = meta.get("doc_id") or meta.get("doc_name") or meta.get("source") or "unknown"
    page = meta.get("page", -1)
    chunk = meta.get("chunk_type", "") or meta.get("title", "")
    return f"{doc}:{page}:{chunk}"

def _id_from_meta_B(meta: dict) -> str:
    # B는 메타에 'source'(파일명), '공고번호', '사업명'이 있음
    src = meta.get("source", "unknown")
    bid = meta.get("공고번호", "")
    name = meta.get("사업명", "")
    return f"{src}:{bid}:{name}"

def retrieve_A(questions: List[str]) -> pd.DataFrame:
    emb = A_load_embedding("kr-sbert")
    retriever = A_load_retriever(base_path=A_DB, model_key="kr-sbert", embedding_model=emb)
    rows = []
    for q in tqdm(questions, desc="[A] retrieval"):
        docs = retriever.invoke(q)[:TOP_K]
        rows.append({
            "question": q,
            "retrieved_docs": " || ".join([d.page_content for d in docs]),
            "retrieved_ids": json.dumps([_id_from_meta_A(d.metadata) for d in docs], ensure_ascii=False),
            "pipeline": "A",
            "top_k": TOP_K
        })
    return pd.DataFrame(rows)

def retrieve_B(questions: List[str]) -> pd.DataFrame:
    retriever = B_get_retriever(
        documents_path="/home/juhee/data_json/all_data",
        index_path=B_DB,
        reuse_index=True,
        limit_files=None
    )
    rows = []
    for q in tqdm(questions, desc="[B] retrieval"):
        docs = retriever.invoke(q)[:TOP_K]
        rows.append({
            "question": q,
            "retrieved_docs": " || ".join([d.page_content for d in docs]),
            "retrieved_ids": json.dumps([_id_from_meta_B(d.metadata) for d in docs], ensure_ascii=False),
            "pipeline": "B",
            "top_k": TOP_K
        })
    return pd.DataFrame(rows)

def generate_A(df_ret: pd.DataFrame) -> pd.DataFrame:
    # A_generation_faiss.rag_chain 은 {"context": str, "question": str}를 기대
    gens = []
    for _, r in tqdm(df_ret.iterrows(), total=len(df_ret), desc="[A] generation"):
        q = r["question"]
        ctx = r["retrieved_docs"]
        out = A_rag_chain.invoke({"context": ctx, "question": q})
        gens.append({
            "question": q,
            "retrieved_docs": ctx,
            "retrieved_ids": r["retrieved_ids"],
            "generated": out,
            "pipeline": "A",
            "top_k": r["top_k"]
        })
    return pd.DataFrame(gens)

def generate_B(df_ret: pd.DataFrame, model_name: str = "gpt-4.1-mini", api_key: str = None) -> pd.DataFrame:
    retriever = B_get_retriever(
        documents_path="/home/juhee/data_json/all_data",
        index_path=B_DB,
        reuse_index=True,
        limit_files=None
    )
    chain = B_create_chain(retriever=retriever, api_key=api_key or os.getenv("OPENAI_API_KEY"), model_name=model_name)
    gens = []
    for _, r in tqdm(df_ret.iterrows(), total=len(df_ret), desc="[B] generation"):
        q = r["question"]
        out = chain.invoke(q)   
        gens.append({
            "question": q,
            "retrieved_docs": r["retrieved_docs"],   
            "retrieved_ids": r["retrieved_ids"],
            "generated": out,
            "pipeline": "B",
            "top_k": r["top_k"]
        })
    return pd.DataFrame(gens)

def to_langsmith_jsonl(df_gen: pd.DataFrame, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for _, r in df_gen.iterrows():
            obj = {
                "inputs": {
                    "input": r["question"],
                    "context": r["retrieved_docs"]  # 컨텍스트 정합성/환각 평가용
                },
                "outputs": {"output": r["generated"]},
                "retrieved_ids": json.loads(r["retrieved_ids"]),
                "pipeline": r["pipeline"]
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    dfq = pd.read_csv(QUESTIONS_CSV)
    questions = dfq["question"].astype(str).tolist()

    dfA_ret = retrieve_A(questions)
    dfB_ret = retrieve_B(questions)
    dfA_ret.to_csv(os.path.join(OUTDIR, "A_retrieval_eval.csv"), index=False, encoding="utf-8-sig")
    dfB_ret.to_csv(os.path.join(OUTDIR, "B_retrieval_eval.csv"), index=False, encoding="utf-8-sig")

    dfA_gen = generate_A(dfA_ret)
    dfB_gen = generate_B(dfB_ret, model_name="gpt-4.1-nano") 
    dfA_gen.to_csv(os.path.join(OUTDIR, "A_generation_eval.csv"), index=False, encoding="utf-8-sig")
    dfB_gen.to_csv(os.path.join(OUTDIR, "B_generation_eval.csv"), index=False, encoding="utf-8-sig")

    to_langsmith_jsonl(dfA_gen, os.path.join(OUTDIR, "langsmith_eval_A.jsonl"))
    to_langsmith_jsonl(dfB_gen, os.path.join(OUTDIR, "langsmith_eval_B.jsonl"))

if __name__ == "__main__":
    main()
