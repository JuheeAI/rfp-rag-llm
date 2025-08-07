import pandas as pd
import json
import os
from A_retriever_faiss import load_langchain_retriever
from A_embedding_model import load_embedding_model
from B_retriever import get_retriever as get_retriever_B

def convert_csv_to_langsmith_jsonl(csv_path, jsonl_path):
    if not os.path.exists(csv_path):
        print(f"[오류] 파일이 존재하지 않음: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    a_embedding_model = load_embedding_model("kr-sbert")
    a_retriever = load_langchain_retriever(
        base_path="/home/data/A_faiss_db",
        model_key="kr-sbert",
        embedding_model=a_embedding_model
    )
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            input_text = str(row["question"]) if "question" in df.columns else str(row["query"])
            output_text = str(row["generated"]) if "generated" in df.columns else str(row["generation"])
            reference = ""
            if "reference" in df.columns and pd.notna(row["reference"]):
                reference = str(row["reference"])

            retrieved_docs = get_retrieved_docs_A(input_text, a_retriever) if "A_generation" in csv_path else get_retrieved_docs_B(input_text)

            json_obj = {
                "inputs": {"input": input_text},
                "outputs": {"output": output_text},
                "retrieved_docs": retrieved_docs,
                "reference": reference
            }
            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

    print(f"[완료] LangSmith 평가 파일 저장됨: {jsonl_path}")

def convert_csv_to_eval_csv(csv_path, output_csv_path):
    if not os.path.exists(csv_path):
        print(f"[오류] 파일이 존재하지 않음: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if ("question" not in df.columns and "query" not in df.columns) or "generated" not in df.columns:
        print(f"[오류] 필요한 컬럼이 없습니다: {csv_path}")
        return

    retrieved_col = "retrieved_docs" if "retrieved_docs" in df.columns else None
    if not retrieved_col:
        print(f"[경고] 검색 문서 컬럼이 없습니다: {csv_path}")
        df["retrieved_docs"] = ""
    else:
        df["retrieved_docs"] = df[retrieved_col]

    df["question"] = df["question"] if "question" in df.columns else df["query"]
    df_simple = df[["question", "retrieved_docs", "generated"]]
    df_simple.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"[완료] 평가용 CSV 저장됨: {output_csv_path}")

def convert_all_to_langsmith_format():
    convert_csv_to_langsmith_jsonl(
        "/home/juhee/rfp-rag-llm/src/A_generation_eval.csv",
        "/home/juhee/rfp-rag-llm/src/langsmith_eval_A.jsonl"
    )
    convert_csv_to_langsmith_jsonl(
        "/home/juhee/rfp-rag-llm/src/B_generation_eval.csv",
        "/home/juhee/rfp-rag-llm/src/langsmith_eval_B.jsonl"
    )
    save_retrieval_results_for_eval(
        "/home/juhee/rfp-rag-llm/src/retrieval_questions.csv",
        "/home/juhee/rfp-rag-llm/src/A_eval_simple.csv",
        mode="A"
    )
    save_retrieval_results_for_eval(
        "/home/juhee/rfp-rag-llm/src/retrieval_questions.csv",
        "/home/juhee/rfp-rag-llm/src/B_eval_simple.csv",
        mode="B"
    )

def get_retrieved_docs_A(query, retriever):
    docs = retriever.invoke(query)
    return " || ".join([doc.page_content for doc in docs])

def get_retrieved_docs_B(query):
    if not hasattr(get_retrieved_docs_B, "retriever"):
        get_retrieved_docs_B.retriever = get_retriever_B(
            documents_path="/home/juhee/data_json/all_data",
            index_path="/home/data/B_faiss_db",
            reuse_index=True,
            limit_files=None
        )
    docs = get_retrieved_docs_B.retriever.invoke(query)
    return " || ".join([doc.page_content for doc in docs])


# 새 함수: 평가용 검색결과 CSV 저장
def save_retrieval_results_for_eval(csv_input_path, csv_output_path, mode="A"):
    if not os.path.exists(csv_input_path):
        print(f"[오류] 입력 파일이 존재하지 않음: {csv_input_path}")
        return

    df = pd.read_csv(csv_input_path)
    results = []

    if mode == "A":
        embedding_model = load_embedding_model("kr-sbert")
        retriever = load_langchain_retriever(
            base_path="/home/data/A_faiss_db",
            model_key="kr-sbert",
            embedding_model=embedding_model
        )
    elif mode == "B":
        retriever = get_retriever_B(
            documents_path="/home/juhee/data_json/all_data",
            index_path="/home/data/B_faiss_db",
            reuse_index=True,
            limit_files=None
        )
    else:
        print(f"[오류] 잘못된 모드: {mode}")
        return

    for _, row in df.iterrows():
        if "question" in df.columns:
            query = str(row["question"])
        elif "query" in df.columns:
            query = str(row["query"])
        else:
            print(f"[경고] 질문 컬럼이 없음: {row}")
            continue

        generated = str(row["generated"]) if "generated" in df.columns else ""
        retrieved_docs = retriever.invoke(query)
        docs_text = " || ".join([doc.page_content for doc in retrieved_docs])
        results.append({
            "question": query,
            "retrieved_docs": docs_text,
            "generated": generated
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"[완료] 평가용 검색결과 CSV 저장됨: {csv_output_path}")

if __name__ == "__main__":
    convert_all_to_langsmith_format()