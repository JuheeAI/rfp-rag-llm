from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os
import openai

load_dotenv()

model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

def cosine_similarity(doc1, doc2):
    emb1 = model.encode(doc1, convert_to_tensor=True)
    emb2 = model.encode(doc2, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb1, emb2).item()
    return round(score, 4)


def evaluate_file(input_path, output_path):
    df = pd.read_csv(input_path)

    scores_cos = []
    for _, row in df.iterrows():
        doc = str(row.get("retrieved_docs", ""))
        gen = str(row.get("generated", ""))
        scores_cos.append(cosine_similarity(doc, gen))

    df["cosine_score"] = scores_cos

    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def gpt_judge(question, retrieved, answer):
        prompt = f"""[질문]
{question}

[검색 문서]
{retrieved}

[모델 응답]
{answer}

위 내용을 기반으로 모델의 응답이 적절했는지 판단해줘. 1~5점 척도로 평가해줘. 단, 다음 기준을 따를 것:
- 질문에 대한 명확한 답변인지
- 검색된 문서를 잘 활용했는지
- 헷갈리거나 애매하지 않은지

출력 형식은 숫자 하나만 출력해줘 (예: 4)."""

        try:
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "당신은 평가자입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content.strip()
            try:
                score = int(content)
                return score
            except ValueError:
                return np.nan
        except Exception as e:
            print(f"[GPT 호출 실패] 질문: {question[:30]}... 에러: {e}")
            return "error"

    gpt_scores = []
    for _, row in df.iterrows():
        score = gpt_judge(str(row["question"]), str(row["retrieved_docs"]), str(row["generated"]))
        gpt_scores.append(score)

    df["gpt_score"] = gpt_scores

    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

def summarize_results():
    df_a = pd.read_csv("src/A_eval_scored.csv")
    df_b = pd.read_csv("src/B_eval_scored.csv")

    print("\nA 평가 요약:")
    print(df_a[["cosine_score", "gpt_score"]].describe())

    print("\nB 평가 요약:")
    print(df_b[["cosine_score", "gpt_score"]].describe())

    # Convert gpt_score to numeric, ignoring errors
    df_a["gpt_score"] = pd.to_numeric(df_a["gpt_score"], errors="coerce")
    df_b["gpt_score"] = pd.to_numeric(df_b["gpt_score"], errors="coerce")

    mean_a = df_a[["cosine_score", "gpt_score"]].mean()
    mean_b = df_b[["cosine_score", "gpt_score"]].mean()

    print("\n평균 점수 비교:")
    print(f"A 평균: cosine = {mean_a['cosine_score']:.4f}, gpt = {mean_a['gpt_score']:.4f}")
    print(f"B 평균: cosine = {mean_b['cosine_score']:.4f}, gpt = {mean_b['gpt_score']:.4f}")

    winner_cosine = "A" if mean_a["cosine_score"] > mean_b["cosine_score"] else "B"
    winner_gpt = "A" if mean_a["gpt_score"] > mean_b["gpt_score"] else "B"

    print(f"\n최종 결론:")
    print(f"- Cosine 유사도 기준 우수 모델: {winner_cosine}")
    print(f"- GPT 평가 기준 우수 모델: {winner_gpt}")

if __name__ == "__main__":
    evaluate_file("src/A_eval_simple.csv", "src/A_eval_scored.csv")
    evaluate_file("src/B_eval_simple.csv", "src/B_eval_scored.csv")
    summarize_results()
