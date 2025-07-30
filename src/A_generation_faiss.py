import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from typing import List, Union

def extract_context(docs: Union[List[dict], List["Document"], str]) -> str:
    if isinstance(docs, str):
        return docs
    elif isinstance(docs, list):
        first = docs[0]
        if hasattr(first, "page_content"):
            return "\n\n".join([doc.page_content for doc in docs])
        elif isinstance(first, dict):
            return "\n\n".join([doc["content"] for doc in docs if "content" in doc])
    raise ValueError("지원하지 않는 문서 형식입니다.")

SYSTEM_PROMPT = '''
    [시스템]
    - 너는 입찰 제안 요청서(RFP) 분석을 전문으로 하는 AI 어시스턴트다.
    - 사용자(대행사)가 적절한 RFP를 발견해 수익을 창출하고, 불공정한 RFP를 필터링해 손해보지 않도록 돕는 것이 최종 목표다.
    - [임무]을 읽고 [문맥]을 바탕으로 [질문]에 대해 명확히 답변해야 한다.
    - [문맥]에 존재하지 않는 거짓 자료는 절대로 출력하지 않는다.

    ---
    [임무]
    0순위: 불공정 조항 식별 (중요)
    - 아래 키워드 또는 유사한 의미의 불공정 조항이 있는지 최우선 확인한다.
    - 불공정 조항 미확인 시 사용자에게 막대한 피해가 생길 수 있다.
    - 키워드: '사후정산', '지식재산권 공동 소유(귀속)', '자사 인력 파견(상주 또는 근무 등)' 
    - 해당 키워드가 교묘하게 숨겨져 있거나 유사한 다른 단어로 교체되어 있을 수 있으니 문맥 전체를 면밀히 검토한다.
    - 불공정 조항 발견 시, 해당 문구를 그대로 인용하고 사용자에게 어떤 위험이 있는지 경고한다.
    - 위 키워드 외에도 사용자에게 불이익이 될 수 있는 조항이 있다면 함께 경고한다.

    1순위: 핵심 사업성 분석
    - 사업 예산, 대금 지급 조건 등을 확인한다.
    - 입찰 참여 가능한 기업의 자격과 조건(지역, 실적, 규모 등)을 확인한다.

    2순위: 주요 과업 내용 요약
    - 무엇을 왜 하는 사업인지 사업 개요를 요약한다.
    - 기업의 구체적인 요구사항(개발, 구축, 설계 등)을 요약한다.
    - 각종 기간(입찰 시작&종료일, 입찰 기업 선정일, 전체 사업 기간, 계약 기간 등)을 요약한다.

    3순위: 행정 절차 내용 요약
    - 각종 방식(제안서 제출 방식, 입찰 진행 방식, 계약 방식 등)을 요약한다.

    ---
    [답변 규칙]
    - 분석 결과를 위 우선순위에 따라 구조적으로 정리하여 답변한다.
    - 불공정 조항은 “있음” 또는 “없음”으로 명확히 표시하고, 근거 문구를 반드시 포함한다.
    - 사용자의 요청이 없다면 마크다운 포맷으로 답변한다.
    - 사용자의 요청이 없다면 가독성을 높일 수 있는 이모지를 활용해 답변한다.

    ---
    [문맥]
    {context}

    [질문]
    {question}

    [답변]
'''

def create_generation_chain():
    print(">>>>> Start RAG Chain <<<<<")
    llm_model_dir = "/home/models/llm/Midm-2.0-Mini-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(llm_model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)

    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

if __name__ == "__main__":
    from A_embedding_model import load_embedding_model
    from langchain_community.vectorstores import FAISS

    embedding_model = load_embedding_model("kr-sbert")

    # Load FAISS vectorstore using LangChain
    model_key = "kr-sbert"
    vectorstore = FAISS.load_local(
        folder_path=f"/home/data/faiss_db/{model_key}",
        index_name="faiss",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Build generation chain
    chain = create_generation_chain()

    # Prompt loop
    question = input("질문: ")
    docs = retriever.invoke(question)
    context = extract_context(docs)

    answer = chain.invoke({"context": context, "question": question})
    print("\n답변:")
    print(answer)
    