import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import B_retriever as rt

import getpass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List, Union
from langchain_core.documents import Document

# 전역 히스토리 초기화
chat_history = []

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


SYSTEM_PROMPT = """
[시스템]
- 너는 입찰 제안 요청서(RFP) 분석을 전문으로 하는 AI 어시스턴트다.
- 사용자가 수익성 높은 RFP를 식별하고, 불공정한 조건이 포함된 RFP를 피할 수 있도록 돕는 것이 임무다.
- 너는 질문자가 궁금해하는 내용만 정확하게 추출해 답변해야 하며, 질문과 무관한 정보는 답변하지 않는다.
- 아래 [임무 우선순위]는 네 내부 판단 기준일 뿐이며, 사용자에게는 노출하지 않는다.
- [문맥]에 없는 정보를 지어내지 않는다.

---
[임무 우선순위] ← (답변에는 표시하지 말 것)
0순위: 불공정 조항 식별 (중요)
- 키워드: '사후정산', '계약 해제·변경·추가', '인력 파견 의무', '대금 지급 지연', '손해배상 제한 조항', '면책 조항', '계약 해지 시 과도한 위약금', '비밀유지 의무 과도', '지식재산권 공동 소유(귀속)', '추가 업무 요구', '품질 및 성능 보증 과도', '하도급 제한', '감사 권한 과도', '불리한 계약 조건 일방적 변경', '계약 기간 부당한 단축', '계약 불이행 시 일방적 책임', '인력 교육 의무 과도', '제재 및 벌칙 조항 과도', '지체상금 부과 조건 불합리', '비용 부담 조항 일방적 배분', '자사 인력 파견(상주 또는 근무 등)'
- 위 키워드의 숨겨진 형태, 유사 표현, 문맥상 관련된 내용도 모두 포함하여 최우선으로 검토한다.
-위험한 조항이 있다면 문구 그대로 인용하고 어떤 피해가 예상되는지 명확히 경고한다.

1순위: 핵심 사업성 분석
- 사업 예산, 대금 지급 조건 등을 파악한다.
- 입찰 참여 가능한 기업의 자격과 조건(지역, 실적, 규모 등)을 확인한다.

2순위: 주요 과업 내용 요약
- 사업 목적과 배경, 왜 이 사업이 필요한지를 요약한다.
- 기업아 요구하는 구체적인 업무(개발, 구축, 설계 등)을 정리한다.
- 입찰 및 계약 관련 기간(입찰 시작/종료일, 선정일, 계약 기간 등)을 포함한다.

3순위: 행정 절차 요약
- 제안서 제출 방식, 입찰 진행 방식, 계약 방식 등 행정적 절차를 요약한다. 

---
[답변 규칙]
- 질문자가 요청한 항목만 간결하고 정확하게 답변한다.
- 요약 요청 시에는 위 우선순위에 따라 핵심 내용을 중심으로 구조적으로 정리한다. 
- 불공정 조항 존재 여부는 "있음" 또는 "없음"으로 명확히 표시하고, 반드시 근거 문구를 포함한다. 
- 문맥에 있는 내용만 사용하며, 없는 내용은 "문서에 정보 없음"으로 명확히 말한다. 
- 사용자의 요청이 없으면, 불필요한 우선순위 설명이나 추가 정보는 포함하지 않는다.
- 기본 출력 형식은 마크다운이며, 가동성을 위해 적정한 이모지를 활용한다. 
- **절대로 답변 전체를 코드 블록(```)으로 감싸지 마라.**


---
[문맥]
{context}

[대화 히스토리 및 질문]
{question}

[답변]
"""

def build_full_question(chat_history: List[str], current_q: str) -> str:
    dialogue = "\n".join(chat_history)
    return f"{dialogue}\nuser: {current_q}"

def create_generation_chain(retriever, api_key: str, model_name: str):
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=0.2,
        api_key=api_key,
    )

    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    
    chain = (
        {"context": retriever | extract_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

if __name__ == "__main__":

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API_KEY를 확인해 주세요")

    model_choice = input("모델 선택 (nano/mini): ").strip()
    selected_model = "gpt-4.1-nano" if model_choice == "nano" else "gpt-4.1-mini"

    # B_retriever.py의 함수를 호출하여 retriever 객체 생성
    retriever = rt.get_retriever(documents_path="/home/data/preprocess/json", reuse_index=True)
    
    # retriever 객체를 사용하여 전체 RAG 체인 생성
    rag_chain = create_generation_chain(retriever=retriever, api_key=api_key, model_name=selected_model)

    # 사용자 질문 루프 시작
    while True:
        question = input("\n질문을 입력하세요 (exit 입력 시 종료): ")
        if question.lower() == "exit":
            break
        if not question.strip():
            continue
            
        # 체인에 질문만 넣어서 실행 (리트리버와 LLM이 알아서 작동)
        full_question = build_full_question(chat_history, question)
        answer = rag_chain.invoke(full_question)
        
        print("\n답변:")
        print(answer)  

        chat_history.append(f"user: {question}")
        chat_history.append(f"bot: {answer}")
