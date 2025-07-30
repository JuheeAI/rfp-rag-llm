# src/backend.py

import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional


##############################################################################
##### RAG 및 모델 체인 로딩 #####
from src.A_generation_faiss import create_generation_chain as create_chain_A
from src.B_generation import create_generation_chain as create_chain_B


##############################################################################
##### A 모델 리트리버 생성 #####
# 이 로직은 서버 시작 시 한 번만 실행되어 메모리에 상주함
from src.A_embedding_model import load_embedding_model 
from langchain_community.vectorstores import FAISS

print(">>>>> Loading Open Source Model (Retriever A) <<<<<")
embedding_model_A = load_embedding_model("kr-sbert")
vectorstore_A = FAISS.load_local(
    folder_path="/home/data/faiss_db/kr-sbert",
    index_name="faiss",
    embeddings=embedding_model_A,
    allow_dangerous_deserialization=True,
)
retriever_A = vectorstore_A.as_retriever(search_kwargs={"k": 3})
chain_A = create_chain_A()


##############################################################################
##### B 모델 리트리버 생성 #####
from langchain_openai import OpenAIEmbeddings

print(">>>>> Loading OpenAI Model (Retriever B) <<<<<")
# B_faiss_db를 만들 때 사용했던 임베딩 모델을 로드해야 함
# 보통 OpenAI 모델은 OpenAIEmbeddings를 사용
# 서버 환경변수에 OPENAI_API_KEY가 설정되어 있어야 함
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY 환경변수 없음")

embeddings_B = OpenAIEmbeddings() 

# B_faiss_db 경로에서 직접 FAISS 인덱스를 로드
vectorstore_B = FAISS.load_local(
    folder_path="/home/data/B_faiss_db/", # B 모델의 FAISS 경로를 명시
    embeddings=embeddings_B,
    index_name="index",
    allow_dangerous_deserialization=True
)
retriever_B = vectorstore_B.as_retriever(search_kwargs={"k": 3})


##############################################################################
##### 백엔드 서버 구축 #####
app = FastAPI()

# 프론트엔드 요청 데이터 형식 정의 
class QueryRequest(BaseModel):
    query: str
    model_source: str
    api_key: Optional[str] = None

# 모델과 요청 데이터를 이용해 RAG 체인을 생성하여 반환
@app.post("/get_answer")
async def get_answer_stream(request: QueryRequest):
    query = request.query
    
    # 프론트에서 받은 model_source에 따라 분기 처리
    # A 모델 응답 반환
    if request.model_source == "Open Source":
        print("INFO: User Request Using Open Source Model")
        # chain_A는 이미 리트리버가 포함된 형태로 생성됨
        # A_generation_faiss.py 구조에 따라 context를 직접 넘겨줘야 함
        docs = retriever_A.invoke(query)
        # extract_context 함수를 A_generation_faiss에서 가져와야 함
        from src.A_generation_faiss import extract_context 
        context = extract_context(docs)
        
        # 스트리밍 응답 반환
        return StreamingResponse(
            (chunk for chunk in chain_A.stream({"context": context, "question": query})),
            media_type="text/event-stream"
        )
        
    # B 모델 응답 반환
    elif request.model_source == "OpenAI" and request.api_key:
        print("INFO: User Request Using OpenAI Model")
        # OpenAI 체인은 요청 시마다 API 키를 받아 새로 생성
        chain_B = create_chain_B(
            retriever=retriever_B,
            api_key=request.api_key,
            model_name="gpt-4.1-mini"
        )
        # 스트리밍 응답 반환
        return StreamingResponse(
            (chunk for chunk in chain_B.stream(query)),
            media_type="text/event-stream"
        )
        
    else:
        # 부적절한 요청에 대한 에러 처리
        async def error_stream():
            yield "Error: Invalid model source or missing API key."
        return StreamingResponse(error_stream(), media_type="text/event-stream", status_code=400)















##########################################
# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel
# from src.A_generation import create_generation_chain
# from src.B_generation import create_generation_chain_openai
##########################################
# chain = create_generation_chain()
# app = FastAPI()

# class QueryRequest(BaseModel):
#     query: str

# @app.post("/get_answer")
# async def get_answer_stream(request: QueryRequest):
#     answer = StreamingResponse(
#         (chunk for chunk in chain.stream(request.query)),
#         media_type="text/event-stream"
#     )
#     return answer



# main.py에서 실행하므로 주석처리
# import uvicorn
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=9000)