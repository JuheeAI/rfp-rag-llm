# src/backend.py

import os
import json
import langchain
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_community.vectorstores import FAISS


########################################################################
# RAG 및 모델 체인 로딩
from src.A_generation import create_generation_chain as create_chain_A
from src.B_generation import create_generation_chain as create_chain_B
from src.B_retriever import get_retriever
from src.A_embedding import load_embedding_model


########################################################################
# 로그 출력 여부 제어 
langchain.debug = True  # 디버그 모드
LOG_QUERY = False  # 사용자 질문 출력
LOG_CONTEXT_IN_TERMINAL = False # 검색 문서 출력
LOG_ANSWER = False  # 모델 답변 출력
DELIMITER = "_|||_" # 데이터와 답변 스트림을 구분할 문자열

if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")


########################################################################
# 모델 로드 함수
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델 A (Open Source) 로딩
    print(">>>>> Loading Open Source Model (Retriever A) <<<<<")
    embedding_model_A = load_embedding_model("kr-sbert")
    retriever_A = FAISS.load_local(
        folder_path="/home/data/A_faiss_db/kr-sbert",
        index_name="index",
        embeddings=embedding_model_A,
        allow_dangerous_deserialization=True,
    ).as_retriever(search_kwargs={"k": 7})
    app.state.chain_A = create_chain_A(retriever=retriever_A)
    
    # 모델 B (OpenAI) 로딩
    print(">>>>> Loading OpenAI Model (Retriever B) <<<<<")
    retriever_B = get_retriever(
        documents_path="/home/data/preprocess/json/", 
        index_path="/home/data/B_faiss_db/", 
        reuse_index=True, 
        k=7, 
        limit_files=None,
    )
    api_key = os.getenv("OPENAI_API_KEY")
    app.state.chain_B = create_chain_B(retriever=retriever_B, api_key=api_key, model_name="gpt-4.1-mini")  
    yield


########################################################################
# FastAPI 앱 객체 생성
app = FastAPI(lifespan=lifespan)

# Request 모델 생성
class QueryRequest(BaseModel):
    query: str
    model_source: str
    api_key: Optional[str] = None


########################################################################
# 콜백 핸들러 클래스
class MyCustomHandler(BaseCallbackHandler):
    def __init__(self):
        self.final_answer = ""
        self.retrieved_docs = [] # 검색된 문서를 저장할 리스트

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        query = inputs.get("question") or inputs.get("input")
        if LOG_QUERY and isinstance(query, str):
            print(f"\n********** [ User Query ] **********\n{query}\n{'.' * 50}")

    def on_retriever_end(self, documents: List[Document], **kwargs: Any) -> None:
        self.retrieved_docs = documents
        if LOG_CONTEXT_IN_TERMINAL:
            print(f"\n********** [ Retrieved Context ] **********\n")
            for doc in documents:
                print(f"📄 {doc.metadata.get('source', 'Unknown Source')}\n{doc.page_content}\n{'-' * 20}")
            print("." * 50)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        self.final_answer += token

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        if LOG_ANSWER and self.final_answer:
            print(f"\n********** [ Final Answer ] **********\n{self.final_answer}\n{'.' * 50}")


########################################################################
# API 라우트 정의
@app.post("/get_answer")
async def get_answer_stream(request: QueryRequest):
    query = request.query
    handler = MyCustomHandler()
    config = {"callbacks": [handler]}
    
    if request.model_source == "Open Source":
        print("INFO: User Request Using Open Source Model")
        chain = app.state.chain_A
    elif request.model_source == "OpenAI" and request.api_key:
        print("INFO: User Request Using OpenAI Model")
        chain = app.state.chain_B
    else:
        async def error_stream():
            yield json.dumps({"error": "Invalid model source or missing API key."})
        return StreamingResponse(error_stream(), media_type="application/json", status_code=400)

    async def stream_generator():
        context_sent = False
        async for chunk in chain.astream(query, config=config):
            if not context_sent and handler.retrieved_docs:
                context_data = [
                    {"source": doc.metadata.get("source", "Unknown"), "content": doc.page_content}
                    for doc in handler.retrieved_docs
                ]
                context_json = json.dumps(context_data, ensure_ascii=False)
                
                yield context_json
                yield DELIMITER
                
                context_sent = True
            
            yield chunk

    return StreamingResponse(stream_generator(), media_type="text/event-stream")