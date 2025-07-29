# src/backend.py

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.A_generation import create_generation_chain
# from src.B_generation import create_generation_chain_openai


chain = create_generation_chain()
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/get_answer")
async def get_answer_stream(request: QueryRequest):
    answer = StreamingResponse(
        (chunk for chunk in chain.stream(request.query)),
        media_type="text/event-stream"
    )
    return answer



# main.py에서 실행하므로 주석처리
# import uvicorn
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=9000)