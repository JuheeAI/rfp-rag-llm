import os
import re
import time
import openai
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import json

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOpenAI
from llama_index.readers.file import HWPReader

# 환경 변수 로드
def find_and_load_dotenv(start_path: Path = Path(__file__).resolve(), filename=".env"):
    current = start_path.parent
    while current != current.parent:
        env_path = current / filename
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f".env loaded from: {env_path}")
            return True
        current = current.parent
    print(".env 파일을 찾지 못했습니다.")
    return False

find_and_load_dotenv()

# 문장 분리 함수
def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

# 문서 로딩 함수 (확장된 메타데이터 반영하여여 수정)
def load_documents(folder_path, limit_files=None):
    all_docs = []
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".json")])
    if limit_files:
        files = files[:limit_files]

    for filename in tqdm(files, desc="Loading documents"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metadata = data.get("csv_metadata", {})
        page_texts = [
            page.get("text", "").strip()
            for page in data.get("pdf_data", [])
            if page.get("text", "").strip()
        ]
        full_text = "\n".join(page_texts)

        if full_text:
            all_docs.append(Document(
                page_content=full_text,
                metadata={
                    "사업명": metadata.get("사업명", ""),
                    "공고번호": metadata.get("공고 번호", ""),
                    "공고차수": metadata.get("공고 차수", ""),
                    "사업금액": metadata.get("사업 금액", ""),
                    "발주기관": metadata.get("발주 기관", ""),
                    "입찰참여시작일": metadata.get("입찰 참여 시작일", ""),
                    "입찰참여마감일": metadata.get("입찰 참여 마감일", ""),
                    "사업요약": metadata.get("사업 요약", ""),
                    "파일명": metadata.get("파일명", ""),
                    "source": filename
                }
            ))
    return all_docs


# 청킹 함수 (json에 - 숫자 - 형식이 많았기에 그에 맞게 수정
def semantic_chunk_documents(documents, max_chunk_len=300, overlap_len=0):
    chunked_docs = []
    for doc in tqdm(documents, desc="Chunking documents"):
        text = doc.text if hasattr(doc, "text") else doc.page_content
        metadata = doc.metadata
        sentences = re.split(r'(?<=[\.\?])\s+', text.strip())

        buffer = ""
        last_sentences = []

        for sentence in sentences:
            if not sentence.strip():
                continue

            # 실제로 유효한 유일한 기준
            if re.match(r"^- \d+ -", sentence):
                if buffer.strip():
                    chunked_docs.append(Document(page_content=buffer.strip(), metadata=metadata))
                buffer = sentence + " "
                last_sentences = [sentence]
                continue

            if len(buffer) + len(sentence) <= max_chunk_len:
                buffer += sentence + " "
                last_sentences.append(sentence)
            else:
                chunked_docs.append(Document(page_content=buffer.strip(), metadata=metadata))
                buffer = " ".join(last_sentences[-overlap_len:]) + " " + sentence + " " if overlap_len > 0 else sentence + " "
                last_sentences = last_sentences[-overlap_len:] + [sentence]

        if buffer.strip():
            chunked_docs.append(Document(page_content=buffer.strip(), metadata=metadata))
    return chunked_docs

# FAISS 인덱스 빌드 (문장 단위가 아닌 전체 글단위로 수정)
def build_faiss_index(docs, embedding, batch_size=50):

    # 중복 제거
    unique_pairs = {}
    for doc in docs:
        key = doc.page_content.strip()
        if key not in unique_pairs:
            unique_pairs[key] = doc.metadata

    texts = list(unique_pairs.keys())
    metadatas = list(unique_pairs.values())

    # 너무 짧은 텍스트 필터링
    filtered = [(t, m) for t, m in zip(texts, metadatas) if len(t) > 20]
    texts, metadatas = zip(*filtered) if filtered else ([], [])

    # 임베딩 수행
    embeddings = []
    print("\nEmbedding in batches...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        embs = embedding.embed_documents(batch)
        embeddings.extend(embs)

    print(f"Total chunks: {len(texts)} | Total embeddings: {len(embeddings)}")

    return FAISS.from_embeddings(
        text_embeddings=list(zip(texts, embeddings)),
        embedding=embedding,
        metadatas=metadatas
    )

    
# 리트리버 생성
def get_retriever(documents_path, index_path="/home/data/B_faiss_db/", reuse_index=True, k=5, limit_files=None):
    start_time = time.time()

    documents = load_documents(documents_path, limit_files=limit_files)
    chunks = semantic_chunk_documents(documents, max_chunk_len=300, overlap_len=1)  

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

    if reuse_index and os.path.exists(index_path):
        print("Loading existing FAISS index...")
        vector_db = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
    else:
        vector_db = build_faiss_index(chunks, embedding)
        vector_db.save_local(index_path)

    retriever = vector_db.as_retriever(
        search_type="similarity",  
        search_kwargs={"k": k}
    )

    print(f"Retriever ready in {time.time() - start_time:.2f} seconds")
    return retriever


# LLM QA 체인 생성
def build_chain(retriever):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    prompt = PromptTemplate.from_template(
        """쿼리 입력 : 
문맥:
{context}

질문:
{question}

답변:"""
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain



# 예시 실행
if __name__ == "__main__":

    retriever = get_retriever("/home/data/data/", reuse_index=True, limit_files=None) # 리트리버 로드
    chain = build_chain(retriever) # LLM QA 체인 생성 

    #쿼리 입력 파트
    while True:
        query = input("\n 질문을 입력하세요 (exit 입력 시 종료): ") 
        if query.lower() == "exit":
            break
        result = chain.invoke(query)
        print("\n답변:")
        print(result)  
