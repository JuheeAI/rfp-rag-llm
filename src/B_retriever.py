import os
import re
import time
import json
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

import openai
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI

# 환경 변수 로드
load_dotenv()

# 문장 분리 함수
def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

# 문서 로딩 함수
def load_documents_json(folder_path, limit_files=None):
    all_docs = []
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".json")])
    if limit_files:
        files = files[:limit_files]

    for filename in tqdm(files, desc="Loading JSON documents"):
        try:
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            사업명 = data.get("사업명", "")
            공고번호 = data.get("공고번호", "")
            페이지들 = data.get("페이지별_데이터", [])

            for page in 페이지들:
                page_text = page.get("text", "")
                if page_text.strip():
                    all_docs.append(Document(
                        page_content=page_text.strip(),
                        metadata={"사업명": 사업명, "공고번호": 공고번호, "source": filename}
                    ))
        except Exception as e:
            print(f"[!] {filename} 불러오기 실패: {e}")
    return all_docs

# 청킹 함수
def semantic_chunk_documents(documents, max_chunk_len=300):
    chunked_docs = []
    for doc in tqdm(documents, desc="Chunking documents"):
        text = doc.page_content
        metadata = doc.metadata
        sentences = split_sentences(text)

        buffer = ""
        for sentence in sentences:
            if not sentence.strip():
                continue
            if re.match(r"^\d{1,2}\.\s", sentence) or sentence.startswith("■") or re.match(r"^[가-하]\)", sentence):
                if buffer.strip():
                    chunked_docs.append(Document(page_content=buffer.strip(), metadata=metadata))
                buffer = sentence + " "
                continue
            if len(buffer) + len(sentence) <= max_chunk_len:
                buffer += sentence + " "
            else:
                chunked_docs.append(Document(page_content=buffer.strip(), metadata=metadata))
                buffer = sentence + " "
        if buffer.strip():
            chunked_docs.append(Document(page_content=buffer.strip(), metadata=metadata))
    return chunked_docs

def faiss_index_exists(index_path):
    return os.path.exists(os.path.join(index_path, "index.faiss")) and os.path.exists(os.path.join(index_path, "index.pkl"))

# FAISS 인덱스 빌드
def build_faiss_index(docs, embedding, batch_size=50):
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    embeddings = []
    print("\nEmbedding in batches...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        embs = embedding.embed_documents(batch)
        embeddings.extend(embs)

    print(f"Total chunks: {len(texts)} | Total embeddings: {len(embeddings)}")

    text_embedding_pairs = list(zip(texts, embeddings))
    return FAISS.from_embeddings(text_embeddings=text_embedding_pairs, embedding=embedding, metadatas=metadatas)


# 리트리버 생성
def get_retriever(documents_path, index_path="faiss_index/", reuse_index=True, k=5, limit_files=None):
    start_time = time.time()

    documents = load_documents_json(documents_path, limit_files=limit_files)
    chunks = semantic_chunk_documents(documents, max_chunk_len=300)

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    if reuse_index and faiss_index_exists(index_path):
        print("Loading existing FAISS index...")
        vector_db = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
    else:
        vector_db = build_faiss_index(chunks, embedding)
        vector_db.save_local(index_path)

    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 2}
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

    retriever_chain = RunnableLambda(lambda q: retriever.get_relevant_documents(q))

    chain = (
        {"context": retriever_chain, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# 예시 실행
if __name__ == "__main__":
    retriever = get_retriever(r"/home/data/preprocess/json", reuse_index=True, limit_files=None)
    chain = build_chain(retriever)

    while True:
        query = input("\n질문을 입력하세요 (exit 입력 시 종료): ")
        if query.lower() == "exit":
            break
        result = chain.invoke(query)
        print("\n답변:")
        print(result)
