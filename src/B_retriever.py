import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from llama_index.readers.file import HWPReader
from tqdm import tqdm

# 🔑 환경 변수 로드
load_dotenv()

# ✅ 문장 분리 함수

def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

# ✅ 문서 로딩 함수 (PDF, HWP 지원)
def load_documents(folder_path, limit_files=None):
    all_docs = []
    hwp_reader = HWPReader()
    files = sorted(os.listdir(folder_path))
    if limit_files:
        files = files[:limit_files]

    for filename in tqdm(files, desc="📄 Loading documents"):
        file_path = os.path.join(folder_path, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            elif filename.endswith(".hwp"):
                docs = hwp_reader.load_data(Path(file_path))
            else:
                continue
            for doc in docs:
                doc.metadata["source"] = filename
            all_docs.extend(docs)
        except Exception as e:
            print(f"[!] {filename} 불러오기 실패: {e}")
    return all_docs

# ✅ 청킹 함수

def semantic_chunk_documents(documents, max_chunk_len=300):
    chunked_docs = []
    for doc in tqdm(documents, desc="🔪 Chunking documents"):
        text = doc.text if hasattr(doc, "text") else doc.page_content
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

# ✅ FAISS 인덱스 빌드 (배치 + 진행바 + 시간측정)
def build_faiss_index(docs, embedding, batch_size=50):
    from langchain_community.vectorstores.faiss import FAISS

    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    embeddings = []
    print("\n🔁 Embedding in batches...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        emb = embedding.embed_documents(batch)
        embeddings.extend(emb)

    print(f"✅ Total chunks: {len(docs)} | Total embeddings: {len(embeddings)}")

    vector_db, _ = FAISS.from_embeddings(embeddings=embeddings, documents=docs)
    return vector_db



# ✅ 최종 리트리버 생성

def get_retriever(documents_path, index_path="faiss_index/", reuse_index=True, k=5, limit_files=None):
    start_time = time.time()
    documents = load_documents(documents_path, limit_files=limit_files)
    chunks = semantic_chunk_documents(documents, max_chunk_len=300)

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    if reuse_index and os.path.exists(index_path):
        print("📁 Loading existing FAISS index...")
        vector_db = FAISS.load_local(index_path, embedding)
    else:
        vector_db = build_faiss_index(chunks, embedding)
        vector_db.save_local(index_path)

    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 2}
    )

    print(f"⏱️ Retriever ready in {time.time() - start_time:.2f} seconds")
    return retriever

# ✅ 실행 (파일 수 제한 가능)
retriever = get_retriever("./data/", limit_files=None)  # limit_files=10 처럼 테스트 가능