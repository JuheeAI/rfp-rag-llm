import os
import re
import time
import json
import hashlib
import faiss
import numpy as np
import nltk
import tiktoken
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

nltk.download('punkt_tab')
nltk.download('punkt')
load_dotenv()


def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())


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

        # 검색 성능을 높이기 위해 중요한 메타데이터를 본문 텍스트 앞에 추가
        metadata_header = (
            f"[문서 요약 정보]\n"
            f"- 사업명: {metadata.get('사업명', '정보 없음')}\n"
            f"- 사업 금액: {metadata.get('사업 금액', '정보 없음')}\n"
            f"- 발주 기관: {metadata.get('발주 기관', '정보 없음')}\n"
            f"- 파일명: {metadata.get('파일명', '정보 없음')}\n\n"
            f"- 사업 요약: {metadata.get('사업 요약', '정보 없음')}\n\n"
        )
        # 헤더와 실제 본문을 합쳐서 새로운 page_content를 만듦
        content_with_metadata = metadata_header + full_text

        if content_with_metadata:
            # Document 객체를 만들 때, 메타데이터가 추가된 텍스트를 page_content로 사용
            all_docs.append(Document(
                page_content=content_with_metadata,
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


def semantic_token_chunk_documents(documents, max_tokens=300, overlap_tokens=50, model_name="text-embedding-3-small"):
    enc = tiktoken.encoding_for_model(model_name)
    chunked_docs = []

    for doc in tqdm(documents, desc="Token-based Chunking"):
        text = doc.text if hasattr(doc, "text") else doc.page_content
        metadata = doc.metadata
        sentences = nltk.sent_tokenize(text)
        buffer = []
        buffer_token_count = 0

        for sentence in sentences:
            sentence_tokens = enc.encode(sentence)
            sentence_len = len(sentence_tokens)

            if buffer_token_count + sentence_len <= max_tokens:
                buffer.append(sentence)
                buffer_token_count += sentence_len
            else:
                chunked_docs.append(Document(page_content=" ".join(buffer), metadata=metadata))
                if overlap_tokens > 0:
                    overlap_text = " ".join(buffer)[-overlap_tokens:]
                    overlap_tokens_list = enc.encode(overlap_text)
                    buffer = [enc.decode(overlap_tokens_list)] + [sentence]
                    buffer_token_count = len(enc.encode(" ".join(buffer)))
                else:
                    buffer = [sentence]
                    buffer_token_count = sentence_len

        if buffer:
            chunked_docs.append(Document(page_content=" ".join(buffer), metadata=metadata))

    return chunked_docs


def get_retriever(documents_path, index_path="/home/data/B_faiss_db", reuse_index=True, k=5, limit_files=None):
    start_time = time.time()

    documents = load_documents(documents_path, limit_files=limit_files)
    if not documents:
        raise ValueError("No documents found.")

    chunks = semantic_token_chunk_documents(
        documents,
        max_tokens=500,
        overlap_tokens=50,
        model_name="text-embedding-3-small"
    )
    if not chunks:
        raise ValueError("No chunks created.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    embedding = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    enc = tiktoken.encoding_for_model("text-embedding-3-small")

    cache_file = "embedding_cache.json"
    cache = json.load(open(cache_file, "r", encoding="utf-8")) if os.path.exists(cache_file) else {}

    if reuse_index and os.path.exists(index_path):
        print("Loading existing FAISS index...")
        faiss_db = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
    else:
        unique_pairs = {doc.page_content.strip(): doc.metadata for doc in chunks}
        texts = list(unique_pairs.keys())
        metadatas = list(unique_pairs.values())
        if not texts:
            raise ValueError("No texts for embedding.")

        filtered = [(t, m) for t, m in zip(texts, metadatas) if len(enc.encode(t)) > 2]
        texts, metadatas = zip(*filtered) if filtered else ([], [])
        if not texts:
            raise ValueError("All texts filtered out.")

        embeddings, new_cache_entries = [], {}
        print("\nEmbedding in batches with caching & token safety...")
        for i in tqdm(range(0, len(texts), 100)):
            batch = texts[i:i + 100]
            batch_to_embed, cache_hits = [], []

            for text in batch:
                h = hashlib.md5(text.encode("utf-8")).hexdigest()
                if h in cache:
                    embeddings.append(cache[h])
                else:
                    batch_to_embed.append(text)
                    cache_hits.append(h)

            if batch_to_embed:
                total_tokens = sum(len(enc.encode(t)) for t in batch_to_embed)
                if total_tokens > 280_000:
                    mid = len(batch_to_embed) // 2
                    embs = embedding.embed_documents(batch_to_embed[:mid]) + embedding.embed_documents(batch_to_embed[mid:])
                else:
                    embs = embedding.embed_documents(batch_to_embed)

                for h, emb in zip(cache_hits, embs):
                    cache[h] = emb
                    new_cache_entries[h] = emb
                embeddings.extend(embs)

        if new_cache_entries:
            json.dump(cache, open(cache_file, "w", encoding="utf-8"))

        faiss_db = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            embedding=embedding,
            metadatas=metadatas
        )
        faiss_db.save_local(index_path)

    retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    print(f"FAISS Retriever ready in {time.time() - start_time:.2f} seconds")
    return retriever


def enrich_documents_with_metadata(docs):
    enriched = []
    for doc in docs:
        meta = doc.metadata
        meta_text = (
            f"[메타데이터]\n"
            f"- 사업명: {meta.get('사업명', '')}\n"
            f"- 공고번호: {meta.get('공고번호', '')}\n"
            f"- 공고차수: {meta.get('공고차수', '')}\n"
            f"- 사업금액: {meta.get('사업금액', '')}\n"
            f"- 발주기관: {meta.get('발주기관', '')}\n"
            f"- 입찰참여시작일: {meta.get('입찰참여시작일', '')}\n"
            f"- 입찰참여마감일: {meta.get('입찰참여마감일', '')}\n"
            f"- 사업요약: {meta.get('사업요약', '')}\n"
            f"- 파일명: {meta.get('파일명', '')}\n"
        )
        enriched.append(meta_text + "\n" + doc.page_content)
    return "\n\n".join(enriched)


def build_chain(retriever):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    prompt = PromptTemplate.from_template(
        """당신은 정부 사업 공고서를 요약해주는 비서입니다.

문맥:
{context}

사용자의 질문:
{question}

답변:"""
    )

    def full_chain_fn(question):
        docs = retriever.invoke(question)
        context = enrich_documents_with_metadata(docs)
        return prompt.format(context=context, question=question)

    chain = RunnablePassthrough() | full_chain_fn | llm | StrOutputParser()
    return chain


if __name__ == "__main__":
    retriever = get_retriever(
        documents_path="/home/data/preprocess/json", 
        index_path="/home/data/B_faiss_db", 
        reuse_index=True, 
        limit_files=None
        )
    chain = build_chain(retriever)
    while True:
        query = input("\n 질문을 입력하세요 (exit 입력 시 종료): ")
        if query.lower() == "exit":
            break
        result = chain.invoke(query)
        print("\n답변:")
        print(result)
