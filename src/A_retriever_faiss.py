import faiss
import pickle
import numpy as np
from A_embedding_model import load_embedding_model
from langchain_core.documents import Document
from langchain.vectorstores.faiss import FAISS
from langchain_core.embeddings import Embeddings

def load_langchain_retriever(base_path: str, model_key: str, embedding_model: Embeddings):
    full_path = f"{base_path}/{model_key}"
    return FAISS.load_local(full_path, embedding_model, allow_dangerous_deserialization=True).as_retriever()

if __name__ == "__main__":

    embedding_model = load_embedding_model("kr-sbert")

    retriever = load_langchain_retriever(
        base_path="/home/data/A_faiss_db",
        model_key="kr-sbert",
        embedding_model=embedding_model
    )

    query = "재단법인스포츠윤리센터_스포츠윤리센터에서 지식재산권 관련 내용 찾아줘"
    docs = retriever.invoke(query)

    for i, doc in enumerate(docs, 1):
        print(f"\n[{i}]")
        print("내용:", doc.page_content)
        print("메타데이터:", doc.metadata)