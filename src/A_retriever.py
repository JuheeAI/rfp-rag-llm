import faiss
import pickle
import numpy as np
from A_embedding import load_embedding_model
from langchain_core.documents import Document
from langchain.vectorstores.faiss import FAISS
from langchain_core.embeddings import Embeddings

def load_langchain_retriever(base_path: str, model_key: str, embedding_model: Embeddings):
    full_path = f"{base_path}/{model_key}"
    return FAISS.load_local(full_path, embedding_model, allow_dangerous_deserialization=True).as_retriever()