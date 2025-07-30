import argparse, chromadb
from chromadb import PersistentClient
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from A_embedding_model import load_embedding_model

def get_chroma_client(persist_path):
    return PersistentClient(
        path=persist_path,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

def query_documents(chroma_client, collection_name, query, model, top_k=5):
    collection = chroma_client.get_collection(name=collection_name)
    print(f"[DEBUG] collection name: {collection_name}")
    print(f"[DEBUG] query: {query}")
    print(f"[DEBUG] top_k: {top_k}")
    print(f"[DEBUG] collection count: {collection.count()}")
    query_embedding = model.encode([query])[0]
    print(f"[DEBUG] query embedding[:5]: {query_embedding[:5]}")
    print("[DEBUG] 현재 컬렉션 ID 목록:", collection.get()["ids"])
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    if not results["documents"] or not results["documents"][0]:
        print("[INFO] 검색된 결과가 없습니다.")
        return
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"\n[결과 {i+1}]")
        print("▶ 관련 문장:", doc)
        print("▶ 메타정보:", meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="검색할 질문 문장")
    parser.add_argument("--model_key", type=str, required=True, help="사용할 임베딩 모델 키")
    parser.add_argument("--persist_path", type=str, required=True, help="ChromaDB persist 디렉토리 경로")
    parser.add_argument("--collection_name", type=str, required=True, help="조회할 컬렉션 이름")
    parser.add_argument("--top_k", type=int, default=3, help="검색할 top-k 결과 수")

    args = parser.parse_args()
    model = load_embedding_model(args.model_key)
    client = get_chroma_client(args.persist_path)
    print("[DEBUG] 현재 존재하는 컬렉션 목록:")
    for col in client.list_collections():
        print(f"[DEBUG] 컬렉션 이름: {col.name}, ID: {col.id}")
    query_documents(client, args.collection_name, args.query, model, top_k=args.top_k)