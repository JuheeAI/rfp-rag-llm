import os, json, argparse, chromadb
from chromadb import Client
from A_embedding_model import extract_texts_from_json, load_embedding_model, embed_texts
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

# ChromaDB 클라이언트 설정
def get_chroma_client(persist_path):
  return chromadb.PersistentClient(
    path=persist_path,
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE
)

# JSON 디렉토리 내 모든 파일 반복
def load_all_json_files(json_dir):
    file_paths = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]
    return file_paths

# 문서 ID, 메타데이터 생성 및 임베딩
def index_documents(chroma_client, collection_name, file_paths, model):
    collection = chroma_client.get_or_create_collection(collection_name)

    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = extract_texts_from_json(data)
        if not chunks:
            continue

        embeddings = embed_texts(model, chunks, for_faiss=False)

        doc_ids = [f"{os.path.basename(path)}-chunk{i}" for i in range(len(chunks))]
        metadatas = [{
            "파일명": os.path.basename(path),
            "청크번호": i,
            "공고번호": data.get("공고번호", "N/A"),
            "사업명": data.get("사업명", "N/A"),
        } for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            ids=doc_ids,
            metadatas=metadatas,
            embeddings=embeddings
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--model_key", type=str, default="kr-sbert")
    parser.add_argument("--persist_path", type=str, default="/tmp/chroma")
    parser.add_argument("--collection_name", type=str, default="rfp_chunks")
    args = parser.parse_args()

    chroma_client = get_chroma_client(args.persist_path)
    model = load_embedding_model(args.model_key)
    file_paths = load_all_json_files(args.json_dir)
    index_documents(chroma_client, args.collection_name, file_paths, model)