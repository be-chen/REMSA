import json
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
import config

def flatten(entry, prefix=""):
    flattened = []
    if isinstance(entry, dict):
        for k, v in entry.items():
            key = f"{prefix}{k}" if prefix == "" else f"{prefix}.{k}"
            flattened.extend(flatten(v, prefix=key))
    elif isinstance(entry, list):
        for i, item in enumerate(entry):
            key = f"{prefix}[{i}]"
            flattened.extend(flatten(item, prefix=key))
    else:
        flattened.append(f"{prefix}: {entry}")
    return flattened

def build_vectorstore():
    print("Building FAISS vector store from FMD...")

    with open(config.FMD_JSONL_PATH, 'r') as f:
        entries = [json.loads(line) for line in f]

    docs = []
    for entry in entries:
        flat_text = "\n".join(flatten(entry))
        docs.append(Document(page_content=flat_text, metadata={"model_id": entry.get("model_id", "unknown")}))

    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"}  # or "cpu"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(config.VECTOR_INDEX_PATH)

    print(f"FAISS index saved to {config.VECTOR_INDEX_PATH}")

if __name__ == "__main__":
    build_vectorstore()
