from langchain_core.tools import BaseTool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import PrivateAttr
import numpy as np
import config
import json
import os

class FMDRetrievalTool(BaseTool):
    name: str = "RetrieveModels"
    description: str = "Retrieve relevant foundation models based on query using cosine similarity filtering."

    _vectorstore: FAISS = PrivateAttr()
    _model_map: dict = PrivateAttr()
    _embedding: HuggingFaceEmbeddings = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Init embedding model
        self._embedding = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda"}
        )

        # Load FAISS vector store
        if os.path.exists(config.VECTOR_INDEX_PATH):
            self._vectorstore = FAISS.load_local(
                config.VECTOR_INDEX_PATH,
                self._embedding,
                allow_dangerous_deserialization=True
            )
        else:
            print("FAISS index not found. Initializing empty store.")
            print("Initializing empty vector index.")
            self._vectorstore = FAISS.from_texts([], self._embedding)

        # Load full FMD model metadata
        self._model_map = {}
        with open(config.FMD_JSONL_PATH, "r") as f:
            for line in f:
                model = json.loads(line)
                self._model_map[model["model_id"]] = model

    def _run(self, query: str | dict):
        try:
            # If it's a dict, serialize it to a string for embedding
            if isinstance(query, dict):
                query = json.dumps(query, indent=2)

            # Embed and normalize query
            query_emb = self._embedding.embed_query(query)
            query_emb = np.array(query_emb)
            query_emb = query_emb / np.linalg.norm(query_emb)

            # Retrieve many candidates (no hard cutoff)
            docs_and_scores = self._vectorstore.similarity_search_with_score(query, k=config.MAX_RETRIEVE)

            results = []
            tmp_list = []
            for doc, score in docs_and_scores:
                # FAISS uses inner product; normalize to interpret as cosine
                doc_emb = self._embedding.embed_query(doc.page_content)
                doc_emb = np.array(doc_emb)
                doc_emb = doc_emb / np.linalg.norm(doc_emb)
                cosine_sim = float(np.dot(query_emb, doc_emb))


                if cosine_sim >= config.SIMILARITY_BOUNDARY:
                    model_id = doc.metadata.get("model_id", "")
                    model_info = self._model_map.get(model_id, {})
                    model_info["similarity"] = round(cosine_sim, 4)
                    if cosine_sim >= config.MIN_SIMILARITY:
                        results.append(model_info)
                    else:
                        tmp_list.append(model_info)

            if len(results) < config.TOP_K:
                tmp_list.sort(key=lambda m: m.get("similarity", 0), reverse=True)
                for r in tmp_list:
                    results.append(r)
                    if len(results) >= config.TOP_K:
                        break
            return {
                "query": query,
                "candidates": results
            }

        except Exception as e:
            print("[Vector Store Error]", e)
            return {"error": f"Vector store error: {str(e)}"}