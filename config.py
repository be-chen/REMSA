# === OpenAI ===
OPENAI_API_KEY = "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"
OPENAI_MODEL_NAME = "gpt-4.1"

# === Foundation Model Database ===
FMD_JSONL_PATH = "data/fmd.jsonl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_INDEX_PATH = "data/fmd_index.faiss"

# === Learning-to-Rank (L2R) Model ===
L2R_MODEL_PATH = "models/cross_encoder.pt"

# === Clarification Settings ===
MAX_CLARIFY = 3
TOP_K = 6 #5
CONFIDENCE_THRESHOLD = 0.6
CLARIFY_TEMPERATURE = 0

# === Retrieval & Memory ===
MAX_RETRIEVE =  50  # number of raw candidates to consider
MIN_SIMILARITY = 0.4  # cosine similarity threshold
SIMILARITY_BOUNDARY = 0.1
MEMORY_PATH = "data/memory.faiss"

# === Adaptation Rules ===
RULES_PATH = "rules.yaml"