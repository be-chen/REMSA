# === OpenAI ===
OPENAI_API_KEY = <your_key>
OPENAI_MODEL_NAME = "gpt-4.1"

# === Foundation Model Database ===
FMD_JSONL_PATH = "data/fmd.jsonl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_INDEX_PATH = "data/fmd_index.faiss"

# === Learning-to-Rank (L2R) Model ===
L2R_MODEL_PATH = "models/cross_encoder.pt"

# === Clarification Settings ===
MAX_CLARIFY = 3
TOP_K = 3 #5
CONFIDENCE_THRESHOLD = 0.6
CLARIFY_TEMPERATURE = 0

# === Retrieval & Memory ===
MAX_RETRIEVE =  50  # number of raw candidates to consider
MIN_SIMILARITY = 0.4  # cosine similarity threshold
SIMILARITY_BOUNDARY = 0.1
MEMORY_PATH = "data/memory.faiss"

# === Adaptation Rules ===
RULES_PATH = "rules.yaml"
