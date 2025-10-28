from langchain.memory import ConversationBufferMemory
import os

def get_memory():
    if not os.path.exists("data/memory.faiss"):
        print("🧠 No memory found, initializing fresh FAISS memory.")
        # You can also initialize a blank FAISS index if you plan to use vector memory.
    else:
        print("🔁 Loading existing memory.")
    
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)
