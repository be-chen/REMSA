from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

from tools.retrieval_tool import FMDRetrievalTool
from tools.ranking_tool import RankerTool
from tools.clarifier_tool import ClarifierTool
from tools.adaptation_tool import AdaptationTool
from tools.explanation_tool import ExplanationTool
from tools.memory_tool import get_memory

import config

# === Initialize the core LLM ===
llm = ChatOpenAI(
    model=config.OPENAI_MODEL_NAME,
    temperature=0,
    openai_api_key=config.OPENAI_API_KEY
)

# === Initialize tools ===
tools = [
    Tool.from_function(
        func=FMDRetrievalTool(
            fmd_path=config.FMD_JSONL_PATH,
            embedding_model=config.EMBEDDING_MODEL_NAME
        ),
        name="RetrieveModels",
        description="Retrieve top-k relevant models from database. k is at least 4, at most 6."
    ),
    Tool.from_function(
        func=RankerTool(),
        name="RankCandidates",
        description="If necessary, rank retrieved top-k candidate models based on user query. The input of this tool should be the dictionary returned from the FMDRetrievalTool that contains both the query and the retrieved candidates."
    ),
    Tool.from_function(lambda query_and_models: RankerTool()._run(query_and_models), name="RankCandidates", description="Rank candidate models based on user query."),
    Tool.from_function(
        func=ClarifierTool(),
        name="ClarifyUser",
        description="Ask for clarification when query is underspecified"
    ),
    Tool.from_function(
        func=ExplanationTool(),
        name="ExplainRecommendation",
        description="Generate explanation for selected model"
    )
]

def create_agent():

    memory = get_memory()  # fresh memory
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        memory=memory,
        verbose=True
    )