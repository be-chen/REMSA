from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr
import config

class ClarifierTool(BaseTool):
    name: str = "Clarifier"
    description: str = "Ask follow-up questions if the user's query is under-specified."

    _llm: ChatOpenAI = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._llm = ChatOpenAI(model=config.OPENAI_MODEL_NAME, temperature=config.CLARIFY_TEMPERATURE, openai_api_key=config.OPENAI_API_KEY)

    def _run(self, input: dict) -> str:
        partial_query = input.get("structured_query", {})
        missing_fields = input.get("missing_fields", [])
        phase = input.get("phase", "mandatory")  # can be "mandatory" or "optional"

        prompt = f"""
You are helping a user select a foundation model for a remote sensing task.

The user's current query is:
{partial_query}

You need to ask follow-up questions for the following missing {phase} fields:
{missing_fields}

Please ask one concise and clear question per field. Stay grounded in remote sensing and avoid technical jargon.

Format your response as a numbered list.
"""
        return self._llm.invoke(prompt).content
