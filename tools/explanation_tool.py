from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr
import json
import config

class ExplanationTool(BaseTool):
    name: str = "ExplainRecommendation"
    description: str = "Generate explanation for why model was recommended."

    _llm: ChatOpenAI = PrivateAttr()

    def __init__(self, model_name=config.OPENAI_MODEL_NAME, **kwargs):
        super().__init__(**kwargs)
        self._llm = ChatOpenAI(model=model_name, temperature=0, openai_api_key=config.OPENAI_API_KEY)

    def _run(self, input_json: dict) -> str:
        query = input_json["query"]
        model = input_json["model"]
        adaptation = input_json.get("adaptation", None)
        
        prompt = f"""You are an expert in remote sensing foundation model selection.

The structured user query is:
{query}

The final ranked candidate models with their metadata are:
{json.dumps(model, indent=2)}

Your task:
1. For each model, output a JSON object with:
   - "model_name"
   - "explanation": several bullet points on why it is recommended
   - "paper_link"
   - "repository"
2. Highlight how the model satisfies or partially satisfies the query.
3. Mention key trade-offs if relevant (accuracy vs. efficiency, modality 
coverage, etc.)."""

        return self._llm.invoke(prompt).content