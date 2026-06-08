from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Union, Optional
import config
import json

class QuerySchema(BaseModel):
    task: str
    modality: str
    application: Optional[str] = None
    sensor: Optional[Union[str, List[str]]] = None
    spatial_resolution: Optional[Union[str, float]] = None
    temporal_resolution: Optional[Union[str, float]] = None
    bands: Optional[List[str]] = None
    deployment_device: Optional[str] = None
    min_performance: Optional[dict] = None
    region: Optional[Union[str, List[str]]] = None
    domain_keywords: Optional[List[str]] = None


class QueryParserTool(BaseTool):
    name: str = "QueryParser"
    description: str = "Parse user natural language query into structured schema."

    def _run(self, query: str) -> dict:
        prompt = f"""
Parse the user's query into the following structured format:

{QuerySchema.schema_json(indent=2)}

User query:
\"\"\"{query}\"\"\"

Return only a JSON object. Do not add explanations or markdown.
"""
        llm = ChatOpenAI(model=config.OPENAI_MODEL_NAME, temperature=0, openai_api_key=config.OPENAI_API_KEY)
        result = llm.invoke(prompt)
        try:
            return json.loads(result.content)
        except Exception:
            return {"error": "Could not parse query."}
