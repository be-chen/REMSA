from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr
from typing import Dict, List, Any
import config
import json

class RankerTool(BaseTool):
    name: str = "RankCandidates"
    description: str = "Use LLM to rank candidate models by relevance to the query, with explanations."

    _llm: ChatOpenAI = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._llm = ChatOpenAI(
            model=config.OPENAI_MODEL_NAME,
            temperature=0,
            openai_api_key=config.OPENAI_API_KEY
        )

    def _run(self, input: Dict[str, Any]) -> str:
        query = input.get("query", "")
        candidates = input.get("candidates", [])

        if not candidates:
            return "No candidates provided for ranking."

        prompt = self._build_prompt(query, candidates)

        try:
            return self._llm.invoke(prompt).content
        except Exception as e:
            print("[LLM Ranking Error]", e)
            return "LLM ranking failed. Returning unranked list."

    def _build_prompt(self, query: str, candidates: List[Dict[str, Any]]) -> str:
        model_infos = ""
        for i, m in enumerate(candidates):
            m.pop("similarity", None)
            candidate = json.dumps(m, indent=2)
            model_infos += (
                f"{i+1}. Candidate Model Info: {candidate}\n"
            )

        return f"""You are an expert in remote sensing foundation model selection.

You will be given:
1. A structured user query specifying task requirements and constraints.
2. A list of candidate models retrieved from a database, each with
metadata fields.

Your goal:
- Rank the candidate models from most to least suitable for the user's query.
- For each model, provide a brief explanation in several bullet points 
describing why it is placed at that rank.
- Prioritize hard constraints (task, modality, required sensor, 
and min_performance if provided), then consider secondary preferences 
(spatial/temporal resolution, application type, domain keywords, etc.).
- When two models equally satisfy the constraints and preferences, prefer 
the model that is more efficient, better validated on diverse benchmarks, 
or more versatile(multimodal, multi-temporal).

Few-shot examples:

[Example 1]
Structured Query:
{
  "task": "land cover classification",
  "modality": "multispectral",
  "sensor": ["Sentinel-2"],
  "min_performance": {
    "metric": ["accuracy"],
    "value": [85]
  }
}

Candidate Models:
1. S2MAE
2. Prithvi
3. CACo

Ranking Output:
1. S2MAE
   - Directly supports Sentinel-2 multispectral data
   - Achieves 99.1\% accuracy on EuroSAT, exceeding 85\% requirement
   - Purpose-built for land cover classification
2. Prithvi
   - Supports multi-temporal multispectral data, including Sentinel-2
   - Accuracy slightly below requirement on similar tasks
   - More generalist FM
3. CACo
   - Only supports RGB modality
   - Accuracy below the 85\% requirement
   - Designed mainly for change detection and event retrieval

[Example 2]
Structured Query:
{{
  "task": "change detection",
  "modality": "SAR",
  "region": ["Europe"]
}}

Candidate Models:
1. SSL4EO-S12
2. MATTER 
3. CACo

Ranking Output:
1. SSL4EO-S12
   - Only candidate that supports SAR modality
   - Global coverage includes Europe
   - Pre-trained for multi-temporal applications
2. MATTER
   - Optical-only; cannot fully satisfy SAR requirement
   - Has multi-temporal capabilities relevant for change detection
3. CACo
   - Limited to RGB, unsuitable for SAR-based change detection

Your Task:
Given the following new query and candidates, produce a ranked list 
with explanations.

Structured Query:
{query}

Candidate Models:
{model_infos}

Please output the ranked list as JSON in the following format:
[
  {{
    "model": <model_name>,
    "rank": <integer>,
    "reason": [<short bullet points>]
  }},
  ...
]
"""

