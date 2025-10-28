from tools.query_parser import QueryParserTool, QuerySchema
from tools.retrieval_tool import FMDRetrievalTool
from tools.ranking_tool import RankerTool
from tools.adaptation_tool import AdaptationTool
from tools.explanation_tool import ExplanationTool
from tools.clarifier_tool import ClarifierTool
from typing import Tuple
import config
import json

MANDATORY_FIELDS = [
    name for name, field in QuerySchema.model_fields.items()
    if field.is_required()
]

class FMSAgent:
    def __init__(self):
        self.parser = QueryParserTool()
        self.retriever = FMDRetrievalTool(
            fmd_path=config.FMD_JSONL_PATH,
            embedding_model=config.EMBEDDING_MODEL_NAME
        )
        self.ranker = RankerTool()
        self.adapt = AdaptationTool()
        self.explainer = ExplanationTool()
        self.clarifier = ClarifierTool()
        self.max_clarify = config.MAX_CLARIFY
        self.max_candidates = 20
        self.confidence_threshold = 0.7

    def _missing_required_fields(self, parsed: dict) -> Tuple[list, list]:
        missing_mandatory = [k for k in MANDATORY_FIELDS if not parsed.get(k)]
        missing_optional = [k for k, v in parsed.items() if not v and k not in MANDATORY_FIELDS]
        return missing_mandatory, missing_optional

    def _filter_candidates(self, candidates: list, constraints: dict) -> list:
        filtered = []
        for model in candidates:
            if not isinstance(model, dict):
                continue
            match = True
            print(json.dumps(model))
            print(constraints['task'])
            print(constraints['modality'])
            if "task" in constraints and constraints["task"] not in json.dumps(model).lower():
                match = False
            if "modality" in constraints and constraints["modality"] not in json.dumps(model).lower():
                match = False
            if match:
                filtered.append(model)
        return filtered

    def _compute_confidence(self, scores: list) -> float:
        if not scores:
            return 0.0
        confs = [s.get("confidence", 0) for s in scores]
        return sum(confs) / len(confs)

    def run(self, user_query: str):
        clarify_count = 0
        query = user_query

        # === Clarify until mandatory constraints are present ===
        while True:
            parsed = self.parser._run(query)
            if "error" in parsed:
                return "[Agent] Error parsing query."

            missing_mandatory, missing_optional = self._missing_required_fields(parsed) # missing madatory fields and optional fields
            if missing_mandatory:
                if clarify_count < self.max_clarify:
                    clarification = self.clarifier._run({"structured_query": parsed, "missing_fields": missing_mandatory, "phase": "mandatory"})
                    print(f"[Agent clarification] {clarification}")
                    user_reply = input("[User Clarification] >> ")
                    query += f"\n{user_reply}"
                    clarify_count += 1
                    continue
                else:
                    return f"[Agent] Could not extract required constraints: {missing_mandatory}"
            break

        # === Retrieve candidates ===
        results = self.retriever._run(query)
        candidates = results["candidates"]
        if isinstance(candidates, dict) and "error" in candidates:
            return f"[Agent] Retrieval failed: {candidates['error']}"

        while True:
            if len(candidates) > self.max_candidates:
                if clarify_count < self.max_clarify:
                    clarification = self.clarifier._run({"structured_query": parsed, "missing_fields": missing_optional, "phase": "optional"})
                    print(f"[Agent clarification] {clarification}")
                    user_reply = input("[User Clarification] >> ")
                    query += f"\n{user_reply}"
                    clarify_count += 1
                    results = self.retriever._run(query)
                    candidates = results["candidates"]
                    print(len(candidates))
                else:
                    results["candidates"]=candidates[:self.max_candidates]
                    break
            else:
                break

        model_score = []
        for c in candidates:
            model_score.append({'model_name': c['model_name'], 'score': c['similarity']})

        # === Rank the candiates with LLM ===
        response = self.ranker._run(results)
        return response, model_score
