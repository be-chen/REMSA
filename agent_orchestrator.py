from tools.query_parser import QueryParserTool, QuerySchema
from tools.retrieval_tool import FMDRetrievalTool
from tools.ranking_tool import RankerTool
# from tools.adaptation_tool import AdaptationTool
from tools.explanation_tool import ExplanationTool
from tools.clarifier_tool import ClarifierTool
from typing import Tuple, Any, Dict
import config
import json
from copy import deepcopy

MANDATORY_FIELDS = [
    name for name, field in QuerySchema.model_fields.items()
    if field.is_required()
]

def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, dict, tuple, set)) and len(value) == 0:
        return True
    return False

def update_b_with_non_empty_a(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    new_b = deepcopy(b)
    for key, a_value in a.items():
        if not is_empty(a_value):
            b_value = new_b.get(key)
            if a_value != b_value:
                new_b[key] = deepcopy(a_value)
    return new_b

class FMSAgent:
    def __init__(self):
        self.parser = QueryParserTool()
        self.retriever = FMDRetrievalTool(
            fmd_path=config.FMD_JSONL_PATH,
            embedding_model=config.EMBEDDING_MODEL_NAME
        )
        self.ranker = RankerTool()
        # self.adapt = AdaptationTool()
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
                    print(f"[Agent clarification] Please answer the following questions: \n{clarification}")
                    user_reply = input("[User Clarification] >> ")
                    if user_reply.lower() in ["exit", "quit"]:
                        return "exit", "exit"
                    query += f"\n{user_reply}"
                    clarify_count += 1
                    continue
                else:
                    return f"[Agent] Could not extract required constraints: {missing_mandatory}"
            break

        # === Retrieve candidates ===
        # results = self.retriever._run(query)
        results = self.retriever._run(parsed)
        candidates = results["candidates"]
        struc_query = results["query"]
        if isinstance(candidates, dict) and "error" in candidates:
            return f"[Agent] Retrieval failed: {candidates['error']}"

        while True:
            if len(candidates) > self.max_candidates:
                if clarify_count < self.max_clarify:
                    clarification = self.clarifier._run({"structured_query": parsed, "missing_fields": missing_optional, "phase": "optional"})
                    print(f"[Agent clarification] Please answer the following questions as much as you can: \n{clarification}")
                    user_reply = input("[User Clarification] >> ")
                    if user_reply.lower() in ["exit", "quit"]:
                        return "exit", "exit"
                    struc_query += f"\n{user_reply}"
                    struc_query_new = self.parser._run(struc_query)
                    if "error" in parsed:
                        return "[Agent] Error parsing query."
                    struc_query = update_b_with_non_empty_a(struc_query, struc_query_new)
                    clarify_count += 1
                    results = self.retriever._run(struc_query)
                    candidates = results["candidates"]
                else:
                    results["candidates"]=candidates[:self.max_candidates]
                    break
            else:
                break

        model_score = []
        for c in candidates:
            model_score.append({'model_name': c['model_name'], 'score': c['similarity']})

        # === Rank the candiates with LLM ===
        results["query"] = struc_query
        response = self.ranker._run(results)
        return response, model_score
