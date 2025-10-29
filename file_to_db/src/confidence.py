from collections import Counter
import numpy as np
from typing import List, Dict, Any, Type, Optional, Union
from pydantic import BaseModel, Field, HttpUrl
import math

class ExtractedField(BaseModel):
    value: Any
    confidence: float

import math
from collections import Counter
from typing import Any, Dict, List, Type, Union
from pydantic import BaseModel

class ConfidenceCalculator:
    def __init__(self, w_cons=0.3, w_logp=0.7):
        self.w_cons = w_cons
        self.w_logp = w_logp

    def _normalize_logprob(self, lp: Union[float, None]) -> float:
        if lp is None:
            return 0.0
        try:
            per_token_prob = math.exp(lp)
        except:
            return 0.0
        return max(min(per_token_prob, 1.0), 0.0)

    def process(
        self,
        schema_cls: Type[BaseModel],
        gens: List[Dict[str, Any]],
        logprobs_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return self._process_fields(schema_cls, gens, logprobs_list)

    def _process_fields(
        self,
        schema_cls: Type[BaseModel],
        gens: List[Dict[str, Any]],
        logprobs_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        results = {}
        field_defs = schema_cls.model_fields

        for field_name in field_defs:
            field_info = field_defs[field_name]
            field_metadata = (field_info.json_schema_extra or {}).get("metadata", {})
            is_free_text = field_metadata.get("free_text", False)

            vals = [g.get(field_name) for g in gens]
            lps = [lp.get(field_name) for lp in logprobs_list]

            # All invalid sentinel
            if all(lp == 1000 for lp in lps):
                results[field_name] = {"value": vals[0], "confidence": 0.0}
                continue

            # Nested list of models
            if (
                isinstance(vals[0], list)
                and vals[0]
                and isinstance(vals[0][0], dict)
                and hasattr(field_info.annotation, "__origin__")
                and issubclass(field_info.annotation.__origin__, list)
                and issubclass(field_info.annotation.__args__[0], BaseModel)
            ):
                sub_model = field_info.annotation.__args__[0]
                chosen_index = 0  # could score which list to pick
                chosen_list = vals[chosen_index]
                chosen_lps = lps[chosen_index]

                nested_items = []
                for idx, item_dict in enumerate(chosen_list):
                    lp_dict = (
                        chosen_lps[idx]
                        if isinstance(chosen_lps, list) and idx < len(chosen_lps)
                        else {}
                    )
                    nested_res = self._process_fields(
                        sub_model, [item_dict], [lp_dict]
                    )
                    nested_items.append(nested_res)
                results[field_name] = nested_items
                continue

            # Nested single model
            if isinstance(vals[0], dict) and issubclass(field_info.annotation, BaseModel):
                nested_res = self._process_fields(field_info.annotation, [vals[0]], [lps[0]])
                results[field_name] = nested_res
                continue

            # List of scalars
            if isinstance(vals[0], list):
                # Count frequencies across all generations
                print(field_name)
                print(vals)
                all_items = []
                for v in vals:
                    all_items.extend(v)
                item_counts = Counter(all_items)

                list_scores = []
                for i, vlist in enumerate(vals):
                    confs = []
                    lp_list = lps[i] if isinstance(lps[i], list) else [-5.0] * len(vlist)
                    for item, lp in zip(vlist, lp_list):
                        self_c = item_counts[item] / len(vals)
                        norm_lp = self._normalize_logprob(lp)
                        conf = self.w_cons * self_c + self.w_logp * norm_lp
                        confs.append(conf)
                    avg_conf = sum(confs) / len(confs) if confs else 0.0
                    list_scores.append(avg_conf)
                chosen_index = max(range(len(vals)), key=lambda i: list_scores[i])
                vlist = vals[chosen_index]
                lp_list = (
                    lps[chosen_index] if isinstance(lps[chosen_index], list) else [-5.0] * len(vlist)
                )

                item_results = []
                for item, lp in zip(vlist, lp_list):
                    self_c = item_counts[item] / len(vals)
                    norm_lp = self._normalize_logprob(lp)
                    conf = round(self.w_cons * self_c + self.w_logp * norm_lp, 4)
                    item_results.append({"value": item, "confidence": conf})
                results[field_name] = {"value": item_results}
                continue

            # Scalar
            if isinstance(vals[0], (str, int, float)):
                if is_free_text:
                    logps_for_field = [lp if isinstance(lp, float) else -5.0 for lp in lps]
                    chosen_index = max(range(len(vals)), key=lambda i: logps_for_field[i])
                    norm_lp = self._normalize_logprob(lps[chosen_index])
                    conf = round(self.w_logp * norm_lp + self.w_cons * 1.0, 4)
                    results[field_name] = {"value": vals[chosen_index], "confidence": conf}
                else:
                    counts = Counter(vals)
                    most_common, count = counts.most_common(1)[0]
                    chosen_index = next(i for i, v in enumerate(vals) if v == most_common)
                    norm_lp = self._normalize_logprob(
                        lps[chosen_index] if isinstance(lps[chosen_index], float) else None
                    )
                    self_c = count / len(vals)
                    conf = round(self.w_cons * self_c + self.w_logp * norm_lp, 4)
                    results[field_name] = {"value": most_common, "confidence": conf}
                continue

            # Fallback
            results[field_name] = {"value": vals[0], "confidence": 0.0}

        return results


    # def process(
    #     self,
    #     schema_cls: Type[BaseModel],
    #     gens: List[Dict[str, Any]],
    #     logprobs_list: List[Dict[str, Any]]
    # ) -> BaseModel:
    #     results = {}
    #     #field_defs = schema_cls.__fields__
    #     field_defs = schema_cls.model_fields

    #     #print("+++++++++++++")
    #     #print(logprobs_list)
    #     print(gens[0])
    #     for field_name in gens[0].keys():
    #         field_info = field_defs[field_name]
    #         print(field_name)
    #         print(field_info)
    #         print(field_info.json_schema_extra)
    #         if field_info.json_schema_extra:
    #             is_free_text = field_info.json_schema_extra.get('metadata').get("free_text", False)
    #         else:
    #             is_free_text = False
    #         #field_info = field_defs[field_name].field_info
    #         #is_free_text = field_info.extra.get("free_text", False)

    #         # Collect all values for this field
    #         print("--------------------")
    #         print(gens)
    #         vals = [g.get(field_name) for g in gens]
    #         print(field_info)
    #         print(field_name)
    #         print(vals)
    #         print(is_free_text)

    #         if logprobs_list[0].get(field_name) == 1000:
    #             results[field_name] = {"value": vals[0], "confidence": 0.0}
    #         else:
    #             if is_free_text:
    #                 # NEW: pick generation with highest logprob
    #                 logprobs_for_field = []
    #                 for i in range(len(vals)):
    #                     lp_entry = logprobs_list[i].get(field_name)
    #                     if isinstance(lp_entry, float):
    #                         logprobs_for_field.append(lp_entry)
    #                     else:
    #                         logprobs_for_field.append(-5.0)
    #                 chosen_index = max(range(len(vals)), key=lambda i: logprobs_for_field[i])
    #                 chosen_value = vals[chosen_index]
    #                 self_c = 1.0
    #             else:
    #                 # For scalar fields (string/int)
    #                 if isinstance(vals[0], (str, int, float)):
    #                     counts = Counter(vals)
    #                     most_common_value, _ = counts.most_common(1)[0]
    #                     chosen_index = next(i for i, v in enumerate(vals) if v == most_common_value)
    #                     chosen_value = vals[chosen_index]
    #                     self_c = counts[most_common_value] / len(vals)

    #                 # For list fields
    #                 elif isinstance(vals[0], list):
    #                     # Flatten all items to count frequencies
    #                     all_items = []
    #                     for v in vals:
    #                         all_items.extend(v)
    #                     item_counts = Counter(all_items)

    #                     # For each generation, compute the average confidence of the list
    #                     list_scores = []
    #                     list_confs = []
    #                     for i, vlist in enumerate(vals):
    #                         item_confs = []
    #                         lp_entry = logprobs_list[i].get(field_name)                 
    #                         lp_list = lp_entry if isinstance(lp_entry, list) else [-5.0] * len(vlist)

    #                         for item, lp in zip(vlist, lp_list):
    #                             item_self_c = item_counts[item] / len(vals)
    #                             norm_lp = self._normalize_logprob(lp)
    #                             item_conf = self.w_cons * item_self_c + self.w_logp * norm_lp
    #                             item_confs.append(item_conf)

    #                         avg_conf = round(sum(item_confs) / len(item_confs), 4) if item_confs else 0.0
    #                         list_confs.append(item_confs)
    #                         list_scores.append(avg_conf)

    #                     # Choose the generation with the highest average confidence
    #                     chosen_index = max(range(len(vals)), key=lambda i: list_scores[i])
    #                     chosen_confs = list_confs[chosen_index]
    #                     chosen_value = vals[chosen_index]

    #                     # Also precompute per-item counts for later
    #                     #chosen_lp_entry = logprobs_list[chosen_index].get(field_name)
    #                     #chosen_lp_list = chosen_lp_entry["value"] if (isinstance(chosen_lp_entry, dict) and isinstance(chosen_lp_entry.get("value"), list)) else [-5.0]*len(chosen_value)

    #                 else:
    #                     # fallback
    #                     chosen_value = vals[0]
    #                     chosen_index = 0
    #                     self_c = 1.0

    #             # Use logprobs of chosen generation
    #             chosen_logp = logprobs_list[chosen_index].get(field_name)

    #             # Handle scalar value
    #             if isinstance(chosen_value, (str, int, float)):
    #                 #print(chosen_logp)
    #                 if isinstance(chosen_logp, float):
    #                     norm_lp = self._normalize_logprob(chosen_logp)
    #                 else:
    #                     norm_lp = 0.0
    #                 print("+++++++++")
    #                 print(logprobs_list)
    #                 print(chosen_index)
    #                 print(chosen_logp)
    #                 print(norm_lp)
    #                 print(self_c)
    #                 conf = round(
    #                     self.w_logp * norm_lp + self.w_cons * self_c, 4
    #                 )
    #                 #results[field_name] = ExtractedField(value=chosen_value, confidence=conf)
    #                 results[field_name] = {"value": chosen_value, "confidence": conf}

    #             # Handle list of values
    #             elif isinstance(chosen_value, list):
    #                 items = []
    #                 for item, lp in zip(chosen_value, chosen_confs):
    #                     items.append({"value": item, "confidence": lp})
    #                 results[field_name] = {"value": items}

    #             else:
    #                 # fallback
    #                 #results[field_name] = ExtractedField(value=chosen_value, confidence=0.0)
    #                 results[field_name] = {"value": chosen_value, "confidence": 0.0}
    #     print(results)
    #     return schema_cls(**results)

