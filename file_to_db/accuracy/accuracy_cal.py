from openai import OpenAI
import numpy as np
import json
import re

client = OpenAI(api_key="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", base_url="https://api.openai.com/v1/")

# Embedding cache to avoid repeated API calls
embedding_cache = {}

def get_embedding(text):
    """
    Get embedding for a text, using cache if available.
    """
    text = text.strip().lower()
    if text in embedding_cache:
        return embedding_cache[text]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    emb = response.data[0].embedding
    embedding_cache[text] = emb
    return emb

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def string_similarity(a, b):
    """
    Compute embedding-based similarity between two strings.
    """
    emb1 = get_embedding(a)
    emb2 = get_embedding(b)
    sim = cosine_similarity(emb1, emb2)
    return sim

def extract_pred_value(pred_item):
    if isinstance(pred_item, dict) and "value" in pred_item:
        return pred_item["value"]
    return pred_item

def compare_values(gt_value, pred_value):
    if pred_value is None:
        return 0.0
    if isinstance(gt_value, str) and isinstance(pred_value, str):
        gt_clean = re.sub(r"\s*\(.*?\)\s*", "", gt_value).strip().lower()
        pred_clean = re.sub(r"\s*\(.*?\)\s*", "", pred_value).strip().lower()
        if gt_clean == pred_clean:
            return 1.0
        else:
            return string_similarity(gt_clean, pred_clean)
    elif isinstance(gt_value, (int, float)) and isinstance(pred_value, (int, float)):
        return 1.0 if gt_value == pred_value else 0.0
    elif isinstance(gt_value, bool) and isinstance(pred_value, bool):
        return 1.0 if gt_value == pred_value else 0.0
    elif isinstance(gt_value, list) and isinstance(pred_value, list):
        return compare_list(gt_value, pred_value)
    elif isinstance(gt_value, dict) and isinstance(pred_value, dict):
        return compare_dicts(gt_value, pred_value)
    else:
        return 0.0

def compare_list(gt_list, pred_list):
    if not gt_list and not pred_list:
        return 1.0
    if not gt_list or not pred_list:
        return 0.0

    pred_unwrapped = [extract_pred_value(p) for p in pred_list]
    pred_remaining = pred_unwrapped.copy()

    gt_scores = []
    for gt_item in gt_list:
        best_score = 0.0
        best_idx = None
        for idx, pred_item in enumerate(pred_remaining):
            score = compare_values(gt_item, pred_item)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None:
            pred_remaining.pop(best_idx)
        gt_scores.append(best_score)
    recall = sum(gt_scores) / len(gt_scores)

    gt_remaining = gt_list.copy()
    pred_scores = []
    for pred_item in pred_unwrapped:
        best_score = 0.0
        best_idx = None
        for idx, gt_item in enumerate(gt_remaining):
            score = compare_values(gt_item, pred_item)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None:
            gt_remaining.pop(best_idx)
        pred_scores.append(best_score)
    precision = sum(pred_scores) / len(pred_scores)

    return (recall + precision) / 2

def compare_dicts(gt_dict, pred_dict):
    scores = []
    for key in gt_dict:
        gt_value = gt_dict[key]
        pred_entry = pred_dict.get(key)

        if isinstance(pred_entry, dict) and "value" in pred_entry:
            pred_value = pred_entry["value"]
        else:
            pred_value = pred_entry

        score = compare_values(gt_value, pred_value)
        print("-------------", file=f_log)
        print(key, file=f_log)
        print(gt_value, file=f_log)
        print(pred_value, file=f_log)
        print(score, file=f_log)
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0

def compute_prediction_accuracy(gt_json, pred_json):
    return compare_dicts(gt_json, pred_json)

if __name__ == "__main__":
    with open("ground_truth.json") as f:
        gt_json = json.load(f)
    with open("predictions.json") as f:
        pred_json = json.load(f)
    
    f_log = open("output.txt", "w")
    accuracy = compute_prediction_accuracy(gt_json, pred_json)
    print(f"Prediction accuracy: {accuracy:.4f}")
