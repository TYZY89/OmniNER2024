import json
import pandas as pd
import argparse
from collections import defaultdict

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def evaluate_extraction(label_data, predict_data):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for label_entities, predict_entities in zip(label_data, predict_data):
        label_set = set(tuple(entity) for entity in label_entities)
        predict_set = set(tuple(entity) for entity in predict_entities)

        true_positives += len(label_set.intersection(predict_set))
        false_positives += len(predict_set - label_set)
        false_negatives += len(label_set - predict_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

def evaluate_extraction_per_attribute_v2(label_data, predict_data):
    attribute_metrics = defaultdict(lambda: {"true_positives": 0, "false_positives": 0, "false_negatives": 0})

    for label_entities, predict_entities in zip(label_data, predict_data):
        label_set = set(tuple(entity) for entity in label_entities)
        predict_set = set(tuple(entity) for entity in predict_entities)

        for entity in label_set:
            if len(entity) > 1:
                if entity in predict_set:
                    attribute_metrics[entity[1]]["true_positives"] += 1
                else:
                    attribute_metrics[entity[1]]["false_negatives"] += 1
            else:
                print("Invalid entity in label_set:", entity)

        for entity in predict_set:
            if len(entity) > 1:
                if entity not in label_set:
                    attribute_metrics[entity[1]]["false_positives"] += 1
            else:
                print("Invalid entity in predict_set:", entity)

    attribute_scores = {}
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    for attribute, metrics in attribute_metrics.items():
        true_positives = metrics["true_positives"]
        false_positives = metrics["false_positives"]
        false_negatives = metrics["false_negatives"]

        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        attribute_scores[attribute] = {"precision": precision, "recall": recall, "f1_score": f1_score, "support": int(true_positives + false_negatives)}

    df = pd.DataFrame(attribute_scores).T
    df = df[['precision', 'recall', 'f1_score', 'support']]  # 调整列的顺序

    total_support = df['support'].sum()
    weights = df['support'] / total_support
    weighted_precision = (df['precision'] * weights).sum()
    weighted_recall = (df['recall'] * weights).sum()
    weighted_f1_score = (df['f1_score'] * weights).sum()

    micro_precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    micro_recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    micro_f1_score = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    macro_precision = df['precision'].mean()
    macro_recall = df['recall'].mean()
    macro_f1_score = df['f1_score'].mean()

    df.loc['micro avg'] = [micro_precision, micro_recall, micro_f1_score, total_true_positives + total_false_negatives]
    df.loc['macro avg'] = [macro_precision, macro_recall, macro_f1_score, total_true_positives + total_false_negatives]
    df.loc['weighted avg'] = [weighted_precision, weighted_recall, weighted_f1_score, total_support]

    print(df)
    return df

def try_fix_json(json_string):
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        try:
            fixed_json_string = json_string.replace('\\"', '"').replace('"{', '{').replace('}"', '}')
            return json.loads(fixed_json_string)
        except json.JSONDecodeError:
            try:
                fixed_json_string = json_string.replace('\'', '"')
                return json.loads(fixed_json_string)
            except json.JSONDecodeError:
                print("无法修复的 JSON 字符串:", json_string)
                return None

def validate_and_fix_entity(entity):
    # Ensure the entity is a tuple and has exactly 2 elements
    if isinstance(entity, tuple) and len(entity) == 2:
        return entity
    elif isinstance(entity, list) and len(entity) == 2:
        return tuple(entity)
    elif isinstance(entity, str):
        parts = entity.split(", ")
        if len(parts) == 2:
            return (parts[0].strip().strip("()").strip('"').strip("'"), parts[1].strip().strip("()").strip('"').strip("'"))
    print("Invalid entity format:", entity)
    return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate entity extraction results")
    parser.add_argument('--path', type=str, required=True, help='Path to the predictions file')
    args = parser.parse_args()

    # Load data and prepare label and prediction lists
    label_data, predict_data = [], []

    with open(args.path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        for line in lines:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                print("JSONDecodeError:", line)
                continue

            label = d['label']
            predict = d['predict']
            true_entities = try_fix_json(label)
            predicted_entities = try_fix_json(predict)

            if true_entities is None or predicted_entities is None:
                continue

            true_entities = [validate_and_fix_entity(entity) for entity in true_entities]
            predicted_entities = [validate_and_fix_entity(entity) for entity in predicted_entities]

            # Remove None entities resulting from validation
            true_entities = [entity for entity in true_entities if entity is not None]
            predicted_entities = [entity for entity in predicted_entities if entity is not None]

            label_data.append(true_entities)
            predict_data.append(predicted_entities)

    # Perform evaluations
    precision, recall, f1_score = evaluate_extraction(label_data, predict_data)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)
    print()

    df = evaluate_extraction_per_attribute_v2(label_data, predict_data)

if __name__ == "__main__":
    main()