# import pandas as pd
# from ast import literal_eval
# import numpy as np

# # 读取CSV文件
# df = pd.read_csv('saves/qwen2_7b_xhs_benchmarks/AAAI/0722_multi_xhs_inst_qwen7b/lora/data_analysis/badcase_analysis_qwen7b.csv')

# # 初始化计数器和错误示例集合
# entity_recognition_error_count = 0
# classification_error_count = 0
# boundary_error_count = 0
# missing_entity_count = 0
# extra_entity_count = 0
# split_merge_error_count = 0

# boundary_errors = set()
# missing_entities_errors = set()
# extra_entities_errors = set()
# split_merge_errors = set()
# classification_errors = set()

# def safe_literal_eval(val):
#     if pd.isna(val):
#         return set()
#     return literal_eval(val)

# def categorize_errors(row):
#     global entity_recognition_error_count, classification_error_count
#     global boundary_error_count, missing_entity_count, extra_entity_count, split_merge_error_count
    
#     predicted_extra = safe_literal_eval(row["模型预测有,人工标注没有"])
#     missing_entities = safe_literal_eval(row["模型预测没有,人工标注有"])
#     correct_entities = safe_literal_eval(row["模型和人工都有"])

#     predicted_extra_set = set(predicted_extra)
#     missing_entities_set = set(missing_entities)
#     correct_entities_set = set(correct_entities)
    
#     # 检查分类错误
#     for missing_entity in list(missing_entities_set):
#         found_classification_error = False
#         for predicted_entity in list(predicted_extra_set):
#             if missing_entity[0] == predicted_entity[0] and missing_entity[1] != predicted_entity[1]:
#                 classification_error_count += 1
#                 classification_errors.add(predicted_entity)
#                 predicted_extra_set.remove(predicted_entity)
#                 missing_entities_set.remove(missing_entity)
#                 found_classification_error = True
#                 break
#         if found_classification_error:
#             continue  # Skip further checks if classification error is found

#     # 检查实体边界错误和实体拆分或合并错误
#     for missing_entity in list(missing_entities_set):
#         found_boundary_error = False
#         found_split_merge_error = False
#         for predicted_entity in list(predicted_extra_set):
#             if (missing_entity[0] in predicted_entity[0] or predicted_entity[0] in missing_entity[0]) and missing_entity[1] == predicted_entity[1]:
#                 boundary_error_count += 1
#                 boundary_errors.add((missing_entity, predicted_entity))
#                 predicted_extra_set.remove(predicted_entity)
#                 missing_entities_set.remove(missing_entity)
#                 found_boundary_error = True
#                 break
#             elif set(missing_entity[0].split()).issubset(set(predicted_entity[0].split())) or set(predicted_entity[0].split()).issubset(set(missing_entity[0].split())):
#                 split_merge_error_count += 1
#                 split_merge_errors.add((missing_entity, predicted_entity))
#                 predicted_extra_set.remove(predicted_entity)
#                 missing_entities_set.remove(missing_entity)
#                 found_split_merge_error = True
#                 break
#         if found_boundary_error or found_split_merge_error:
#             continue  # Skip further checks if boundary or split/merge error is found

#     # 实体识别错误
#     for missing_entity in missing_entities_set:
#         entity_recognition_error_count += 1
#         missing_entity_count += 1
#         missing_entities_errors.add(missing_entity)
    
#     for predicted_entity in predicted_extra_set:
#         entity_recognition_error_count += 1
#         extra_entity_count += 1
#         extra_entities_errors.add(predicted_entity)
            
# # 遍历每一行，分类错误
# df.apply(categorize_errors, axis=1)

# # 输出最终结果
# print(f"一类数量：{entity_recognition_error_count}")
# print(f"二类数量：{classification_error_count}")
# print(f"一类(实体边界错误): {boundary_error_count}")
# print(f"一类(实体缺失或遗漏): {missing_entity_count}")
# print(f"一类(多余实体): {extra_entity_count}")
# print(f"一类(实体拆分或合并错误): {split_merge_error_count}")

# # 输出每种错误的示例
# print("\n实体边界错误示例:")
# for example in list(boundary_errors)[:20]:
#     print(example)

# print("\n实体缺失或遗漏错误示例:")
# for example in list(missing_entities_errors)[:20]:
#     print(example)

# print("\n多余实体错误示例:")
# for example in list(extra_entities_errors)[:20]:
#     print(example)

# print("\n实体拆分或合并错误示例:")
# for example in list(split_merge_errors)[:20]:
#     print(example)

# print("\n分类错误示例:")
# for example in list(classification_errors)[:20]:
#     print(example)



import pandas as pd
from ast import literal_eval
import numpy as np

# 读取CSV文件
df = pd.read_csv('saves/badcase_analysis_qwen7b.csv')

# 初始化计数器和错误示例集合
entity_recognition_error_count = 0
classification_error_count = 0
boundary_error_count = 0
missing_entity_count = 0
extra_entity_count = 0
split_merge_error_count = 0

boundary_errors = []
missing_entities_errors = []
extra_entities_errors = []
split_merge_errors = []
classification_errors = []

def safe_literal_eval(val):
    if pd.isna(val):
        return set()
    return literal_eval(val)

def categorize_errors(row):
    global entity_recognition_error_count, classification_error_count
    global boundary_error_count, missing_entity_count, extra_entity_count, split_merge_error_count
    
    text = row["text"]
    predicted_extra = safe_literal_eval(row["模型预测有,人工标注没有"])
    missing_entities = safe_literal_eval(row["模型预测没有,人工标注有"])
    correct_entities = safe_literal_eval(row["模型和人工都有"])

    predicted_extra_set = set(predicted_extra)
    missing_entities_set = set(missing_entities)
    correct_entities_set = set(correct_entities)
    
    # 检查分类错误
    for missing_entity in list(missing_entities_set):
        found_classification_error = False
        for predicted_entity in list(predicted_extra_set):
            if missing_entity[0] == predicted_entity[0] and missing_entity[1] != predicted_entity[1]:
                classification_error_count += 1
                classification_errors.append((text, predicted_entity, missing_entity))
                predicted_extra_set.remove(predicted_entity)
                missing_entities_set.remove(missing_entity)
                found_classification_error = True
                break
        if found_classification_error:
            continue  # Skip further checks if classification error is found

    # 检查实体边界错误和实体拆分或合并错误
    for missing_entity in list(missing_entities_set):
        found_boundary_error = False
        found_split_merge_error = False
        for predicted_entity in list(predicted_extra_set):
            if (missing_entity[0] in predicted_entity[0] or predicted_entity[0] in missing_entity[0]) and missing_entity[1] == predicted_entity[1]:
                boundary_error_count += 1
                boundary_errors.append((text, missing_entity, predicted_entity))
                predicted_extra_set.remove(predicted_entity)
                missing_entities_set.remove(missing_entity)
                found_boundary_error = True
                break
            elif set(missing_entity[0].split()).issubset(set(predicted_entity[0].split())) or set(predicted_entity[0].split()).issubset(set(missing_entity[0].split())):
                split_merge_error_count += 1
                split_merge_errors.append((text, missing_entity, predicted_entity))
                predicted_extra_set.remove(predicted_entity)
                missing_entities_set.remove(missing_entity)
                found_split_merge_error = True
                break
        if found_boundary_error or found_split_merge_error:
            continue  # Skip further checks if boundary or split/merge error is found

    # 实体识别错误
    for missing_entity in missing_entities_set:
        entity_recognition_error_count += 1
        missing_entity_count += 1
        missing_entities_errors.append((text, missing_entity))
    
    for predicted_entity in predicted_extra_set:
        entity_recognition_error_count += 1
        extra_entity_count += 1
        extra_entities_errors.append((text, predicted_entity))
            
# 遍历每一行，分类错误
df.apply(categorize_errors, axis=1)

# 输出最终结果
print(f"一类数量：{entity_recognition_error_count}")
print(f"二类数量：{classification_error_count}")
print(f"一类(实体边界错误): {boundary_error_count}")
print(f"一类(实体缺失或遗漏): {missing_entity_count}")
print(f"一类(多余实体): {extra_entity_count}")
print(f"一类(实体拆分或合并错误): {split_merge_error_count}")

# 输出每种错误的示例
print("\n实体边界错误示例:")
for example in boundary_errors[:100]:
    print(f"文本: {example[0]}")
    print(f"错误对/正确对: {example[1]}, {example[2]}")

print("\n实体缺失或遗漏错误示例:")
for example in missing_entities_errors[:100]:
    print(f"文本: {example[0]}")
    print(f"错误对: {example[1]}")

print("\n多余实体错误示例:")
for example in extra_entities_errors[:100]:
    print(f"文本: {example[0]}")
    print(f"错误对: {example[1]}")

print("\n实体拆分或合并错误示例:")
for example in split_merge_errors[:100]:
    print(f"文本: {example[0]}")
    print(f"错误对/正确对: {example[1]}, {example[2]}")

print("\n分类错误示例:")
for example in classification_errors[:100]:
    print(f"文本: {example[0]}")
    print(f"错误对/正确对: {example[1]}, {example[2]}")
