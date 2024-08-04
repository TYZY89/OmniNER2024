import json

import pandas as pd

test_data = json.load(open("data/test.json", "r"))
print(len(test_data))
text_list = []
pred_list = []
label_list = []
FP_list = []
FN_list = []
TP_list = []
with open('saves/lora/predict/generated_predictions.jsonl', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    assert len(lines) == len(test_data)

    results = [json.loads(line.replace('\\\\','')) for line in lines]
    for d ,t in zip(results,test_data):
        label=d['label']
        predict= d['predict']
        text = t['input']
        str_false_positives=[]
        str_false_negatives=[]
        try:
            true_entities = json.loads(label)
            # print(true_entities)
            predicted_entities = json.loads(predict)

            label_set = set(tuple(entity) for entity in true_entities)
            predict_set = set(tuple(entity) for entity in predicted_entities)
            print(text, '\n人工标注结果:', label_set, '\n模型预测结果:', predict_set)
            true_positives = label_set.intersection(predict_set)
            false_positives = predict_set - label_set
            false_negatives = label_set - predict_set
            #set转list
            str_false_positives = [str(item) for item in false_positives]
            str_false_negatives = [str(item) for item in false_negatives]
            # print(false_negatives,false_positives)
            print('预测错误的属性：',''.join((str_false_positives)))
            print('模型遗漏的属性：',''.join((str_false_negatives)))
            print()
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            # 打印错误堆栈
            print(predict)
            continue  # 如果JSON解码失败，则跳过当前样本

        # print(type(label))
        # print(type(predict))
        # print(type(str_false_negatives))
        # print(type(str_false_positives))

        text_list.append(text)
        pred_list.append(label)
        label_list.append(predict)
        # FP_list.append(str_false_positives)
        # FN_list.append(str_false_negatives)
        FP_list.append(false_positives if len(false_positives)>0 else '')
        FN_list.append(false_negatives if len(false_negatives)>0 else '' )
        TP_list.append(true_positives if len(true_positives)>0 else '')

assert len(text_list) == len(pred_list) == len(label_list)==len(FP_list) == len(FN_list)==len(TP_list)

print(len(text_list))
data = pd.DataFrame({'text': text_list,'模型预测有,人工标注没有':FP_list,'模型预测没有,人工标注有':FN_list, '模型和人工都有':TP_list})
data = data[~((data['模型预测有,人工标注没有'] == '') & (data['模型预测没有,人工标注有'] == ''))]
print(data.shape)

data.to_csv('saves/badcase_analysis_qwen7b.csv',index=False, header=True, encoding='utf-8-sig')