# -*- coding: utf-8 -*-
"""
Created on: 2024/07/19 11:12
Author: zhouyong
Description: 
"""
import json
import numpy as np
import pandas as pd

entity_type_set = set()
instance_list = []
note_id_set = set()

# instruction = " "
instruction = '''
你是一个先进的人工智能模型，专门用于识别文本中的实体属性。
命名实体包括 '时间_日维度时间', '时间_节假日', '气味/味道_味道', '产品系列', '规则/玩法', '场景_娱乐场景', '方法/技能_技能', '品牌', '产品特性', '赛事/展演_体育赛事', '方法/技能_其他', '风格_建筑风格', '场所', '风格_产品艺术风格', '款式/形态_其他', '特定人物_主播网红', '人群/宠物_身体特征', '行为_心理/情绪', '品类_产品', '场景_民俗场景', 'IP_其他', 'aoi_产地', '赛事/展演_文娱展演', '风格_妆造风格', '气味/味道_口感', '人群/宠物_肤质', '品类_其他', '款式/形态_包装', 'IP_游戏', '特定人物_其他', '款式/形态_款式外形', '气味/味道_气味', '人群/宠物_性别', '时间_其他', '功能/功效_其他', '价格/金额_价格', '品类_专业/学科', 'IP_角色', '方法/技能_方法', '场景_学习场景', '负向效果', '场景_其他', '赛事/展演_其他赛事', '品类_服务', '数量', '资质等级', '体验_其他', '人群/宠物_其他', '功能/功效_人体功效', '人群/宠物_职业', '人群/宠物_教育阶段', '体验_环境体验', '场景_生活场景', 'IP_创作作品', '体验_使用体验', '工艺', '行为_操作行为', '图案', '营销/活动', '风格_其他', '功能/功效_生物功效', '时间_年/月维度时间', '特定人物_明星名人', '赛事/展演_学术赛事', '材质', '攻略', '行为_其他', '价格/金额_其他', '场景_虚拟场景', '场景_工作场景', '行为_活动', '行为_交互行为', '赛事/展演_游戏赛事', '产品规格', 'poi_景点', '品类_品种', '价格/金额_价格描述', '人群/宠物_身份', '人群/宠物_年龄', '颜色', '人群/宠物_健康痛点', 'poi_其他', '时间_时间周期', '行为_交易类型', '时间_季节', '场景_装修场景', '成分', '体验_服务体验', '功能/功效_产品功能', 'aoi_其他'。
你的任务是根据给定的文本内容，准确地识别和分类这些实体。

指令：
仔细阅读提供的文本。
识别文本中的所有命名实体。
将每个实体分类为预定义的类别之一。
如果文本中没有可识别的命名实体，返回一个空列表。

输入格式：
文本数据，可能来自不同的领域和背景。
输出格式：一个包含命名实体及其类别的列表。每个实体应以[实体, 类别]的格式返回，例如：["北京", "地点"]。
示例：
输入文本："五天四晚亲子游|教你如何轻松玩转内蒙?  内蒙古素有“草原天堂”之称  一望无际的草原?"
输出实体列表：[[
                "五天四晚",
                "时间_时间周期"
            ],
            [
                "亲子游",
                "品类_服务"
            ],
            [
                "内蒙",
                "aoi_其他"
            ],
            [
                "内蒙古",
                "aoi_其他"
            ],
            [
                "草原",
                "场所"
            ],
            [
                "教你如何轻松玩转内蒙",
                "攻略"
            ]]

特别说明：
确保识别的准确性和类别的正确性。
对于模糊不清或可能存在歧义的实体，不输出。
不需要对文本进行总结或解释，只需专注于命名实体的识别和分类。
下面是输入的数据，请给出输出结果:
'''.replace('\n', '').replace(' ','').replace('\t','')

source_data = pd.read_csv('data/entity_instruct_merge_train_0708_choose_100.csv', header=0, encoding='utf-8')
print('原始数据条数：', source_data.shape[0])

# 保留sentence不为空的数据
source_data = source_data.dropna(subset=['sentence'])
print('sentence不为空的数据条数：', source_data.shape[0])

# 遍历每一行
for index, row in source_data.iterrows():
    # 获取当前行数据
    sentence = row['sentence']
    note_id = row['note_id']
    first_label_name = row.get('first_label_name', '')
    label = row.get('label', '')

    instance = {}
    instance['note_id'] = note_id
    instance['instruction'] = instruction
    instance['input'] = sentence

    if pd.isna(label) or label == '':
        instance['output'] = '[]'
        instance_list.append(instance)
        note_id_set.add(note_id)
        continue

    try:
        enty_jo = json.loads(label)
        outputs = []
        for jo in enty_jo:
            entity = jo['text'].replace('  ', ' ').replace('\t', ' ')
            if ''.join(entity.split()) not in ''.join(sentence.split()):
                print(entity, '\t', sentence)
                continue
            entity_type = jo['labels'][0]
            entity_type_set.add(entity_type)
            outputs.append([entity, entity_type])

        instance['output'] = json.dumps(outputs, ensure_ascii=False)
        instance_list.append(instance)
        note_id_set.add(note_id)
    except json.JSONDecodeError as e:
        print("JSON解析错误:", e)
        print(note_id, sentence, label)
    except Exception as e:
        print("发生异常:", e)
        print(note_id, sentence, label)

print(len(instance_list))
print('唯一note_id数量：', len(note_id_set))

# 保存为一个JSON文件
output_file = 'data/output.json'
json.dump(instance_list, open(output_file, 'w'), ensure_ascii=False, indent=4)

print('实体类型个数', len(entity_type_set))
print(entity_type_set)

# 统计instance的平均长度
instance_len = []
for i in instance_list:
    instance_len.append(len(json.dumps(i, ensure_ascii=False)))
# 统计instance_len的平均长度，最大值，最小值，90%分位点
print('instance_len的平均长度', sum(instance_len) / len(instance_list))
print('instance_len的最大值', max(instance_len))
print('instance_len的最小值', min(instance_len))
print('instance_len的90%分位点', np.percentile(instance_len, 90))

# # shuffle 打乱数据
# np.random.seed(10)
# np.random.shuffle(instance_list)
# train_size = int(len(instance_list) * 0.9)
# train_data = instance_list[:train_size]
# test_data = instance_list[train_size:]
# print('train_size=', train_size, 'test_size=', len(test_data))
# json.dump(train_data, open('data/xhs_entity_instruct_train_0708/xhs_entity_instructUIE_train_0708_plus_note_id.json', 'w'), ensure_ascii=False, indent=4)
# json.dump(test_data, open('data/xhs_entity_instruct_train_0708/xhs_entity_instructUIE_test_0708_plus_note_id.json', 'w'), ensure_ascii=False, indent=4)
# print('实体类型个数', len(entity_type_set))
# print(entity_type_set)

# # 统计instance的平均长度
# instance_len = []
# for i in instance_list:
#     instance_len.append(len(json.dumps(i, ensure_ascii=False)))
# # 统计instance_len的平均长度，最大值，最小值，90%分位点
# print('instance_len的平均长度', sum(instance_len) / len(instance_list))
# print('instance_len的最大值', max(instance_len))
# print('instance_len的最小值', min(instance_len))
# print('instance_len的90%分位点', np.percentile(instance_len, 90))