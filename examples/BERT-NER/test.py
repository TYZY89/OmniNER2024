import json
import os

def _read_json(input_file):
    def clean_text(text):
        # 去除特殊字符，只保留字母、数字、中文和基本标点符号
        return ''.join([c for c in text if c.isalnum() or c in "，。！？、；：(){}[]<>《》【】" or c.isspace()])

    lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            text = line['text']
            clean_txt = clean_text(text)
            label_entities = line.get('label', None)
            words = list(clean_txt)
            labels = ['O'] * len(words)
            if label_entities is not None:
                for key, value in label_entities.items():
                    for sub_name, sub_index in value.items():
                        for start_index, end_index in sub_index:
                            # 调整索引以适应清理后的文本
                            original_substring = text[start_index:end_index+1]
                            clean_substring = clean_text(original_substring)
                            new_start_index = clean_txt.find(clean_substring)
                            new_end_index = new_start_index + len(clean_substring) - 1
                            
                            if new_start_index == -1 or new_end_index >= len(words):
                                print(f"Index adjustment error: original '{original_substring}', clean '{clean_substring}'")
                                print(f"Full text: {text}")
                                print(f"Clean text: {clean_txt}")
                                continue

                            extracted_substring = ''.join(words[new_start_index:new_end_index+1])
                            if extracted_substring != clean_substring:
                                print(f"Mismatch found: extracted '{extracted_substring}' != expected '{clean_substring}'")
                                print(f"Full text: {text}")
                                print(f"Words: {words}")
                                print(f"Start index: {new_start_index}, End index: {new_end_index}")
                                continue  # 跳过有问题的数据
                            
                            if new_start_index < len(labels) and new_end_index < len(labels):
                                try:
                                    if new_start_index == new_end_index:
                                        labels[new_start_index] = 'S-' + key
                                    else:
                                        labels[new_start_index] = 'B-' + key
                                        labels[new_start_index+1:new_end_index+1] = ['I-' + key] * (new_end_index - new_start_index)
                                except IndexError as e:
                                    print(f"Error assigning labels: {e}")
                                    print(f"Start index: {new_start_index}, End index: {new_end_index}")
                                    print(f"Labels length: {len(labels)}, Words length: {len(words)}")
                                    print(f"Full text: {text}")
                                    print(f"Clean text: {clean_txt}")
                                    continue
                            else:
                                print(f"Index out of range when assigning labels: start_index={new_start_index}, end_index={new_end_index}, labels_length={len(labels)}")
            lines.append({"words": words, "labels": labels})
    return lines

# 测试
input_file = 'datasets/omniner/dev.json'  # 替换为你的文件路径
_read_json(input_file)

