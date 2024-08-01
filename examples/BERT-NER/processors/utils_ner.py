import csv
import json
import torch
from transformers import BertTokenizer

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_text(self,input_file):
        lines = []
        with open(input_file,'r') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words":words,"labels":labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words":words,"labels":labels})
        return lines

    @classmethod
    # def _read_json(self,input_file):
    #     lines = []
    #     with open(input_file,'r') as f:
    #         for line in f:
    #             line = json.loads(line.strip())
    #             text = line['text']
    #             label_entities = line.get('label',None)
    #             words = list(text)
    #             labels = ['O'] * len(words)
    #             if label_entities is not None:
    #                 for key,value in label_entities.items():
    #                     for sub_name,sub_index in value.items():
    #                         for start_index,end_index in sub_index:
    #                             assert  ''.join(words[start_index:end_index+1]) == sub_name
    #                             if start_index == end_index:
    #                                 labels[start_index] = 'S-'+key
    #                             else:
    #                                 labels[start_index] = 'B-'+key
    #                                 labels[start_index+1:end_index+1] = ['I-'+key]*(len(sub_name)-1)
    #             lines.append({"words": words, "labels": labels})
    #     return lines

    def _read_json(self, input_file):
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


def get_entity_bios(seq,id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entity_bio(seq,id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entities(seq,id2label,markup='bios'):
    '''
    :param seq:
    :param id2label:
    :param markup:
    :return:
    '''
    assert markup in ['bio','bios']
    if markup =='bio':
        return get_entity_bio(seq,id2label)
    else:
        return get_entity_bios(seq,id2label)

def bert_extract_item(start_logits, end_logits):
    S = []
    start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0][1:-1]
    end_pred = torch.argmax(end_logits, -1).cpu().numpy()[0][1:-1]
    for i, s_l in enumerate(start_pred):
        if s_l == 0:
            continue
        for j, e_l in enumerate(end_pred[i:]):
            if s_l == e_l:
                S.append((s_l, i, i + j))
                break
    return S
