""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """
import torch
import logging
import os
import copy
import json
from .utils_ner import DataProcessor,get_entities
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, subject):
        self.guid = guid
        self.text_a = text_a
        self.subject = subject
    def __repr__(self):
        return str(self.to_json_string())
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeature(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_len, segment_ids, start_ids,end_ids, subjects):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_ids = start_ids
        self.input_len = input_len
        self.end_ids = end_ids
        self.subjects = subjects

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_input_mask, all_segment_ids, all_start_ids,all_end_ids,all_lens = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_input_mask = all_input_mask[:, :max_len]
    all_segment_ids = all_segment_ids[:, :max_len]
    all_start_ids = all_start_ids[:,:max_len]
    all_end_ids = all_end_ids[:, :max_len]
    return all_input_ids, all_input_mask, all_segment_ids, all_start_ids,all_end_ids,all_lens

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,
                                 cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=1,
                                 sep_token="[SEP]", pad_on_left=False, pad_token=0, pad_token_segment_id=0,
                                 sequence_a_segment_id=0, mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        if isinstance(example.text_a, list):
            example.text_a = " ".join(example.text_a)
        tokens = tokenizer.tokenize(example.text_a)
        label_ids = [label_map[x] for x in example.labels]
        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # Add special tokens
        tokens += [sep_token]
        label_ids += [label_map['O']]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [label_map['O']]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [label_map['O']] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token] * padding_length

        # 确保 label_ids 的长度与 max_seq_length 一致
        if len(label_ids) > max_seq_length:
            label_ids = label_ids[:max_seq_length]
        elif len(label_ids) < max_seq_length:
            label_ids += [pad_token] * (max_seq_length - len(label_ids))

        # 添加调试信息，确保长度一致
        if len(input_ids) != max_seq_length or len(input_mask) != max_seq_length or len(segment_ids) != max_seq_length or len(label_ids) != max_seq_length:
            logger.error(f"Length mismatch detected!")
            logger.error(f"input_ids: {len(input_ids)}, expected: {max_seq_length}")
            logger.error(f"input_mask: {len(input_mask)}, expected: {max_seq_length}")
            logger.error(f"segment_ids: {len(segment_ids)}, expected: {max_seq_length}")
            logger.error(f"label_ids: {len(label_ids)}, expected: {max_seq_length}")
            logger.error(f"tokens: {tokens}")
            logger.error(f"labels: {label_ids}")

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len,
                                      segment_ids=segment_ids, label_ids=label_ids))
    return features

class CnerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.char.bmes")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.char.bmes")), "test")

    def get_labels(self):
        """See base class."""
        return ["O", "CONT", "ORG","LOC",'EDU','NAME','PRO','RACE','TITLE']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-','I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            subject = get_entities(labels,id2label=None,markup='bios')
            examples.append(InputExample(guid=guid, text_a=text_a, subject=subject))
        return examples

class CluenerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["O", "address", "book","company",'game','government','movie','name','organization','position','scene']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            labels = line['labels']
            subject = get_entities(labels,id2label=None,markup='bios')
            examples.append(InputExample(guid=guid, text_a=text_a, subject=subject))
        return examples

class OmninerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["O", '时间_日维度时间','时间_节假日','气味/味道_味道','产品系列','规则/玩法','场景_娱乐场景','方法/技能_技能','品牌','产品特性','赛事/展演_体育赛事','方法/技能_其他','风格_建筑风格','场所','风格_产品艺术风格','款式/形态_其他','特定人物_主播网红','人群/宠物_身体特征','行为_心理/情绪','品类_产品','场景_民俗场景','IP_其他','aoi_产地','赛事/展演_文娱展演','风格_妆造风格','气味/味道_口感','人群/宠物_肤质','品类_其他','款式/形态_包装','IP_游戏','特定人物_其他','款式/形态_款式外形','气味/味道_气味','人群/宠物_性别','时间_其他','功能/功效_其他','价格/金额_价格','品类_专业/学科','IP_角色','方法/技能_方法','场景_学习场景','负向效果','场景_其他','赛事/展演_其他赛事','品类_服务','数量','资质等级','体验_其他','人群/宠物_其他','功能/功效_人体功效','人群/宠物_职业','人群/宠物_教育阶段','体验_环境体验','场景_生活场景','IP_创作作品','体验_使用体验','工艺','行为_操作行为','图案','营销/活动','风格_其他','功能/功效_生物功效','时间_年/月维度时间','特定人物_明星名人','赛事/展演_学术赛事','材质','攻略','行为_其他','价格/金额_其他','场景_虚拟场景','场景_工作场景','行为_活动','行为_交互行为','赛事/展演_游戏赛事','产品规格','poi_景点','品类_品种','价格/金额_价格描述','人群/宠物_身份','人群/宠物_年龄','颜色','人群/宠物_健康痛点','poi_其他','时间_时间周期','行为_交易类型','时间_季节','场景_装修场景','成分','体验_服务体验','功能/功效_产品功能','aoi_其他']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            labels = line['labels']
            subject = get_entities(labels,id2label=None,markup='bios')
            examples.append(InputExample(guid=guid, text_a=text_a, subject=subject))
        return examples

ner_processors = {
    "cner": CnerProcessor,
    'cluener':CluenerProcessor,
    'omniner': OmninerProcessor
}


