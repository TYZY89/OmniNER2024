""" Named entity recognition fine-tuning: utilities to work with CLUENER task. """
import torch
import logging
import os
import copy
import json
from .utils_ner import DataProcessor
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, input_len,segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len

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
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:,:max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels,all_lens

# def convert_examples_to_features(examples,label_list,max_seq_length,tokenizer,
#                                  cls_token_at_end=False,cls_token="[CLS]",cls_token_segment_id=1,
#                                  sep_token="[SEP]",pad_on_left=False,pad_token=0,pad_token_segment_id=0,
#                                  sequence_a_segment_id=0,mask_padding_with_zero=True,):
#     """ Loads a data file into a list of `InputBatch`s
#         `cls_token_at_end` define the location of the CLS token:
#             - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
#             - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
#         `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
#     """
#     label_map = {label: i for i, label in enumerate(label_list)}
#     features = []
#     for (ex_index, example) in enumerate(examples):
#         if ex_index % 10000 == 0:
#             logger.info("Writing example %d of %d", ex_index, len(examples))
#         if isinstance(example.text_a,list):
#             example.text_a = " ".join(example.text_a)
#         tokens = tokenizer.tokenize(example.text_a)
#         label_ids = [label_map[x] for x in example.labels]
#         # Account for [CLS] and [SEP] with "- 2".
#         special_tokens_count = 2
#         if len(tokens) > max_seq_length - special_tokens_count:
#             tokens = tokens[: (max_seq_length - special_tokens_count)]
#             label_ids = label_ids[: (max_seq_length - special_tokens_count)]

#         # The convention in BERT is:
#         # (a) For sequence pairs:
#         #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
#         #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
#         # (b) For single sequences:
#         #  tokens:   [CLS] the dog is hairy . [SEP]
#         #  type_ids:   0   0   0   0  0     0   0
#         #
#         # Where "type_ids" are used to indicate whether this is the first
#         # sequence or the second sequence. The embedding vectors for `type=0` and
#         # `type=1` were learned during pre-training and are added to the wordpiece
#         # embedding vector (and position vector). This is not *strictly* necessary
#         # since the [SEP] token unambiguously separates the sequences, but it makes
#         # it easier for the model to learn the concept of sequences.
#         #
#         # For classification tasks, the first vector (corresponding to [CLS]) is
#         # used as as the "sentence vector". Note that this only makes sense because
#         # the entire model is fine-tuned.
#         tokens += [sep_token]
#         label_ids += [label_map['O']]
#         segment_ids = [sequence_a_segment_id] * len(tokens)

#         if cls_token_at_end:
#             tokens += [cls_token]
#             label_ids += [label_map['O']]
#             segment_ids += [cls_token_segment_id]
#         else:
#             tokens = [cls_token] + tokens
#             label_ids = [label_map['O']] + label_ids
#             segment_ids = [cls_token_segment_id] + segment_ids

#         input_ids = tokenizer.convert_tokens_to_ids(tokens)
#         # The mask has 1 for real tokens and 0 for padding tokens. Only real
#         # tokens are attended to.
#         input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
#         input_len = len(label_ids)
#         # Zero-pad up to the sequence length.
#         padding_length = max_seq_length - len(input_ids)
#         if pad_on_left:
#             input_ids = ([pad_token] * padding_length) + input_ids
#             input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
#             segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
#             label_ids = ([pad_token] * padding_length) + label_ids
#         else:
#             input_ids += [pad_token] * padding_length
#             input_mask += [0 if mask_padding_with_zero else 1] * padding_length
#             segment_ids += [pad_token_segment_id] * padding_length
#             label_ids += [pad_token] * padding_length

#         assert len(input_ids) == max_seq_length
#         assert len(input_mask) == max_seq_length
#         assert len(segment_ids) == max_seq_length
#         assert len(label_ids) == max_seq_length
#         if ex_index < 5:
#             logger.info("*** Example ***")
#             logger.info("guid: %s", example.guid)
#             logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
#             logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
#             logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
#             logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
#             logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

#         features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask,input_len = input_len,
#                                       segment_ids=segment_ids, label_ids=label_ids))
#     return features
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
        return ["X",'B-CONT','B-EDU','B-LOC','B-NAME','B-ORG','B-PRO','B-RACE','B-TITLE',
                'I-CONT','I-EDU','I-LOC','I-NAME','I-ORG','I-PRO','I-RACE','I-TITLE',
                'O','S-NAME','S-ORG','S-RACE',"[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-','I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
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

    # def get_labels(self):
    #     """See base class."""
    #     return ["X", "B-address", "B-book", "B-company", 'B-game', 'B-government', 'B-movie', 'B-name',
    #             'B-organization', 'B-position','B-scene',"I-address",
    #             "I-book", "I-company", 'I-game', 'I-government', 'I-movie', 'I-name',
    #             'I-organization', 'I-position','I-scene',
    #             "S-address", "S-book", "S-company", 'S-game', 'S-government', 'S-movie',
    #             'S-name', 'S-organization', 'S-position',
    #             'S-scene','O',"[START]", "[END]"]
    def get_labels(self):
        """See base class."""
        return ["X", "B-时间_日维度时间", "B-时间_节假日",
                "I-时间_日维度时间", "I-时间_节假日",
                "S-时间_日维度时间", "S-时间_节假日",
                'O',"[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
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
        # return ["X", "B-时间_日维度时间", "B-时间_节假日",
        # "I-时间_日维度时间", "I-时间_节假日",
        # "S-时间_日维度时间", "S-时间_节假日",
        # 'O',"[START]", "[END]"]
        # """See base class."""
        return ["X",'B-成分','B-品类_产品','B-品牌','B-数量','B-行为_操作行为','B-人群/宠物_身体特征','B-行为_交互行为','B-人群/宠物_身份','B-价格/金额_价格描述','B-时间_节假日','B-产品特性','B-产品系列','B-颜色','B-营销/活动','B-时间_年/月维度时间','B-材质','B-款式/形态_款式外形','B-功能/功效_人体功效','B-风格_产品艺术风格','B-风格_妆造风格','B-人群/宠物_其他','B-体验_使用体验','B-行为_活动','B-时间_时间周期','B-攻略','B-aoi_其他','B-时间_季节','B-气味/味道_气味','B-负向效果','B-场景_生活场景','B-产品规格','B-人群/宠物_性别','B-人群/宠物_肤质','B-价格/金额_价格','B-人群/宠物_职业','B-特定人物_其他','B-aoi_产地','B-人群/宠物_健康痛点','B-场所','B-品类_服务','B-款式/形态_其他','B-特定人物_明星名人','B-人群/宠物_教育阶段','B-时间_日维度时间','B-品类_其他','B-品类_品种','B-款式/形态_包装','B-功能/功效_产品功能','B-人群/宠物_年龄','B-图案','B-poi_其他','B-方法/技能_方法','B-行为_心理/情绪','B-体验_服务体验','B-风格_其他','B-特定人物_主播网红','B-气味/味道_味道','B-工艺','B-IP_创作作品','B-场景_工作场景','B-poi_景点','B-场景_娱乐场景','B-场景_学习场景','B-品类_专业/学科','B-方法/技能_技能','B-赛事/展演_文娱展演','B-风格_建筑风格','B-价格/金额_其他','B-IP_角色','B-IP_游戏','I-成分','I-品类_产品','I-品牌','I-数量','I-行为_操作行为','I-人群/宠物_身体特征','I-行为_交互行为','I-人群/宠物_身份','I-价格/金额_价格描述','I-时间_节假日','I-产品特性','I-产品系列','I-颜色','I-营销/活动','I-时间_年/月维度时间','I-材质','I-款式/形态_款式外形','I-功能/功效_人体功效','I-风格_产品艺术风格','I-风格_妆造风格','I-人群/宠物_其他','I-体验_使用体验','I-行为_活动','I-时间_时间周期','I-攻略','I-aoi_其他','I-时间_季节','I-气味/味道_气味','I-负向效果','I-场景_生活场景','I-产品规格','I-人群/宠物_性别','I-人群/宠物_肤质','I-价格/金额_价格','I-人群/宠物_职业','I-特定人物_其他','I-aoi_产地','I-人群/宠物_健康痛点','I-场所','I-品类_服务','I-款式/形态_其他','I-特定人物_明星名人','I-人群/宠物_教育阶段','I-时间_日维度时间','I-品类_其他','I-品类_品种','I-款式/形态_包装','I-功能/功效_产品功能','I-人群/宠物_年龄','I-图案','I-poi_其他','I-方法/技能_方法','I-行为_心理/情绪','I-体验_服务体验','I-风格_其他','I-特定人物_主播网红','I-气味/味道_味道','I-工艺','I-IP_创作作品','I-场景_工作场景','I-poi_景点','I-场景_娱乐场景','I-场景_学习场景','I-品类_专业/学科','I-方法/技能_技能','I-赛事/展演_文娱展演','I-风格_建筑风格','I-价格/金额_其他','I-IP_角色','I-IP_游戏','S-成分','S-品类_产品','S-品牌','S-数量','S-行为_操作行为','S-人群/宠物_身体特征','S-行为_交互行为','S-人群/宠物_身份','S-价格/金额_价格描述','S-时间_节假日','S-产品特性','S-产品系列','S-颜色','S-营销/活动','S-时间_年/月维度时间','S-材质','S-款式/形态_款式外形','S-功能/功效_人体功效','S-风格_产品艺术风格','S-风格_妆造风格','S-人群/宠物_其他','S-体验_使用体验','S-行为_活动','S-时间_时间周期','S-攻略','S-aoi_其他','S-时间_季节','S-气味/味道_气味','S-负向效果','S-场景_生活场景','S-产品规格','S-人群/宠物_性别','S-人群/宠物_肤质','S-价格/金额_价格','S-人群/宠物_职业','S-特定人物_其他','S-aoi_产地','S-人群/宠物_健康痛点','S-场所','S-品类_服务','S-款式/形态_其他','S-特定人物_明星名人','S-人群/宠物_教育阶段','S-时间_日维度时间','S-品类_其他','S-品类_品种','S-款式/形态_包装','S-功能/功效_产品功能','S-人群/宠物_年龄','S-图案','S-poi_其他','S-方法/技能_方法','S-行为_心理/情绪','S-体验_服务体验','S-风格_其他','S-特定人物_主播网红','S-气味/味道_味道','S-工艺','S-IP_创作作品','S-场景_工作场景','S-poi_景点','S-场景_娱乐场景','S-场景_学习场景','S-品类_专业/学科','S-方法/技能_技能','S-赛事/展演_文娱展演','S-风格_建筑风格','S-价格/金额_其他','S-IP_角色','S-IP_游戏','O',"[START]","[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

ner_processors = {
    "cner": CnerProcessor,
    'cluener': CluenerProcessor,
    'omniner': OmninerProcessor
}
