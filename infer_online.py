#!/usr/bin/python

# encoding: utf-8

"""
@desc: 模型推理部分，分为本地载入模型推理或者tensorflow serving grpc 推理
"""
import os
import grpc
import codecs
import pickle
import warnings

import tensorflow as tf
from bert import tokenization
from public_tools.ner_utils import get_entity, get_result
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import types_pb2


class InputFeatures(object):
    """A single set of features of msra_data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class Args():
    def __init__(self):
        self.kflod = 2
        self.max_seq_length = 128
        self.is_training =False
        self.use_one_hot_embeddings = False
        self.batch_size = 1
        self.dev_file = './data/clue_ner/dev.txt'
        self.dev_file_json = './data/clue_ner/dev.json'
        self.dev_result_file = './data/clue_ner/submit/cluener_predict_dev.json'
        self.test_file = './data/clue_ner/test.json'
        self.test_result_file = './data/clue_ner/submit/cluener_predict.json'

        self.bert_config_file = 'D:/Expriment/pretrain_model_tf/bert/bert_config.json'
        self.output_dir = 'D:/Expriment/model_output/ner_tool/bert_wol/single_task/clue_ner/runs/checkpoints'
        self.vocab_file = 'D:/Expriment/pretrain_model_tf/bert/vocab.txt'
        self.label2id_dic_dir = 'D:/Expriment/model_output/ner_tool/bert_wol/single_task/clue_ner/runs/checkpoints'
        self.export_dir = "D:/Expriment/model_output/ner_tool/bert_wol/single_task/clue_ner/exported_model/1/1622539079"


class InferenceBase(object):
    def __init__(self, vocab_file, label2id_dic_dir, url=None, model_name=None,
                 signature_name=None, export_dir=None, do_lower_case=True):
        """
        预测的基类，分为两种方式预测
            a. grpc 请求方式
            b. 本地导入模型方式

        :arg
        vocab_file: bert 预训练词典的地址，这里在 'chinese_L-12_H-768_A-12/vocab.txt '中
        labels: str 或 list 类型，需要被转化为id的label，当为str类型的时候，即为标签-id的pkl文件名称；
                当为list时候，即为标签列表。
        url: string类型，用于调用模型测试接口，host:port，例如'10.0.10.69:8500'
        export_dir: string类型，模型本地文件夹目录，r'model\1554373222'
        model_name: string类型，tensorflow serving 启动的时候赋予模型的名称，当
                    url被设置的时候一定要设置。
        signature_name: string类型，tensorflow serving 的签名名称，当
                    url被设置的时候一定要设置。
        do_lower_case: 是否进行小写处理

        :raise
        url和export_dir至少选择一个，当选择url的时候，model_name和signature_name不能为
        None。
        """
        self.url = url
        self.export_dir = export_dir

        if export_dir:
            self.predict_fn = tf.contrib.predictor.from_saved_model(self.export_dir)

        if self.url:
            channel = grpc.insecure_channel(self.url)
            self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
            self.request = predict_pb2.PredictRequest()
            self.model_name = model_name
            self.signature_name = signature_name

            self.request.model_spec.name = self.model_name

            self.request.model_spec.signature_name = self.signature_name

            if self.model_name is None or self.signature_name is None:
                raise ValueError('`model_name` and `signature_name` should  not NoneType')

        if url is None and export_dir is None:
            raise ValueError('`url` or `export_dir`is at least of one !')

        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        print('Tokenizer加载完毕！')
        # 获取id2char字典
        self.id2char = self.tokenizer.inv_vocab
        self.id2label = self._load_id_map_label(label2id_dic_dir)
        print('标签信息加载完毕！')

    def local_infer(self, examples):
        """
        导入本地的PB文件进行预测
        """
        pass

    def tf_serving_infer(self, examples):
        """
        使用tensorflow serving进行grpc请求预测
        """
        pass

    def preprocess(self, sentences, max_seq_length):
        pass

    def create_example(self):
        pass

    @staticmethod
    def _load_id_map_label(label2id_dic_dir):
        # 加载label->id的词典
        with codecs.open(os.path.join(label2id_dic_dir, 'label2id.pkl'), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        return id2label


class NerInfer(InferenceBase):
    def __init__(self, vocab_file, label2id_dic_dir, url=None, model_name=None,
                 signature_name=None, export_dir=None, do_lower_case=True):
        """
        bert ner, 参数解释查看 `InferenceBase`
        """
        super(NerInfer, self).__init__(vocab_file, label2id_dic_dir, url, model_name, signature_name, export_dir, do_lower_case)

    def preprocess(self, sentences, max_seq_length):
        """
        对sentences进行预处理，并生成examples

        :arg
        sentences: 二维列表，即输入的句子，输入有一下要求：
                （1）可以是一段话，但是每个句子最好小于64个字符串长度
                （2）长度不可以小于2
        max_seq_length: 输入的每一个句子的最大长度

        :return
        examples: tf.train.Example对象
        new_tokens: 二维列表，sentences清洗后的tokens
        sentences_index: 二维列表，分句后，对应到原始句子的下标
                        例如：[[0], [1, 2]...]
        """
        if not sentences or not isinstance(sentences, list):
            raise ValueError('`sentences` must be list object and not a empty list !')

        examples = []
        sentence = sentences[0]
        feature = self.convert_single_example(sentence, max_seq_length)
        features = dict()
        features['input_ids'] = tf.train.Feature(int64_list=tf.train.Int64List(value=feature.input_ids))
        features['input_mask'] = tf.train.Feature(int64_list=tf.train.Int64List(value=feature.input_mask))
        features['segment_ids'] = tf.train.Feature(int64_list=tf.train.Int64List(value=feature.segment_ids))
        features['label_ids'] = tf.train.Feature(int64_list=tf.train.Int64List(value=feature.label_ids))
        example = tf.train.Example(features=tf.train.Features(feature=features))
        examples.append(example.SerializeToString())
        return examples, feature

    def convert_single_example(self, example, max_seq_length):
        """
        将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
        :param example: 一个样本
        :param max_seq_length:
        :return: InputFeatures对象
        """
        tokens = self.tokenizer.tokenize(example)
        # tokens = tokenizer.tokenize(example.text)
        # 序列截断
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
        ntokens = []
        segment_ids = []
        ntokens.append("[CLS]")  # 句子开始设置CLS 标志
        segment_ids.append(0)
        # append("O") or append("[CLS]") not sure!
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
        ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
        segment_ids.append(0)
        # append("O") or append("[SEP]") not sure!
        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
        input_mask = [1] * len(input_ids)

        # padding, 使用
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            ntokens.append("**NULL**")
            # label_mask.append(0)
        # print(len(input_ids))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # assert len(label_mask) == max_seq_length

        # 结构化为一个类
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_ids=[8],
            # label_mask = label_mask
        )
        return feature

    def infer(self, sentences, max_seq_length):
        """
        对外的测试接口

        :arg
        sentences:  二维列表，即输入的句子，输入有一下要求：
                （1）可以是一段话，但是每个句子最好小于64个字符串长度
                （2）长度不可以小于2
        max_seq_length: 输入的每一个句子的最大长度

        sentences_entities: 返回每一个句子的实体
        """
        examples, feature = self.preprocess(sentences, max_seq_length)
        if self.url:
            predictions = self.tf_serving_infer(examples)
        else:
            predictions = self.local_infer(examples)
        y_pred = predictions['output']
        sentences_entities = self.get_entity_result(feature, self.id2char, self.id2label, y_pred)
        return sentences_entities

    def tf_serving_infer(self, examples):
        """
        使用tensorflow serving预测

        :arg
        examples: tf.train.Example 对象

        :return
        二维列表，预测结果
        """
        self.request.inputs['examples'].CopyFrom(tf.make_tensor_proto(examples, dtype=types_pb2.DT_STRING))
        response = self.stub.Predict(self.request, 5.0)

        predictions = {}
        for key in response.outputs:
            tensor_proto = response.outputs[key]
            nd_array = tf.contrib.util.make_ndarray(tensor_proto)
            predictions[key] = nd_array

        return predictions

    def local_infer(self, examples):
        """
        本地进行预测，参数解释同上
        """
        predictions = self.predict_fn({'examples': examples})

        return predictions

    @staticmethod
    def get_entity_result(feature, id2char, id2label, y_pred):
        """
        提取实体

        :arg
        tokens: 二维列表，句子处理后得到的token
        tags: 二维列表，预测的结果
        sentences_index: 二维列表，句子拆分后，对应到原句的index

        :return
        sentences_entities: 二维列表，返回实体结果，例如[('昆凌', 'PER')...]
        """
        sent_tag = []
        y_pred_clean = []
        input_ids_clean = []
        # 去除 [CLS] 和 [SEP]获取正确的tag范围
        print([id2char[i] for i in feature.input_ids])
        print(len(feature.input_ids))
        print(y_pred[0][0])
        print([id2label[i] for i in list(y_pred[0])])
        print(len(list(y_pred[0])))
        for index, id in enumerate(feature.input_ids):
            char = id2char[id]
            tag = id2label[list(y_pred[0])[index]]
            if char == "[CLS]":
                continue
            if char == "[SEP]":
                break
            input_ids_clean.append(id)
            sent_tag.append(tag)
            y_pred_clean.append(list(y_pred[0])[index])

        sent_tag = ' '.join(sent_tag)
        print(sentence + '\n' + sent_tag)
        entity = get_entity([sentence], [y_pred_clean], id2label)
        print('predict_result:')
        print(entity)
        return entity


if __name__ == '__main__':
    args = Args()
    nerinfer = NerInfer(vocab_file=args.vocab_file, label2id_dic_dir=args.label2id_dic_dir, url='192.168.9.29:8500', model_name='ner_model', signature_name='serving_default')

    # while True:
    #     sentence = input('请输入句子：')
    #     print(nerinfer.infer([sentence], max_seq_length))
    sentence = '或者可以直接登陆美国使馆的网站来查询'
    print(nerinfer.infer([sentence], args.max_seq_length))
