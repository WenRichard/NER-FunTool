## NER实验及模型部署

本NER项目包含多个中文数据集，模型采用BiLSTM+CRF、BERT+Softmax、BERT+Cascade、BERT+WOL等，其中BiLSTM+CRF部分采用的字符向量为BERT字向量，最后在CLUE_NER数据集上进行实验对比，并采用TFServing进行模型部署进行线上推理和线下推理。  
详细请见知乎：  
[NER实战：从实验到BERT模型部署服务](https://zhuanlan.zhihu.com/p/378307648)

#### 如何运行？以运行BERT+CRF模型为例
- step1：模型训练  
python run_bert_crf.py 
- step2：模型验证测试  
python predict_bert_crf.py
- step3：模型推理预测  
python infer_online.py 
  
**模型评价方式**
对于NER模型，评判模型是否好坏的方法有2种
- **标签评价法**：对模型预测的标签进行评价，计算预测标签和原始标签的p，r，f1值  
tag_evaluating.py 其中可以设置是否不把O标签考虑在内  
- **实体评价法**：对预测的实体进行评价，计算预测实体的f1值等等    
entity_evaluating.py 

#### 模型推理模模式  
一、基于checkpoints进行online，offline推理预测  
1.dev_offline  预测标签与真实标签评价结果+预测实体与真实实体评价结果+写入评测文件  
2.predict_online  识别即时输入句子的实体  
3.predict_offline  预测结果写入文件，提交比赛结果，本项目对应于clue的ner预测标准文件格式  
  
二、基于tfserving + pb文件进行online，offline推理预测    

模型推理部分，分为本地载入模型推理或者tensorflow serving grpc 推理，具体详见 infer_online.py 和 infer_offline.py  
  

#### 实验结果    
|          | model             | precision | recall | f1    | entity_f1 | epoch | predict_time |
| -------- | ----------------- | --------- | ------ | ----- | --------- | ----- | ------------ |
| clue_ner | lstm_crf          | 0.926     | 0.865  | 0.890 | 0.671     | 20    |              |
|          | bert_crf          | 0.871     | 0.820  | 0.840 | 0.741     | 3     |              |
|          | bert_wol          | 0.900     | 0.851  | 0.874 | 0.800     | 3     |              |
|          | casecade_lstm_crf | 0.942     | 0.891  | 0.916 | 0.783     | 20    |              |
|          | casecade_bert_crf | 0.892     | 0.842  | 0.866 | 0.800     | 3     |              |


**TODO**  
1.加上ALBERT，ROBERTA等预训练语言模型  
2.在模型中融合入词向量等特征信息  

**Reference**  
1.[NER](https://github.com/wavewangyue/ner)  
2.[CLUE_NER](https://arxiv.org/abs/2001.04351)
3.[NerAdapter](https://github.com/Vincent131499/NerAdapter)

**如果觉得我的工作对您有帮助，请不要吝啬右上角的小星星哦！欢迎Fork和Star！也欢迎提交PR！**    
**留言请在Issues或者email  richardxie1205@gmail.com**
    

