# NER实验及模型部署

本NER项目包含多个中文数据集，模型采用BiLSTM+CRF、BERT+Softmax、BERT+Cascade、BERT+WOL等，其中BiLSTM+CRF部分采用的字符向量为BERT字向量，最后在CLUE_NER数据集上进行实验对比，并采用TFServing进行模型部署进行线上推理和线下推理。  

## 运行BERT+CRF模型为例
- step1：模型训练  
python run_bert_crf.py 
- step2：模型验证测试  
python predict_bert_crf.py
- step3：模型推理预测  
python infer_online.py 

## 模型推理模模式  
一、基于checkpoints进行online，offline推理预测  
1.dev_offline  预测标签与真实标签评价结果+预测实体与真实实体评价结果+写入评测文件  
2.predict_online  识别即时输入句子的实体  
3.predict_offline  预测结果写入文件，提交比赛结果，本项目对应于clue的ner预测标准文件格式  
  
二、基于tfserving + pb文件进行online，offline推理预测    

模型推理部分，分为本地载入模型推理或者tensorflow serving grpc 推理，具体详见 infer_online.py 和 infer_offline.py  

— 实验结果    


