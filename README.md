# text_gcn

添加: **多标签文本分类**

## 模型使用说明
将自己的数据处理成两个文件：数据文件，标签文件。  
数据文件存放于`data/corpus/{dataset}.txt`，标签文件存放于`data/{dataset}.txt`, 格式是`id`, `train/text标识`，`(多)类别`  
- `python build_graph {dataset}` 创建图特征数据
- `python train.py {dataset} True` 进行多标签任务训练

原始repo: 

[GCN_AAAI2019](https://github.com/yao8839836/text_gcn/)

原始论文:

Liang Yao, Chengsheng Mao, Yuan Luo. "Graph Convolutional Networks for Text Classification." In 33rd AAAI Conference on Artificial Intelligence (AAAI-19), 7370-7377


## Require

Python 3.6

Tensorflow >= 1.4.0
