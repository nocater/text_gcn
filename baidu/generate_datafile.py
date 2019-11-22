import pandas as pd
import jieba
import numpy as np

"""
Generate dataset.txt datset.clean.txt
"""

dataset = 'baidu_95'

df = pd.read_csv(f'./{dataset}.csv', header=None, names=["labels", "item"], dtype=str)

# shuffle
# the baidu_95 is shuffled


# cut words
df['item'] = df.item.apply(lambda x: list(jieba.cut(x)))
# remove stopwords
stopwords = open('./stopwords.txt').readlines()
stopwords = [i.strip() for i in stopwords]
df['item'] = df.item.apply(lambda x: [word for word in x if word not in stopwords])

# write to corpus/baidu_95.clean.txt corpus/baidu_95.txt ./baidu_95.txt
with open(f'../data/corpus/{dataset}.txt', 'w') as f:
    for line in df.item:
        f.write(' '.join(line))

with open(f'../data/corpus/{dataset}.clean.txt', 'w') as f:
    for line in df.item:
        f.write(' '.join(line))

# split dataset
# sublabel 先使用单标签分类验证模型效果
sublabel = 1
index_split = len(df) * 0.9
with open(f'../data/{dataset}.txt', 'w') as f:
    for index, row in df.iterrows():
        category = 'train' if index <= index_split else 'test'
        f.write(f'{index}\t{category}\t{row[0].split()[sublabel]}\n')

print(f'Dataset{dataset} file is generated, please use build_graph')