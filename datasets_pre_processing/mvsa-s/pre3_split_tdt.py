"""
划分训练集、验证集、测试集
"""

import os
import random
random.seed(34)

MVSA_Single_dir = 'data/MVSA_Single'
fin = open(os.path.join(MVSA_Single_dir, 'all.tsv'), 'r', encoding='utf-8')
ftrain = open(os.path.join(MVSA_Single_dir, 'train.tsv'), 'w', encoding='utf-8')
fdev = open(os.path.join(MVSA_Single_dir, 'dev.tsv'), 'w', encoding='utf-8')
ftest = open(os.path.join(MVSA_Single_dir, 'test.tsv'), 'w', encoding='utf-8')

lines = fin.readlines()
title = lines.pop(0)
ftrain.write(title)
fdev.write(title)
ftest.write(title)

l = list(range(len(lines)))
random.shuffle(l)  # 打乱

# 训练集:验证集:测试集 = 8:1:1
train_range = len(l) // 10 * 8
dev_range = len(l) // 10 * 1

for idx in sorted(l[:train_range]):  # 训练集
    ftrain.write(lines[idx])

for idx in sorted(l[train_range: train_range + dev_range]):  # 验证集
    fdev.write(lines[idx])

for idx in sorted(l[train_range + dev_range: ]):  # 测试集
    ftest.write(lines[idx])

fin.close()
ftrain.close()
fdev.close()
ftest.close()
