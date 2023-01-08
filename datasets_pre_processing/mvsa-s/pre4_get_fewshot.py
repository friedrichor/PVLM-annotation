"""
随机抽取用于 few-shot 的数据集
"""

import os
import random
random.seed(34)

MVSA_Single_dir = 'data/MVSA_Single'
fin_name = os.path.join(MVSA_Single_dir, 'train.tsv')
fout1_name = os.path.join(MVSA_Single_dir, 'few-shot1.tsv')
fout2_name = os.path.join(MVSA_Single_dir, 'few-shot2.tsv')

cata_num = 3
cata_start_idx = [1, 982, 1309, 3225]  # the last row of each category 
                                       # train.tsv中每一类的最后一行的行数
                                       # negative 2-982; neutral 983-1309; positive 1310-3225
k = 12

fin = open(fin_name, 'r', encoding='utf-8')
fout1 = open(fout1_name, 'w', encoding='utf-8')
fout2 = open(fout2_name, 'w', encoding='utf-8')

lines = fin.readlines()
fout1.write(lines[0])
fout2.write(lines[0])

for i in range(cata_num):
    sample_range = range(cata_start_idx[i], cata_start_idx[i+1])
    sample_idx = random.sample(sample_range, k=2*k)  # 从 sample_range 中随机选出 2*k=24个
    # sample_idx.sort()
    for idx in sorted(sample_idx[:k]):  # sample_idx 中前 k 个写入 few-shot1.tsv
        fout1.write(lines[idx])
    for idx in sorted(sample_idx[k:]):  # sample_idx 中后 k 个写入 few-shot2.tsv
        fout2.write(lines[idx])

fin.close()
fout1.close()
fout2.close()
