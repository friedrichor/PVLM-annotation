'''
清洗数据，删除图片和文字之间情感不一致的样本
'''

# Please excute following commands first
# $ unzip MVSA-Single.zip
# $ mv data MVSA-S_data
import os

MVSA_Single_dir = 'data/MVSA_Single'
fin = open(os.path.join(MVSA_Single_dir, 'labelResultAll.txt'), 'r', encoding='utf-8')
fout = open(os.path.join(MVSA_Single_dir, 'MVSA-S_id_label.txt'), 'w' , encoding='utf-8')

fout.write('id	label\n')
lines = fin.readlines()
lines.pop(0)
count = 0  # 数据集中图文情感相反的数量
for l in lines:
    id, senti = l.split()
    s1, s2 = senti.split(',')
    # 统一图文的情感为一个情感
    if (s1 == 'positive' and s2 == 'negative') or (s2 == 'positive' and s1 == 'negative'):  # 图文情感相反，脏数据，清洗掉
        count += 1
    elif s1 == s2:  # 图文情感一致
        fout.write(f'{id}\t{s1}\n')
    elif s1 == 'neutral':  # 若文本为 neutral，则将该文本情感定为与图片一致的
        fout.write(f'{id}\t{s2}\n')
    elif s2 == 'neutral':  # 若图像为 neutral，则将该文本情感定为与文本一致的
        fout.write(f'{id}\t{s1}\n')
    else:
        raise RuntimeError('Error')
print(count)

fin.close()
fout.close()
