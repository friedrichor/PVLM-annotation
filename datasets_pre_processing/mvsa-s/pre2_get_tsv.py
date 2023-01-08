'''
Filter https:// http:// @username #hashtag
清洗数据，并调整数据格式，将数据信息写入 all.tsv
'''
import os
import re

MVSA_Single_dir = 'data/MVSA_Single'
data_dir = os.path.join(MVSA_Single_dir, 'data')
label_file = os.path.join(MVSA_Single_dir, 'MVSA-S_id_label.txt')
fout_name = os.path.join(MVSA_Single_dir, 'all.tsv')
stopwords = [',', '.', ';', '(', ')', 'RT']  # 停用词

fin = open(label_file, 'r', encoding='utf-8')
fout = open(fout_name, 'w' , encoding='utf-8')
fout.write('index	#1 Label	#2 ImageID	#3 String	#3 String\n')  # 多了个 #3 String，不知道为啥

filtered_by_len = []  # 数据清洗后句子长度不足的句子id列表
conts = []
lines = fin.readlines()
lines.pop(0)
for line in lines:
    idx, label = line.split()
    txt = os.path.join(data_dir, idx + '.txt')
    # select utf-8 encoded text only, some texts are encoded in cp866
    with open(txt, 'r', encoding='utf-8', errors='ignore') as f:
        l = f.readlines()
        assert len(l) == 1
        cont = l[0]
        
        # re.sub()用法的详细介绍 https://blog.csdn.net/jackandsnow/article/details/103885422
        cont = re.sub(r'https://\S+', '', cont)  # 删除 https:// 开头的单词
        cont = re.sub(r'http://\S+', '', cont)  # 删除 http:// 开头的单词
        cont = re.sub(r'\S*@\S+', '', cont)  # @username  删除中间或开头是@的单词  如 a b@c d -> a d, a @c d -> a d (@结尾的单词不删除) 
        cont = re.sub(r'#\S+', '', cont)     # #hashtag   删除单词中#以后的内容 如 a#b -> a 
        cont = cont.split()  # 句子 -> 单词列表
        cont = [w for w in cont if w not in stopwords]  # 删除停用词

        if len(cont) < 3:  # 清洗后句子长度小于3的不作为训练集
            filtered_by_len.append(idx)
            continue
        cont = ' '.join(cont)  # 单词列表 -> 句子
        conts.append(f'{idx}\t{label}\t{idx}.jpg\t{cont}\t\n')

conts.sort(key=lambda x: x.split('\t')[1])  # 以 label 排序
for line in conts:
    fout.write(line)

fin.close()
fout.close()

# print(filtered_by_len)
print(len(filtered_by_len))
