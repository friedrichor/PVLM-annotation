from transformers import BertTokenizer
from utils import processors
from dataset import MSADataset
import os
import csv

processor = processors['mvsa-s']
print(f'processor = {processor}')
label_list, label_map, template_dict = processor(1)
print(f'label_list = {label_list}')
print(f'label_map = {label_map}')
print(f'template_dict = {template_dict}')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
vocab = tokenizer.get_vocab()  # {word: index}
label_id_list = [vocab[token] for token in label_list]
print(f'label_id_list = {label_id_list}')
print(vocab['bad'], vocab['no'], vocab['good'])

label_id_map = {key: vocab[label_map[key]] for key in label_map.keys()}
print(f'label_id_map = {label_id_map}')


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        lines.pop(0)  # remove the header row
        return lines


lines = _read_tsv('datasets/mvsa-s/few-shot1.tsv')  # 读取 few-shot 数据集
line = lines[0]  # 第 idx+1 行，因为 lines.pop(0)
# 以下注释为 processor = processors['mvsa-s']; template = 1 的情况
label_id = label_id_map[line[1]]  # label_id_map = {'negative': 2919, 'neutral': 2053, 'positive': 2204}
                                               # line[1] 为 label 项
img_id = line[2]
text_x = line[3].lower()
print(f'text_x = {text_x}')
text_a = line[4].lower()
print(f'text_a = {text_a}')

tokens_x = tokenizer.tokenize(text_x)
print(f'tokens_x = {tokens_x}')
tokens_a = tokenizer.tokenize(text_a)
print(f'tokens_a = {tokens_a}')


p_idx = 0
input_tokens = []
for i in template_dict['map']:
    # template_dict = {'content': [' [CLS] the sentence " ', ' " has [MASK] emotion [SEP] '], 'map': [0, 'x', 1]}
    if i == 'a':
        input_tokens.extend(tokens_a)
    elif i == 'x':
        input_tokens.extend(tokens_x)
    elif i == 'p':
        input_tokens.extend([prompt_token] * prompt_shape[p_idx])
        p_idx += 1
    else:
        input_tokens.extend(tokenizer.tokenize(template_dict['content'][i]))
    print(f'i = {i}, input_tokens = {input_tokens}')
input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
print(f'input_ids = {input_ids}')
attention_mask = [1] * len(input_ids)
print(f'attention_mask = {attention_mask}')
labels = [-100] * len(input_ids)
print(f'labels = {labels}')
print(f'tokenizer.mask_token_id = {tokenizer.mask_token_id}')
for i, id in enumerate(input_ids):
    if id == tokenizer.mask_token_id:
        labels[i] = label_id
print(f'labels = {labels}')



for k, v in vocab.items():
    if v == tokenizer.mask_token_id:
        print(f'tokenizer.mask_token_id  {k}.key = {v}')
    elif v == tokenizer.sep_token_id:
        print(f'tokenizer.sep_token_id  {k}.key = {v}')
    elif k == '[unused1]':
        print(f'[unused1]  {k}.key = {v}')
    elif k == '[unused2]':
        print(f'[unused2]  {k}.key = {v}')
       


print(f'len(vocab) = {len(vocab)}')
img_token = '[unused2]'
img_token_id = vocab[img_token]
print(f'img_token_id = {img_token_id}')
img_token_len = 1
addon = [img_token_id] * img_token_len + [tokenizer.sep_token_id]
print(f'addon = {addon}')
input_ids = [input_ids[0]] + addon + input_ids[1:]
print(f'input_ids = {input_ids}')
attention_mask = [1] * (img_token_len + 1) + attention_mask
print(f'attention_mask = {attention_mask}')
labels = [-100] * (img_token_len + 1) + labels
print(f'labels = {labels}')