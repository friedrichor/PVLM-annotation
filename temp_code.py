from transformers import BertTokenizer, BertForMaskedLM
from utils import processors
from dataset import MSADataset
import os
import csv
import timm
import json

'''
model_name = 'nf_resnet50'
backbone = timm.create_model(model_name, pretrained=True)
if model_name == "resnet50":
    global_pool = backbone.global_pool
else:
    global_pool = backbone.head.global_pool
# print(backbone)
# print(backbone.head)
# print(global_pool)
print(backbone.forward_features)
'''

'''
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
vocab = tokenizer.get_vocab()
input_tokens = ['[CLS]', 'a', 'b', '[SEP]']
input_tokens_id = tokenizer.convert_tokens_to_ids(input_tokens)  
print(f'input_tokens_id = {input_tokens_id}')

lm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
output = lm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
loss, logits = output.loss, output.logits
print(f'lm_model.loss = {lm_model.loss}')
'''
import torch
from transformers import BertModel
from transformers import BertTokenizer
from transformers import BertForMaskedLM 

# 获取tokenizer
bert_tokenzier = BertTokenizer.from_pretrained('bert-base-uncased')
# 加载bert mask 预训练模型
maskbert = BertForMaskedLM.from_pretrained('bert-base-uncased')
input_text = "POST MATCH: Dean Smith furious after penalty debacle"
output_text = "POST MATCH: Dean Smith furious after penalty debacle"
#
output_labels = bert_tokenzier(output_text, add_special_tokens=True, padding=True, return_tensors='pt')['input_ids']
print(f'output_labels = {output_labels}')
input_tokens = bert_tokenzier(input_text, add_special_tokens=True, padding=True, return_tensors='pt')
print(f'input_tokens = {input_tokens}')
#
maskbert_outputs = maskbert(**input_tokens, labels=output_labels, return_dict=True,
                            output_hidden_states=True)
maskbert_logits = maskbert_outputs.logits
print(f'maskbert_logits = {maskbert_logits}')
print("maskbert logits shape: ", maskbert_logits.size())
maskbert_logits_pre = maskbert_logits[output_labels != 101]
print(f'maskbert_logits_pre = {maskbert_logits_pre}')
print("maskbert_logits_pre shape: ", maskbert_logits_pre.size())
maskbert_loss = maskbert_outputs.loss
maskbert_hidden = maskbert_outputs.hidden_states[-1]