import csv
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms


# 以下注释为 processor = processors['mvsa-s']; template = 1; img_token_len=1 的情况
class MSADataset(Dataset):
    def __init__(self, args, processor, mode='train', max_seq_length=128):
        self.dataset = args.dataset
        self.no_img = args.no_img
        
        self.data_dir = args.data_dir
        self.img_dir = args.img_dir

        self.tokenizer = BertTokenizer.from_pretrained(args.model_name, local_files_only=True)
        self.vocab = self.tokenizer.get_vocab()  # {word: index}

        # parser.add_argument("--prompt_token", type=str, default='[unused1]')
        self.prompt_token = args.prompt_token
        self.prompt_token_id = self.vocab[args.prompt_token]  # prompt_token='[unused1]', prompt_token_id=2
        # prompt_shape_pt='33-0', prompt_shape_pvlm='33-3'
        self.prompt_shape = [int(i) for i in args.prompt_shape.split('-')[0]]
        self.prompt_img_len = int(args.prompt_shape[-1])

        # parser.add_argument("--img_token", type=str, default='[unused2]')
        self.img_token_id = self.vocab[args.img_token]  # img_token='[unused2]', img_token_id=3
        # for img_token_len in 1 2 3 4 5
        self.img_token_len = args.img_token_len
        self.max_seq_length = max_seq_length
        
        # 以下注释为 processor = processors['mvsa-s']; template = 1 的情况
        self.template = args.template
        label_list, label_map, self.template_dict = processor(args.template)
        # label_list = ['bad', 'no', 'good']
        # label_map = {'negative': 'bad', 'neutral': 'no', 'positive': 'good'}
        # template_dict = {'content': [' [CLS] the sentence " ', ' " has [MASK] emotion [SEP] '], 'map': [0, 'x', 1]}
        self.label_id_list = [self.vocab[token] for token in label_list]  # [2919, 2053, 2204] (['bad', 'no', 'good']对应的索引)
        self.label_id_map = {key: self.vocab[label_map[key]] for key in label_map.keys()}  # {'negative': 2919, 'neutral': 2053, 'positive': 2204}

        if mode == 'train':
            print("[#] Looking At {}".format(os.path.join(self.data_dir, args.few_shot_file)))
            self.lines = self._read_tsv(os.path.join(self.data_dir, args.few_shot_file))  # 读取 few-shot 数据集
        else:
            print("[#] Looking At {}".format(os.path.join(self.data_dir, f"{mode}.tsv")))
            self.lines = self._read_tsv(os.path.join(self.data_dir, f"{mode}.tsv"))

        if not args.no_img:  # if args.no_img == False
            print("[|] Reading imgs...")
            self.img_dict = self._read_imgs()
        
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            lines.pop(0)  # remove the header row
            return lines

    def _read_imgs(self):
        img_dict = {}  # {img_id: img(tensor格式)}
        for line in self.lines:  # self.lines 为 few-shot 数据集的内容
            img_id = line[2]  # 2 ImageID
            img = Image.open(os.path.join(self.img_dir, img_id)).convert('RGB')
            img = transforms.Resize([224, 224])(img)
            img = transforms.ToTensor()(img)  # (3,224,224)
            img_dict[img_id] = img
        return img_dict

    def _supervised_encode(self, tokens_x, tokens_a, label_id):
        if len(tokens_x) > self.max_seq_length:  # default 128
            tokens_x = tokens_x[:self.max_seq_length]  # 所有句子长度都不超过 max_seq_length，否则截短至 max_seq_length

        # textual template
        # [CLS] for " [a] " , the sentence " [x] " has [bad/no/good] emotion [SEP]
        # [CLS] for " [a] " , the sentence " [x] " presents a [negative/neutral/positive] sentiment [SEP]
        # [CLS] [negative/neutral/positive] [p] [a] [p] [x] [p]
        p_idx = 0
        input_tokens = []
        for i in self.template_dict['map']:
            # template_dict = {'content': [' [CLS] the sentence " ', ' " has [MASK] emotion [SEP] '], 'map': [0, 'x', 1]}
            # [CLS] the sentence " 原句子 " has [MASK] emotion [SEP] 其中[MASK]为情感label，通过对原句子改写的方式实现 prompt
            if i == 'a':
                input_tokens.extend(tokens_a)
            elif i == 'x':
                input_tokens.extend(tokens_x)
            elif i == 'p':
                input_tokens.extend([self.prompt_token] * self.prompt_shape[p_idx])
                p_idx += 1
            else:
                input_tokens.extend(self.tokenizer.tokenize(self.template_dict['content'][i]))
            # i = 0, input_tokens = ['[CLS]', 'the', 'sentence', '"']
            # i = x, input_tokens = ['[CLS]', 'the', 'sentence', '"', 'post', 'match', ':', 'dean', 'smith', 'furious', 'after', 'penalty', 'de', '##ba', '##cle']
            # i = 1, input_tokens = ['[CLS]', 'the', 'sentence', '"', 'post', 'match', ':', 'dean', 'smith', 'furious', 'after', 'penalty', 'de', '##ba', '##cle', '"', 'has', '[MASK]', 'emotion', '[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)  
        # input_ids = [101, 1996, 6251, 1000, 2695, 2674, 1024, 4670, 3044, 9943, 2044, 6531, 2139, 3676, 14321, 1000, 2038, 103, 7603, 102]
        attention_mask = [1] * len(input_ids)
        # attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        labels = [-100] * len(input_ids)
        # labels = [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]

        for i, id in enumerate(input_ids):
            if id == self.tokenizer.mask_token_id: # tokenizer.mask_token_id = 103    {[MASK]: 103}
                labels[i] = label_id  # {'negative': 2919, 'neutral': 2053, 'positive': 2204} 中的 value
        # labels = [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 2919, -100, -100]

        assert len(input_ids) == len(attention_mask) == len(labels)
        return input_ids, attention_mask, labels

    def __len__(self):
        return len(self.lines)
        
    def __getitem__(self, idx):
        line = self.lines[idx]  # 第 idx+1 行，因为 lines.pop(0)
        # 以下注释为 processor = processors['mvsa-s']; template = 1 的情况
        # 以 datasets/mvsa-s/few-shot1.tsv 第1行数据为例
        label_id = self.label_id_map[line[1]]  # label_id_map = {'negative': 2919, 'neutral': 2053, 'positive': 2204}
                                               # line[1] 为 label 项
        img_id = line[2]
        text_x = line[3].lower()  # post match: dean smith furious after penalty debacle
        text_a = line[4].lower()  # 无内容
        if self.dataset in ['t2015', 't2017']:
            text_x = text_x.replace('$t$', text_a)

        tokens_x = self.tokenizer.tokenize(text_x)  # ['post', 'match', ':', 'dean', 'smith', 'furious', 'after', 'penalty', 'de', '##ba', '##cle']
        tokens_a = self.tokenizer.tokenize(text_a)  # []
        
        input_ids, attention_mask, labels = self._supervised_encode(tokens_x, tokens_a, label_id)
        # input_tokens = ['[CLS]', 'the', 'sentence', '"', 'post', 'match', ':', 'dean', 'smith', 'furious', 'after', 'penalty', 'de', '##ba', '##cle', '"', 'has', '[MASK]', 'emotion', '[SEP]']
        # input_ids = [101, 1996, 6251, 1000, 2695, 2674, 1024, 4670, 3044, 9943, 2044, 6531, 2139, 3676, 14321, 1000, 2038, 103, 7603, 102]
        # attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # labels = [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 2919, -100, -100]
        # labels -- 除了情感相关的单词为对应的vocab索引外，其余全为 -100

        if self.no_img:  # 无图像（预训练时）
            img = None
        else:  # PVLM
            img = self.img_dict[img_id]  # tensor形式的img
            if self.template == 1 or self.template == 2:
                # [CLS]             [textual template] [SEP]
                # [CLS] [Img] [SEP] [textual template] [SEP]
                addon = [self.img_token_id] * self.img_token_len + [self.tokenizer.sep_token_id]  # [Img] [SEP]
                # tokenizer.sep_token_id -- {[SEP]:102}
                # for img_token_len in 1 2 3 4 5, 以 img_token_len=1 为例
                # addon = [3] * 2 + [102] = [3, 102]
                input_ids = [input_ids[0]] + addon + input_ids[1:]
                # input_ids: [CLS] [textual template] [SEP] -> [CLS] [Img] [SEP] [textual template] [SEP]
                # input_ids = [101, 3, 102, 1996, 6251, 1000, 2695, 2674, 1024, 4670, 3044, 9943, 2044, 6531, 2139, 3676, 14321, 1000, 2038, 103, 7603, 102]
                attention_mask = [1] * (self.img_token_len + 1) + attention_mask  # [Img] [SEP] 的 attention_mask 加进去
                # attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                labels = [-100] * (self.img_token_len + 1) + labels  # [Img] [SEP] 的 labels 加进去
                # labels = [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 2919, -100, -100]
            elif self.template == 3:
                # [CLS]      [M][P][A][P][C][SEP]
                # [CLS][I][P][M][P][A][P][C][SEP]
                addon = [self.img_token_id] * self.img_token_len + [self.prompt_token_id] * self.prompt_img_len
                input_ids = [input_ids[0]] + addon + input_ids[1:]
                attention_mask = [1] * (self.img_token_len + self.prompt_img_len) + attention_mask
                labels = [-100] * (self.img_token_len + self.prompt_img_len) + labels

            assert len(input_ids) == len(attention_mask) == len(labels)
            
        return {
            'img_id': img_id,  # 248.jpg
            'img': img,  # tensor形式的img
            "text_x": text_x,  # post match: dean smith furious after penalty debacle
            "text_a": text_a,  # 无内容
            'input_ids': input_ids,  # [101, 3, 102, 1996, 6251, 1000, 2695, 2674, 1024, 4670, 3044, 9943, 2044, 6531, 2139, 3676, 14321, 1000, 2038, 103, 7603, 102]
            'attention_mask': attention_mask,  # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            'labels': labels  # [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 2919, -100, -100]
        }
    
    def collate_fn(self, batch):
        input_ids = [torch.tensor(instance["input_ids"]) for instance in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        attention_mask = [torch.tensor(instance["attention_mask"]) for instance in batch]
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        labels = [torch.LongTensor(instance["labels"]) for instance in batch]
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        returns = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if not self.no_img:
            imgs = [instance['img'] for instance in batch]
            imgs = torch.stack(imgs, dim=0)
            returns['imgs'] = imgs
        
        return returns
    