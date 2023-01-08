import timm
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM


class PromptEncoder(nn.Module):
    '''learnable token generator modified from P-tuning
    https://github.com/THUDM/P-tuning
    '''
    def __init__(self, prompt_token_len, hidden_size, device, lstm_dropout):
        super().__init__()
        print("[#] Init prompt encoder...")
        # Input [0,1,2,3,4,5,6,7,8]
        self.seq_indices = torch.LongTensor(list(range(prompt_token_len))).to(device)
        # Embedding
        self.embedding = nn.Embedding(prompt_token_len, hidden_size)
        # LSTM
        self.lstm_head = nn.LSTM(input_size=hidden_size,
                                       hidden_size=hidden_size // 2,
                                       num_layers=2,
                                       dropout=lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, hidden_size))

    def forward(self):
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        return output_embeds


class VisualEncoder(nn.Module):
    def __init__(self, model_name, img_token_len, embedding_dim):
        super().__init__()
        self.is_resnet = False
        self.img_token_len = img_token_len
        self.embedding_dim = embedding_dim
        self.backbone = timm.create_model(model_name, pretrained=True)  # 这里 model_name 是 args.visual_model_name
        # parser.add_argument("--visual_model_name", type=str, choices=VISUAL_MODELS, default='nf_resnet50')
        # VISUAL_MODELS = ['nf_resnet50', 'resnet50', 'resnetv2_50x1_bitm', 'vit_base_patch16_224']
        if "resnet" in model_name:
            self.is_resnet = True
            if model_name == "resnet50":
                self.global_pool = self.backbone.global_pool
            else:
                self.global_pool = self.backbone.head.global_pool
            self.visual_mlp = nn.Linear(2048, img_token_len * embedding_dim)  # 2048 -> n * 768
        elif "vit" in model_name:
            self.visual_mlp = nn.Linear(768, img_token_len * embedding_dim)  # 768 -> n * 768
        
    def forward(self, imgs_tensor):
        bs = imgs_tensor.shape[0]
        visual_embeds = self.backbone.forward_features(imgs_tensor)
        if self.is_resnet:
            visual_embeds = self.global_pool(visual_embeds).reshape(bs, 2048)
        visual_embeds = self.visual_mlp(visual_embeds)
        visual_embeds = visual_embeds.reshape(bs, self.img_token_len, self.embedding_dim)

        return visual_embeds


class MSAModel(torch.nn.Module):
    '''main model
    '''
    def __init__(self, args, label_id_list):
        super().__init__()
        self.args = args
        self.label_id_list = label_id_list

        # parser.add_argument("--model_name", type=str, default='bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained(args.model_name, local_files_only=True)
        self.lm_model = BertForMaskedLM.from_pretrained(args.model_name)

        self.embeddings = self.lm_model.bert.get_input_embeddings()  # embeddings = Embedding(30522, 768, padding_idx=0)
        self.embedding_dim = self.embeddings.embedding_dim  # 768

        if not args.no_img:  # 有img
            self.img_token_id = self.tokenizer.get_vocab()[args.img_token]  # img_token_id=3, img_token='[unused2]'
            self.img_token_len = args.img_token_len
            self.visual_encoder = VisualEncoder(args.visual_model_name, self.img_token_len, self.embedding_dim)

        if args.template == 3:
            self.prompt_token_id = self.tokenizer.get_vocab()[args.prompt_token]
            self.prompt_token_len = sum([int(i) for i in args.prompt_shape.split('-')[0]]) + int(args.prompt_shape[-1])
            self.prompt_encoder = PromptEncoder(self.prompt_token_len, self.embedding_dim, args.device, args.lstm_dropout)

    def embed_input(self, input_ids, imgs=None):
        bs = input_ids.shape[0]
        embeds = self.embeddings(input_ids)

        if self.args.template == 3:
            prompt_token_position = torch.nonzero(input_ids == self.prompt_token_id).reshape((bs, self.prompt_token_len, 2))[:, :, 1]
            prompt_embeds = self.prompt_encoder()
            for bidx in range(bs):
                for i in range(self.prompt_token_len):
                    embeds[bidx, prompt_token_position[bidx, i], :] = prompt_embeds[i, :]
        
        if not self.args.no_img:  # 有img
            visual_embeds = self.visual_encoder(imgs)
            img_token_position = torch.nonzero(input_ids == self.img_token_id).reshape((bs, self.img_token_len, 2))[:, :, 1]
            for bidx in range(bs):
                for i in range(self.img_token_len):
                    embeds[bidx, img_token_position[bidx, i], :] = visual_embeds[bidx, i, :]
        
        return embeds
    
    def forward(self, input_ids, attention_mask, labels, imgs=None):  # 这些参数都是 MSADataset __getitem__ 中的返回值
        inputs_embeds = self.embed_input(input_ids, imgs)
        output = self.lm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        loss, logits = output.loss, output.logits  # logits 得到序列中每一个词的概率 
                                                   # logits: torch.Size([1, 13, 30522])  13为句子长度, 30522为vocab单词数

        pred = logits[labels != -100]  # 把labels中为-100的(非情感标签)都去掉, 只保留情感标签的logits, size=[1, 30522]
        probs = pred[:, self.label_id_list]  # 得到vocab中 [2919, 2053, 2204](['bad', 'no', 'good']对应的索引) 的分别概率, size=[1,3]
        pred_labels_idx = torch.argmax(probs, dim=-1).tolist()  # 最后一个维度的最大值的索引
        y_ = [self.label_id_list[i] for i in pred_labels_idx]  # 预测的情感标签, [2919, 2053, 2204]其中之一

        y = labels[labels != -100]  # 真实的情感标签

        return loss, y_, y.tolist()
