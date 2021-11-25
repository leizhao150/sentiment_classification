# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 20:22
# @Author  : leizhao150
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import nn, optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import OpenAIGPTConfig, OpenAIGPTTokenizerFast, \
    OpenAIGPTModel, BertTokenizer, BertConfig, BertModel

torch.manual_seed(2021)

bert_path = r'F:\tools\bert-large-uncased'
gpt_path = r'F:\tools\gpt-model'

# 数据处理
class TextDataSet(Dataset):

    def __init__(self, path, max_len, pr='bert'):
        super(TextDataSet, self).__init__()
        self.datas = pd.read_csv(path, names=['text', 'label'], sep=';')
        if pr == 'gpt':
            self.tokenizer = OpenAIGPTTokenizerFast.from_pretrained(gpt_path)
            self.tokenizer.pad_token = self.tokenizer.unk_token
        else:
            self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.max_len = max_len

    def __getitem__(self, item):
        text, label = self.datas.text[item], self.datas.label[item]
        encode = self.tokenizer.encode_plus(text = text, max_length = self.max_len,
                    padding = 'max_length', truncation = True, return_tensors = 'pt')
        label = torch.as_tensor(label, dtype=torch.long)
        return encode, label

    def __len__(self):
        return len(self.datas)

class Model(nn.Module):

    def __init__(self, num_classes, hidden_dim=128, dropout=0.5, pr='bert'):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.pr = pr

        if pr == 'gpt':
            self.config = OpenAIGPTConfig.from_pretrained(gpt_path)
            self.pr = OpenAIGPTModel.from_pretrained(gpt_path, config=self.config)
            self.embed_size = self.config.hidden_size
        else:
            self.config = BertConfig.from_pretrained(bert_path)
            self.pr = BertModel.from_pretrained(bert_path, config=self.config)
            self.embed_size = self.config.hidden_size

        self.layer_norm1 = nn.LayerNorm(normalized_shape=self.embed_size)
        self.fc = nn.Linear(self.embed_size, hidden_dim)
        self.max_pool = nn.AdaptiveMaxPool1d(output_size = 1)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        if self.pr == 'bert':
            _, x = self.pr(**input)
        else:
            x = self.pr(**input)
            x = x[0].transpose(1, 2)
            x = self.max_pool(x).squeeze(-1)
        x = self.layer_norm1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.layer_norm2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        return x

# 训练函数
def train(pr, bs=16):
    # 定义模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(num_classes=6, hidden_dim=128, dropout=0.5, pr=pr).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
    criterion = nn.CrossEntropyLoss()

    # 数据加载器
    train_dataSet = TextDataSet('./datas/train.txt', 64, pr=pr)
    val_dataSet = TextDataSet('./datas/val.txt', 64, pr=pr)
    train_dataloader = DataLoader(train_dataSet, batch_size=bs)
    val_dataloader = DataLoader(val_dataSet, batch_size=bs)

    # 训练和测试
    best_f1 = 0
    for epoch in range(20):
        model.train()
        target, pred, losses = [], [], []
        with tqdm(train_dataloader) as train_pbar:
            for inputs, labels in train_pbar:
                inputs = {k: v.squeeze(1).to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_pbar.set_description("train: loss:%s" % (round(loss.item(), 3)))
                outputs = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
                target.extend(labels.detach().cpu().tolist())
                pred.extend(outputs.detach().cpu().tolist())
                losses.append(loss.item())
        # 学习率衰减
        scheduler.step(np.mean(losses))
        # 计算评估指标
        p = precision_score(target, pred, average='macro', zero_division=0)
        r = recall_score(target, pred, average='macro', zero_division=0)
        f1 = f1_score(target, pred, average='macro', zero_division=0)
        print("train: p:%s->r:%s->f1:%s" % (round(p, 4), round(r, 4), round(f1, 4)))
        print("-" * 80)

        model.eval()
        target, pred = [], []
        with tqdm(val_dataloader) as val_pbar:
            for inputs, labels in val_pbar:
                inputs = {k: v.squeeze(1).to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                outputs = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
                val_pbar.set_description("val: loss:%s" % (round(loss.item(), 4)))
                target.extend(labels.detach().cpu().tolist())
                pred.extend(outputs.detach().cpu().tolist())
        # 计算评估指标
        p = precision_score(target, pred, average='macro', zero_division=0)
        r = recall_score(target, pred, average='macro', zero_division=0)
        f1 = f1_score(target, pred, average='macro', zero_division=0)
        print("val: p:%s->r:%s->f1:%s" % (round(p, 4), round(r, 4), round(f1, 4)))
        print("=" * 80)
        # 保存最好的模型
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), open('./model/model_%s.bin' % pr, 'wb'))

def test(pr, bs):
    # 定义模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(num_classes=6, hidden_dim=128, dropout=0.5, pr=pr).to(device)
    model.load_state_dict(torch.load(open('./model/model_%s.bin'%pr, 'rb')))
    criterion = nn.CrossEntropyLoss()
    print(model)

    # 数据加载器
    test_dataSet = TextDataSet('./datas/val.txt', 64, pr=pr)
    test_dataloader = DataLoader(test_dataSet, batch_size=bs)

    # 评测
    model.eval()
    target, pred = [], []
    with tqdm(test_dataloader) as test_pbar:
        for inputs, labels in test_pbar:
            inputs = {k: v.squeeze(1).to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            outputs = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            test_pbar.set_description("val: loss:%s" % (loss.item()))
            target.extend(labels.detach().cpu().tolist())
            pred.extend(outputs.detach().cpu().tolist())
    # 计算评估指标
    p = precision_score(target, pred, average='macro', zero_division=0)
    r = recall_score(target, pred, average='macro', zero_division=0)
    f1 = f1_score(target, pred, average='macro', zero_division=0)
    print("p:%s->r:%s->f1:%s" % (round(p * 100, 3), round(r * 100, 3), round(f1 * 100, 3)))