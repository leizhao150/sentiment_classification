# -*- coding: utf-8 -*-
# @Time    : 2021/5/26
# @Author  : leizhao150

import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nltk.tokenize as nk
import numpy as np
from tqdm import tqdm

torch.manual_seed(2021)

# 数据处理
class TextDataSet(Dataset):

    def __init__(self, path, vocab_path, max_len):
        super(TextDataSet, self).__init__()
        datas = pd.read_csv(path, names=['text', 'label'], sep=';')
        vocab = pd.read_csv(vocab_path, names=['words'])
        self.word2ix = {w:i for i, w in enumerate(vocab['words'])}
        self.texts = datas['text']
        self.labels = datas['label']
        self.max_len = max_len

    def __getitem__(self, i):
        text = self.texts.iloc[i]
        label = self.labels.iloc[i]
        tokens = [self.word2ix[i] for i in nk.word_tokenize(text)]
        if len(tokens) >= self.max_len:
            tokens = tokens[:self.max_len]
        else:
            pad = [self.word2ix['[PAD]']] * (self.max_len-len(tokens))
            tokens.extend(pad)
        return np.array(tokens), np.array(label)

    def __len__(self):
        return len(self.texts)

# 创建模型
class Model(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, seq_len,
                 n_classes, num_layers=2, bidirectional=True, dropout=0.5, is_add_att=False):
        super(Model, self).__init__()
        self.is_add_att = is_add_att
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional)
        self.weight = nn.Parameter(torch.zeros(hidden_dim*2))
        self.layer_norm = nn.LayerNorm(normalized_shape=[hidden_dim*2])
        self.fc = nn.Linear(hidden_dim*2, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(p=dropout)

    # 注意力机制层
    def attention_layer(self, lstm_output, input):
        lstm_output = lstm_output.transpose(0, 1)
        score = torch.matmul(torch.tanh(lstm_output), self.weight)
        score = torch.masked_fill(score, input.eq(0), float('-inf'))
        weights = torch.softmax(score, dim=-1)
        attention_output = lstm_output * weights.unsqueeze(-1)
        attention_output = self.fc(attention_output)
        return attention_output

    def forward(self, input):
        x = self.embedding(input)
        x = self.dropout(x)
        x = x.transpose(0, 1)
        output, (hn, cn) = self.lstm(x)
        # 是否加入注意力机制
        if self.is_add_att:
            x = self.attention_layer(output, input)
            x = x.transpose(1, 2)
            x1 = torch.max_pool1d(x, kernel_size=self.seq_len).squeeze(-1)
            x2 = torch.avg_pool1d(x, kernel_size=self.seq_len).squeeze(-1)
            x = torch.cat([x1, x2], dim=-1)
        else:
            x = hn.permute(1, 2, 0)
            x1 = torch.max_pool1d(x, kernel_size=2*self.num_layers).squeeze(-1)
            x2 = torch.avg_pool1d(x, kernel_size=2*self.num_layers).squeeze(-1)
            x = torch.cat([x1, x2], dim=-1)
        x = self.layer_norm(x)
        x = self.fc(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out(x)
        return x


def train(is_add_att):
    # 定义模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(vocab_size=11705, embed_dim=128, hidden_dim=256, seq_len=128,
                  n_classes=6, num_layers=2, bidirectional=True, is_add_att=is_add_att).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
    criterion = nn.CrossEntropyLoss()

    # 数据加载器
    train_dataSet = TextDataSet('../datas/train.txt', '../datas/vocab.txt', 128)
    val_dataSet = TextDataSet('../datas/val.txt', '../datas/vocab.txt', 128)
    train_dataloader = DataLoader(train_dataSet, batch_size=64)
    val_dataloader = DataLoader(val_dataSet, batch_size=64)

    # 训练和测试
    best_f1 = 0
    for epoch in range(20):
        model.train()
        target, pred, losses = [], [], []
        with tqdm(train_dataloader) as train_pbar:
            for x, y in train_pbar:
                x = torch.as_tensor(x, dtype=torch.long).to(device)
                y = torch.as_tensor(y, dtype=torch.long).to(device)
                out = model(x)
                loss = criterion(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_pbar.set_description("train: loss:%s" % (loss.item()))
                out = torch.argmax(torch.softmax(out, dim=-1), dim=-1)
                target.extend(y.detach().cpu().tolist())
                pred.extend(out.detach().cpu().tolist())
                losses.append(loss.item())
        # 学习率衰减
        scheduler.step(np.mean(losses))
        # 计算评估指标
        p = precision_score(target, pred, average='macro', zero_division=0)
        r = recall_score(target, pred, average='macro', zero_division=0)
        f1 = f1_score(target, pred, average='macro', zero_division=0)
        print("train: p:%s->r:%s->f1:%s" % (p, r, f1))
        print("-" * 80)

        model.eval()
        target, pred = [], []
        with tqdm(val_dataloader) as val_pbar:
            for x, y in val_pbar:
                x = torch.as_tensor(x, dtype=torch.long).to(device)
                y = torch.as_tensor(y, dtype=torch.long).to(device)
                out = model(x)
                loss = criterion(out, y)
                out = torch.argmax(torch.softmax(out, dim=-1), dim=-1)
                val_pbar.set_description("val: loss:%s" % (loss.item()))
                target.extend(y.detach().cpu().tolist())
                pred.extend(out.detach().cpu().tolist())
        # 计算评估指标
        p = precision_score(target, pred, average='macro', zero_division=0)
        r = recall_score(target, pred, average='macro', zero_division=0)
        f1 = f1_score(target, pred, average='macro', zero_division=0)
        print("val: p:%s->r:%s->f1:%s" % (p, r, f1))
        print("=" * 80)
        # 保存最好的模型
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), open('./model/model_%s.bin'%is_add_att, 'wb'))

def test(is_add_att):
    # 定义模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(vocab_size=11705, embed_dim=128, hidden_dim=256, seq_len=128,
                  n_classes=6, num_layers=2, bidirectional=True, is_add_att=is_add_att).to(device)
    model.load_state_dict(torch.load(open('./model/model_%s.bin'%is_add_att, 'rb')))
    criterion = nn.CrossEntropyLoss()
    print(model)

    # 数据加载器
    test_dataSet = TextDataSet('../datas/test.txt', '../datas/vocab.txt', 128)
    test_dataloader = DataLoader(test_dataSet, batch_size=128)

    # 评测
    model.eval()
    target, pred = [], []
    with tqdm(test_dataloader) as val_pbar:
        for x, y in val_pbar:
            x = torch.as_tensor(x, dtype=torch.long).to(device)
            y = torch.as_tensor(y, dtype=torch.long).to(device)
            out = model(x)
            loss = criterion(out, y)
            out = torch.argmax(torch.softmax(out, dim=-1), dim=-1)
            val_pbar.set_description("val: loss:%s" % (loss.item()))
            target.extend(y.detach().cpu().tolist())
            pred.extend(out.detach().cpu().tolist())
    # 计算评估指标
    p = precision_score(target, pred, average='macro', zero_division=0)
    r = recall_score(target, pred, average='macro', zero_division=0)
    f1 = f1_score(target, pred, average='macro', zero_division=0)
    print("p:%s->r:%s->f1:%s" % (round(p * 100, 3), round(r * 100, 3), round(f1 * 100, 3)))

