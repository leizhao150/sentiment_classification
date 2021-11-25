# -*- coding: utf-8 -*-
# @Time    : 2021/5/26 23:13
# @Author  : leizhao150

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. 读取数据
train_datas = pd.read_csv('./datas/train.txt', sep=';', names=['text', 'label'])
dev_datas = pd.read_csv('./datas/val.txt', sep=';', names=['text', 'label'])
test_datas = pd.read_csv('./datas/test.txt', sep=';', names=['text', 'label'])
x_train, y_train = train_datas['text'].tolist(), train_datas['label'].tolist()
x_dev, y_dev = dev_datas['text'].tolist(), dev_datas['label'].tolist()
x_test, y_test = test_datas['text'].tolist(), test_datas['label'].tolist()

# 2. TF-IDF特征化
tf = TfidfVectorizer()
x_train = tf.fit_transform(x_train)
x_dev = tf.transform(x_dev)
x_test = tf.transform(x_test)

# 3. 使用卡方检验，选择1000个最佳特征
sk = SelectKBest(chi2, k=1000)
x_train = sk.fit_transform(x_train, y_train)
x_dev = sk.transform(x_dev)
x_test = sk.transform(x_test)

# 4. 归一化
mms = MaxAbsScaler()
x_train = mms.fit_transform(x_train)
x_dev = mms.fit_transform(x_dev)
x_test = mms.fit_transform(x_test)

parameters = {
    'kernel': ['rbf', 'linear'],
    'C': np.logspace(-3, 3, 7),
    'gamma': np.logspace(-3, 3, 7)
}
model = SVC(random_state=1)
model = GridSearchCV(model, param_grid=parameters, cv=5, n_jobs=14)
model.fit(x_train, y_train)
pred = model.predict(x_dev)

# 验证集结果
print("最优参数:", model.best_params_)
print('验证集：')
print(">>准确率:%.3f"%(accuracy_score(y_dev, pred)*100))
print(">>精确率:%.3f"%(precision_score(y_dev, pred, average='macro')*100))
print(">>召回率:%.3f"%(recall_score(y_dev, pred, average='macro')*100))
print(">>F1值:%.3f"%(f1_score(y_dev, pred,average='macro')*100))

# 测试集结果
pred = model.predict(x_test)
print('测试集：')
print(">>准确率:%.3f"%(accuracy_score(y_test, pred)*100))
print(">>精确率:%.3f"%(precision_score(y_test, pred, average='macro')*100))
print(">>召回率:%.3f"%(recall_score(y_test, pred, average='macro')*100))
print(">>F1值:%.3f"%(f1_score(y_test, pred,average='macro')*100))

