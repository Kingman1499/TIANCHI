import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split # 数据分割ˇ
from sklearn.metrics import classification_report # 学习器评估
from sklearn import svm # 支持向量机
from sklearn import metrics

data = pd.read_csv("E:/人工智能/实验/大数据/数据/dataSet.csv", index_col="time")

data['label_y'] = data['4']	#以购买操作作为标记

outputSet = data.ix['2014-12-18']
outputSet = outputSet.loc[outputSet['3'] == 1]

X = data.ix[:, ['user_id', 'item_id', 'item_category', '3', '4', 'time_mark']]
y = data['label_y']
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = svm.SVC(C=100, class_weight='balanced')
# 训练模型
clf.fit(X_train, y_train)
# 预测
predict = clf.predict(X_test)
# 评估
print(clf.score(X_test, y_test))
print(classification_report(y_test, predict))
print(metrics.f1_score(y_test, predict, average='weighted'))

X = outputSet.ix[:, ['user_id', 'item_id', 'item_category', '3', '4', 'time_mark']]
# 预测
output = clf.predict(X)
X['output'] = output
X = X.loc[X['output'] > 0.0]
X = X.ix[:, ['user_id', 'item_id']]
del X['time']
# 保存结果
X.to_csv('tianchi_mobile_recommendation_predict.csv', index=False)
