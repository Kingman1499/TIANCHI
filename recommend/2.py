import pandas as pd
import numpy as np
data = pd.read_csv("relation.csv")
miniDataSet = data.loc[data['1'] != 1] # 筛除浏览样本
del miniDataSet['id']
del miniDataSet['1']
miniDataSet.to_csv("dataSet.csv", index=False) # 保存为新的数据集
