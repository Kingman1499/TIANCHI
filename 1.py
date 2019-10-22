'''
数据预处理
'''
import pandas as pd
import numpy as np

print("读取数据。")
Duser=pd.read_csv('E:/人工智能/实验/大数据/数据/tianchi_fresh_comp_train_user.csv')
Pitem=pd.read_csv('E:/人工智能/实验/大数据/数据/tianchi_fresh_comp_train_item.csv')
print("数据读取完毕！")

'''
user_geohash列缺省值过多，故删除user_geohash列
'''
del Duser["user_geohash"]
print("processing time column")
Duser['time'] = Duser['time'].astype(str)	# transform df['time'] to string type so we can use str's attribution
Duser['time'] = Duser['time'].str.slice(0, 10)	# 节选Time中前十个元素，也就是年月日被保留下来，时间剔除
Duser = Duser.loc[Duser['time'].str.contains('2014-12-12') == False] # loc：用于通过行/列标签索引数据，把除了双十二的数据放入D集合中
print("process time column DONE!")



print("processing item columns")
del Pitem["item_geohash"]

Pitem['item_id'] = Pitem['item_id'].astype(str)	#transform itemP['item_id'] to string type
itemsub = set(Pitem['item_id']) # P表里面的itemsub集合用于储存item_id
Duser['item_id'] = Duser['item_id'].astype(str)
itemlst = list(Duser['item_id'])#用于对比D集合里面的item有没有P集合里没有的，没有的则删除
itemmark = list() #创建一个列表存商品数据


for item in itemlst:
    if item not in itemsub:
        itemmark.append(False)
    else:
        itemmark.append(True)

Duser['item_mark'] = itemmark # get new column ['item_mark']
Duser['item_mark'] = Duser['item_mark'].astype(str)
Duser = Duser.loc[Duser['item_mark'].str.contains('True')]	#用loc索引只要item_mark为true的数据
print("process item columns DONE!")

Duser.to_csv("result.csv", index=False) # save as new file.csv


'''
对用户行为进行编码
1-look,2-save,3-putin,4-buy
'''
'''
used for one-hot encoding
'''
Duser = pd.read_csv('result.csv')		# 读取文件
result2 = pd.get_dummies(Duser['behavior_type'])	# one-hot encoding
#result2.rename(columns={'Unnamed': 'id'}, inplace=True)

#idlst = list(range(1970968))
#Duser['id'] = idlst
Duser = Duser.join(result2)#与原有的数据合并
#Duser = pd.merge(Duser, result2, left_on='index', right_on='index') # 合并DataFrame

del Duser['behavior_type']
#del Duser['id']
del Duser['item_mark']
#Duser.rename(columns={'1': 'look', '2': 'save', '3': 'putin', '4': 'buy'}, inplace=True)#对列名重命名
Duser.rename(columns={'1': 'look'}, inplace=True)
Duser.to_csv('result2.csv', index=False) # save as new file.csv


'''
标记日期
'''

Duser = pd.read_csv('result2.csv')
Duser['time'] = pd.to_datetime(Duser['time']) # transform from str to datetime so that we can use datetime.attrinbution for judgement
timemark = list() # 将判断结果储存在新list中

def judge(date):	# 判断日期

    if ((date.month==12 and date.day==17) or (date.month==12 and date.day==18)):
        timemark.append(1)
    else:
        timemark.append(0)


Duser['time'].apply(judge)
Duser['time_mark'] = timemark  # mark 2 days early than 2014-12-19(_2014-12-17_,_2014-12-18_)
Duser.to_csv('tmmk_vsr.csv')

Duser = pd.read_csv('tmmk_vsr.csv')
Duser.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
Duser['2days'] = Duser['4'] * Duser['time_mark'] # 标记处19号前两天的购买数据
Duser.to_csv('2days.csv', index=False) # save as new file.csv




'''
加权
'''


df = pd.read_csv('2days.csv')
df['1'].value_counts()

# df['look'].sum() = 1863827
# df['like'].sum() = 32506
# df['putin'].sum() = 53646
# df['buy'].sum() = 20989
df['wight'] = (df['1'] * (20989 / 1863827) + df['2'] * (20989 / 32506) + df['3'] * (20989 / 53646) + df['3'] + df['time_mark']) * ((2 - df['2days']) / 2)
df.to_csv('relation.csv', index=False) # save as new file.csv


