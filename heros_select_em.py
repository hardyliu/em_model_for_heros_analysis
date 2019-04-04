# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:21:10 2019

@author: hardyliu
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

data_ori= pd.read_csv('./heros.csv',encoding='gb18030')

features = [u'最大生命',u'生命成长',u'初始生命',u'最大法力', u'法力成长',
            u'初始法力',u'最高物攻',u'物攻成长',u'初始物攻',u'最大物防',u'物防成长',
            u'初始物防', u'最大每5秒回血', u'每5秒回血成长', u'初始每5秒回血',
            u'最大每5秒回蓝', u'每5秒回蓝成长', u'初始每5秒回蓝', u'最大攻速', u'攻击范围']
data = data_ori[features]

# 对英雄属性之间的关系进行可视化分析
# 设置 plt 正确显示中文
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
#plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

corr = data.corr()

plt.figure(figsize=(14,14))

sns.heatmap(corr,annot=True)
plt.show()

#通过上一步的热力图分析，取下面的特征作为后续分析的特征
features_remain = [u'最大生命', u'初始生命', u'最大法力', u'最高物攻', 
                   u'初始物攻', u'最大物防', u'初始物防', u'最大每5秒回血', 
                   u'最大每5秒回蓝', u'初始每5秒回蓝', u'最大攻速', u'攻击范围']


data = data_ori[features_remain]

data[u'最大攻速']=data[u'最大攻速'].apply(lambda x:float(x.strip('%'))/100)

data[u'攻击范围']=data[u'攻击范围'].map({'近战':0,'远程':1})
#对数据进行规范化处理
ss=StandardScaler()

data = ss.fit_transform(data)

from sklearn.mixture import GaussianMixture

#选择GMM模型
gmm = GaussianMixture(n_components=30,covariance_type='full')

gmm.fit(data)

prediction=gmm.predict(data)

print(prediction)

data_ori.insert(0,'分组',prediction)
#输出到CSV
data_ori.to_csv('./hero_out.csv',index=False,sep=',',encoding='gb18030')


from sklearn.metrics import calinski_harabaz_score
#分数越高代表聚类效果越好
print("Calinski-Harabaz:",calinski_harabaz_score(data, prediction))



