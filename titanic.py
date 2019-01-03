# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

data_train = pd.read_csv('/home/cui/kaggle/train.csv')
fig = plt.figure()
fig.set(alpha=0.2)

plt.subplot2grid((2, 3), (0, 0))
data_train.Survived.value_counts().plot(kind='bar')  # 柱状图
plt.title('Surivived(1 survived)')  # 标题
plt.ylabel(u'Suriv_num')

plt.subplot2grid((2, 3), (0, 1))
data_train.Pclass.value_counts().plot(kind='bar')  # 柱状图
plt.title('Num')  # 标题
plt.ylabel('Pclass')

plt.subplot2grid((2, 3), (0, 2))
# data_train.Survived.value_counts().plot(kind='bar') #柱状图
plt.scatter(data_train.Survived, data_train.Age)
plt.title('Surivived')  #标题
plt.ylabel('Age')

plt.subplot2grid((2, 3), (1, 0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.ylabel('Age')
plt.ylabel('density')
# data_train.Survived.value_counts().plot(kind='bar') #柱状图
plt.title('Age of Pclass')  # 标题
plt.legend(('1P', '2P', '3P'), loc='best')

plt.subplot2grid((2, 3), (1, 2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title('Embarked')
plt.ylabel('num')
plt.show()