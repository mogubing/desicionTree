# -*- coding: utf-8 -*-
_author_ = 'huihui.gong'
_date_ = '2020/1/15'

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import model_selection
from sklearn import tree
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score,precision_score,average_precision_score,precision_recall_curve,plot_precision_recall_curve
import graphviz
from matplotlib import pyplot as plt
irisdata=pd.read_csv("D:\\PycharmProjects\\actualProject1116\\decisiontree200114\\datas\\iris.data",header=None)
# print(irisdata)
print(irisdata.shape)
x=irisdata.iloc[:,0:4]
# 将字符转化为数字
y=pd.Categorical(irisdata.iloc[:,4]).codes
# print(y)
# print(x.describe())
# print(x.shape)
# print(y.shape)
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.3,random_state=10)
# 1、数据标准化（标准正态分布、区间缩放、单位向量）
ss=preprocessing.MinMaxScaler()
x_train=ss.fit_transform(x_train,y_train)
x_test=ss.transform(x_test)
# print(x_train)
# 2、特征选择（卡方系数SelectKbest、方差选择法）
# k是指选取几个特征，也可以是all
# ch2=SelectKBest(chi2,k=3)
# x_train=ch2.fit_transform(x_train,y_train)
# x_test=ch2.transform(x_test)
# # 获取特征得分
# print(ch2.scores_)
# print(ch2.get_support(indices=True))
# # 第二个变量：花萼宽度被删除
# print(x_train.shape)
# 3、降维（PCA）
# pca=PCA(n_components=2)
# x_train=pca.fit_transform(x_train,y_train)
# x_test=pca.transform(x_test)
# print(x_train)
# print(x_train.shape)
# 构建模型
irisclf=tree.DecisionTreeClassifier(criterion='entropy')
irisclf.fit(x_train,y_train)
# 十交叉验证
cross_score=model_selection.cross_validate(irisclf,x_train,y_train,cv=10,n_jobs=5,scoring='accuracy')
print(np.mean(cross_score['test_score']))
# 画出决策树图，结合安装软件，graphviz，然后cmd执行：dot -Tpng apple.dot -o apple.png
# with open("D:\\PycharmProjects\\actualProject1116\\decisiontree200114\\iris.dot", 'w', encoding='utf-8') as f:
#     f=tree.export_graphviz(irisclf,feature_names=["huaeleng","huabanleng","huabanwide"],out_file=f)
predit_y=irisclf.predict(x_test)
print(irisclf.score(x_test,y_test))
result_compare=(predit_y==y_test)
print("准确率为:%f"%(np.mean(result_compare)))
# 折腾一番后，分值不如最开始的0.967,所以去掉第三步降维，分数为0.978，去掉第二步，还是0.978
# 常用评价模型好坏的指标如下：
# 准确率(accuracy)计算公式为：
# 注：准确率是我们最常见的评价指标，而且很容易理解，就是被分对的样本数除以所有的样本数，通常来说，正确率越高，分类器越好。
# 除了看准确率还需要结合精确率（精度）和召回率来评价模型的好坏
# 精确率:被划分为正例中，实为正例的比率
# 召回率:有多少正例被划分为正例
print(precision_score(y_test,predit_y,average='weighted'))
print(recall_score(y_test,predit_y,average='weighted'))
# print(average_precision_score(y_test,predit_y))
# plt_roc=plot_precision_recall_curve(irisclf,x_test,y_test)
# print(plt_roc)1








