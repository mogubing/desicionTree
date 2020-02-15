# -*- coding: utf-8 -*-
import numpy as np
from graphviz import Graph
from sklearn import tree
# 创建数据[红，大]，1==是，0==否
data=np.array([[1,1],[1,0],[0,1],[0,0]])
# 数据标注：1==好苹果，0==坏苹果
target=np.array([1,1,0,0])
clf=tree.DecisionTreeClassifier(criterion='entropy')
clf=clf.fit(data,target)
with open("./decisiontree200114/apple.dot",'w',encoding='utf-8') as f:
    # feature_names根据数据的特征，一一对应的写上名字。比如本例：数据是苹果大不大和苹果红不红。
    f=tree.export_graphviz(clf,feature_names=['big apple','red apple'],out_file=f)
# 画出来(不用也可以)
# graph1=Graph(gra)
predit_result=clf.predict([[1,1],[0,0],[0,1],[1,0],[0,0],[1,1]])
print(predit_result)
print(clf.score(data,target))
print(clf)

# 关于决策树可视化的使用
# 下载graphviz并将其安装路径贴到path下（环境变量）
# 代码中使用with open将tree.export_graphviz的文件存到当前路径下，如生成文件apple.dot
# 然后打开控制台cmd，在apple.dot目录下，dot -Tpng apple.dot -o apple.png 打开apple.png,即可看见决策树图片啦

