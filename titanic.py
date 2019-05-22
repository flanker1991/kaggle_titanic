import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,AdaBoostClassifier,RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV,cross_val_score,cross_val_predict,StratifiedKFold,learning_curve,validation_curve,train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,confusion_matrix,silhouette_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans

#定义各类函数
#kmeans聚类SSE绘图和轮廓系数计算
def select_clusters(data):
    sse=[]
    for i in range(2,10):
        km=KMeans(n_clusters=i,n_init=5,n_jobs=-1)
        km.fit(data)
        print(i,silhouette_score(data,km.labels_))
        sse.append(km.inertia_)
    plt.figure(figsize=(12,8))
    plt.plot(range(2,10),sse)
    plt.xlabel('number of clusters')
    plt.ylabel('SSE')
    
#分组间隔点计算
def interval_points(data,k):  
    km=KMeans(n_clusters=k,n_init=5,n_jobs=-1)
    km.fit(data)
    a=pd.DataFrame(km.cluster_centers_).sort_values(by=0)
    a.columns=['a']
    a=a.reset_index(drop=True)
    b=a.loc[0:k-2]
    c=a.loc[1:k-1].reset_index(drop=True)
    d=pd.concat([b,c],ignore_index=True,sort=False,axis=1)
    d.columns=['a','b']
    return (d.a+d.b)/2

#特征重要度排序
def feature_importances(clf,x,y):
    clf.fit(x,y)
    importance=clf.feature_importances_
    index=importance.argsort()[::-1]
    plt.figure(figsize=(16,12))
    plt.bar(range(x.shape[1]),importance[index])
    plt.xticks(range(x.shape[1]),x.columns[index],fontsize=25,rotation=90)
    plt.yticks(fontsize=25)
    for i in range(x.shape[1]):
        print(i,x.columns[index[i]],importance[index[i]])

#选择前k个特征
def feature_select(clfs,x,y,k):
    features_top_k=pd.Series()
    for i, clf in enumerate(clfs):
        clf.fit(x,y)
        importance=clf.feature_importances_
        feature_sorted= pd.DataFrame({'feature': list(x_train), 'importance': importance}).sort_values('importance', ascending=False)
        features = feature_sorted.head(k)['feature']
        features_top_k=pd.concat([features_top_k,features],ignore_index=True).drop_duplicates()
    return features_top_k

#学习曲线
def plot_learning_curve(x,y,clf,title):
    plt.figure(figsize=(16,12))
    plt.title(title,fontsize=25)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        clf, x, y, cv=5, n_jobs=-1, train_sizes=np.linspace(.1,1.0,10))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt

#分类算法网格搜索调优
def gscv(x,y,clf,param_grid):
        gs=GridSearchCV(clf,param_grid=param_grid,cv=5,scoring='accuracy',n_jobs=-1)
        gs.fit(x,y)
        print(gs.best_score_,gs.best_params_)
        return gs.best_estimator_

#回归算法网格搜索调优
def gscvr(x,y,clf,param_grid):
        gs=GridSearchCV(clf,param_grid=param_grid,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)
        gs.fit(x,y)
        print(gs.best_score_,gs.best_params_)
        return gs.best_estimator_

#混淆矩阵
def plot_confusion_matrix(x,y,clf,title):
    y_pred = cross_val_predict(clf, x, y, cv=5)
    cm = confusion_matrix(y_pred, y)
    fig, ax = plt.subplots(figsize=(12,8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, s=cm[i,j],
                    ha="center", va="center",
                    color="white" if cm[i, j]>cm.max()/2 else "black")
    return ax

#假正例，假反例提取
def find_FP_FN(x,y,clf):
    FP_index=[]
    FN_index=[]
    y_pred = cross_val_predict(clf, x, y, cv=5)
    for i in range(len(y)):
        if y[i]==1 and y_pred[i]==0:
            FN_index.append(i)
        elif y[i]==0 and y_pred[i]==1:
            FP_index.append(i)
    return FP_index,FN_index

#stacking建模
def get_stack_data(clfs, x_train, y_train, x_test):
    stack_train=np.zeros((len(x_train),len(clfs)))
    stack_test=np.zeros((len(x_test),len(clfs)))
    for j, clf in enumerate(clfs):
        oof_test = np.zeros((len(x_test),5))
        skf=StratifiedKFold(n_splits =5,shuffle=False)
        for i,(train_s_index,test_s_index) in enumerate(skf.split(x_train,y_train)):
            x_train_s=x_train.loc[train_s_index]
            y_train_s=y_train.loc[train_s_index]
            x_test_s=x_train.loc[test_s_index]
            clf.fit(x_train_s,y_train_s)
            stack_train[test_s_index,j]=clf.predict_proba(x_test_s)[:,1]
            oof_test[:,i]=clf.predict_proba(x_test)[:,1]
        stack_test[:,j]=oof_test.mean(axis=1)
    return stack_train,stack_test

#多分类器得分计算
def multimodel_accuracy(clfs):
    scores=[]
    for i,(name,clf) in enumerate(clfs.items()):
        score=cross_val_score(clf,x_train,y_train,cv=5,scoring='accuracy').mean()
        scores.append(score)
    for i,(name,clf) in enumerate(clfs.items()):
        print(i,name,scores[i])

#计算分类器相关度
def pred_matrix_corr(x,y,clfs):
    names=[]
    pred_matrix=np.zeros((len(x),len(clfs)))
    pred_matrix=pd.DataFrame(pred_matrix)
    for j,(name,clf) in enumerate(clfs.items()):
        y_pred=cross_val_predict(clf,x,y,cv=5,method='predict_proba')[:,1]
        pred_matrix.iloc[:,j]=y_pred
        names.append(name)
    pred_matrix.columns=names
    plt.figure(figsize=(16,16))
    sns.heatmap(pred_matrix.corr(),linewidths=0.2,vmax=1.0,square=True,linecolor='white', annot=True)

#一、数据分析与可视化
#1.1数据概况
#导入数据
train=pd.read_csv('D:/data_project/titanic/train.csv')
test=pd.read_csv('D:/data_project/titanic/test.csv')
submission=pd.read_csv('D:/data_project/titanic/gender_submission.csv')
pd.set_option("display.max_columns",None)
#合并测试集与训练集
total_initial=pd.concat([train,test],ignore_index=True,sort=False)
total_initial.columns=['pid','survived','pclass','name','sex','age','sibsp','parch','ticket','fare','cabin','embarked']
total=total_initial.copy()
#总览数据
total.head()
total.info()
total.isnull().sum()
total.describe()
total_initial.hist(bins=50,figsize=(20,15))

#1.1不同pclass的生存率
train=total.loc[:890]
plt.figure(figsize=(12,8))
train.groupby('pclass').survived.mean().plot.bar(fontsize=25)
plt.xlabel('pclsss',fontsize=25)
plt.xticks(fontsize=25,rotation=0)

#1.2name,title
#提取姓名中的称谓
total['title']=total['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train=total.loc[:890]
#称谓类型和数量统计
pd.value_counts(total.title)
#不同称谓的生存率
plt.figure(figsize=(16,8))
train.groupby(['title'])['survived'].mean().plot.bar(fontsize=25)
plt.xlabel('title',fontsize=25)
plt.xticks(fontsize=25,rotation=45)

#1.3sex
#不同性别的生存率
plt.figure(figsize=(12,8))
train.groupby('sex')['survived'].mean().plot.bar(fontsize=25)
plt.xlabel('sex',fontsize=25)
plt.xticks(fontsize=25,rotation=0)
#性别编码
total['sex'] = pd.factorize(total['sex'])[0]

#1.4 sibsp兄弟姐妹数量对生存的影响
plt.figure(figsize=(12,8))
sns.set(font_scale=2)
sns.barplot('sibsp','survived',data=train,ci=0)

#1.5parch父母子女数量对生存的影响
plt.figure(figsize=(12,8))
sns.barplot('parch','survived',data=train,ci=0)

#1.6fare&ticket
#船票号码数量统计
len(total.ticket.unique())
pd.value_counts(total.ticket)
#重新计算票价
ticket_count=total.groupby('ticket')['pid'].transform('count')
total['fare']=total.fare/ticket_count
#3等仓s上船点的中位数票价填充缺失值
total[total.fare.isnull()]
median=total[total.pclass==3][total.embarked=='S'].fare.median()
total['fare']=total.fare.fillna(median)
#票价分布直方图
plt.figure(figsize=(20,10))
total.fare.plot.hist(bins=50,fontsize=25)
plt.xlabel('fare',fontsize=25)
plt.ylabel('count',fontsize=25)

#1.7cabin
#cabin分类
def get_cabin(x):
    if pd.isnull(x):
        return 'N'
    else:
        return 'FN'
total['cabin']=total.cabin.apply(get_cabin)
#不同cabin的生存率
train=total.loc[:890]
plt.figure(figsize=(12,8))
train.groupby('cabin')['survived'].mean().plot.bar(fontsize=25)
plt.xlabel('cabin',fontsize=25)
plt.xticks(rotation=0)

#1.8embarked
#缺失行填补
total[total.embarked.isnull()]
plt.figure(figsize=(16,12))
sns.boxplot('embarked','fare',hue='pclass',data=total)
total['embarked']=total.embarked.fillna('C')
#不同登船点的生存率
train=total.loc[:890]
plt.figure(figsize=(12,8))
sns.barplot('embarked','survived',data=train,ci=0)


#二、特征工程
#2.1称谓归类
title_Dict = {}
title_Dict.update(dict.fromkeys(['Mme','Dona','Ms','Lady','Mrs','Countess'],'Mrs'))
title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
title_Dict.update(dict.fromkeys(['Mr','Jonkheer'], 'Mr'))
title_Dict.update(dict.fromkeys(['Master','Col','Major','Dr','Don','Sir'],'Master'))
title_Dict.update(dict.fromkeys(['Rev','Capt'], 'Rev'))
total['title']=total['title'].map(title_Dict)
#查看归类后对生存的影响
train=total.loc[:890]
plt.figure(figsize=(12,8))
sns.barplot(x='title',y='survived',data=train,ci=0)

#2.2名字长度
total['name_len'] = total.name.apply(len)
pd.value_counts(total.name_len)
#名字长度分布
plt.figure(figsize=(12,8))
sns.distplot(total.name_len,bins=20,kde=False)
#不同长度下的生存率
train=total.loc[:890]
sns.set(font_scale=2.5)
plt.figure(figsize=(20,8))
sns.barplot('name_len','survived',data=train,ci=0)
#删除name特征
total=total.drop(['name'],axis=1)
#k-means选择分组数量,计算分组间隔点
select_clusters(total[['name_len']])
interval_points(total[['name_len']],4)
#分组
def name_len_modify(x):
    if x<23.2:
        return 0
    elif x<32.3:
        return 1
    elif x<43.5:
        return 2
    else:
        return 3
total['nlg']=total.name_len.apply(name_len_modify)
#查看分组后不同长度的生存率
train=total.loc[:890]
plt.figure(figsize=(12,8))
sns.barplot('nlg','survived',data=train,ci=0)

#2.3家庭人数
#计算家庭人数
total['family']=total['parch']+total['sibsp']+1
#数量分布
train=total.loc[:890]
plt.figure(figsize=(12,8))
sns.distplot(total.family,kde=False)
plt.ylabel('count')
#不同家庭人数的生存率
plt.figure(figsize=(12,8))
sns.barplot('family','survived',data=train,ci=0)
#分组
def family_modify(x):
    if x<=1:
        return 'single'
    elif x<=5:
        return 'small'
    else:
        return 'big'
total['family_group']=total.family.apply(family_modify)
#分组后的生存率
train=total.loc[:890]
plt.figure(figsize=(12,8))
sns.barplot('family_group','survived',data=train,ci=0)

#2.4fare
#对数变换
total['fare_log']=total.fare
index=total[total.fare!=0].index
total.loc[index,'fare_log']=np.log(total.loc[index,'fare_log'])
#变换后的票价分布直方图
plt.figure(figsize=(20,10))
total.fare_log.plot.hist(bins=50,fontsize=25)
plt.xlabel('fare_log',fontsize=25)
plt.ylabel('count',fontsize=25)
#k-means选择分组数量,计算分组间隔点
select_clusters(total[['fare_log']])
interval_points(total[['fare_log']],6)
#分组
def fare_modify(x):
    if x<1:
        return 0
    elif x<1.82:
        return 1
    elif x<2.27:
        return 2
    elif x<2.91:
        return 3
    elif x<3.58:
        return 4
    else:
        return 5
total['fare_group']=total.fare_log.apply(fare_modify)
total=total.drop(['fare_log'],axis=1)
#不同票价的生存率
train=total.loc[:890]
plt.figure(figsize=(12,8))
train.groupby('fare_group')['survived'].mean().plot.bar(fontsize=25)
plt.xlabel('fare',fontsize=25)
plt.xticks(rotation=0)

#2.4cabin,embarked,title,特征独热化
total=pd.concat([total,pd.get_dummies(total[['family_group','cabin','title','embarked']])],axis=1)
total=total.drop(['family_group','cabin','title','embarked','ticket','fare','name_len','family'],axis=1)

#2.5age
#回归插补缺失值
age_df=total.drop(['pid','survived'],axis=1)
age_df_notnull = age_df.loc[(age_df['age'].notnull())]
age_df_isnull = age_df.loc[(age_df['age'].isnull())]
#提取训练数据
x_age_train=age_df_notnull.drop(['age'],axis=1)
y_age_train=age_df_notnull.iloc[:,2]
#建模调参
rfr=RandomForestRegressor(n_jobs=-1,n_estimators=200,warm_start=True) 
param_grid={'max_depth':[4,5,6,7],
            'max_features':[5,6,7,8,9]}
rfr=gscvr(x_age_train,y_age_train,rfr,param_grid)
#预测并填补
rfr.fit(x_age_train,y_age_train)
age_predict=rfr.predict(age_df_isnull.drop(['age'],axis=1))
total.loc[total['age'].isnull(),['age']]=age_predict
#查看新的年龄分布
plt.figure(figsize=(12,8))
total.age.plot.hist(bins=30)
#k-means离散化
select_clusters(total[['age']])
interval_points(total[['age']],5)
def age_modify(x):
    if x<=13.4:
        return 0
    elif x<=26:
        return 1
    elif x<=37.2:
        return 2
    elif x<=51.4:
        return 3
    else :
        return 4
total['age_group']=total.age.apply(age_modify)
total=total.drop(['age'],axis=1)
#不同年龄的生存率
train=total.loc[:890]
plt.figure(figsize=(12,8))
train.groupby('age_group')['survived'].mean().plot.bar(fontsize=25)
plt.xlabel('age_group',fontsize=25)
plt.xticks(fontsize=25,rotation=0)
plt.figure(figsize=(12,8))

#2.6特征选择
x_train=total.iloc[0:891,2:]
y_train=total.iloc[0:891,1]
#相关度分析
cor=total.iloc[0:891,1:].corr()   
plt.figure(figsize=(30,30))
sns.heatmap(cor,linewidths=0.2,vmax=1.0,square=True,linecolor='white', annot=True)

#随机森林计算重要度
rf=RandomForestClassifier(n_estimators=300, n_jobs=-1,random_state=0)
param_grid={'max_depth':[3,4,5,6],
            'max_features':[3,4,5,6,7]}
rf=gscv(x_train,y_train,rf,param_grid)
feature_importances(rf,x_train,y_train)
#adaboost计算重要度
ada=AdaBoostClassifier()
param_grid={'n_estimators':[200,500,700],
            'learning_rate':[0.05,0.1,0.2]}
ada=gscv(x_train,y_train,ada,param_grid)
feature_importances(ada,x_train,y_train)
#xgboost计算重要度
xgb= XGBClassifier(min_child_weight=1,
                   subsample=0.8,colsample_bytree=0.8,
                   objective='binary:logistic')
param_grid={'learning_rate':[0.05,0.1,0.2,0.5,1],
            'max_depth':range(1,4),
            'gamma':np.linspace(0,0.4,3)}
xgb=gscv(x_train,y_train,xgb,param_grid)
feature_importances(xgb,x_train,y_train)
#选择前k项特征
clfs=[rf,ada,xgb]
features=feature_select(clfs,x_train,y_train,8)
x_train=x_train[features]


#三、模型评估与参数调优
#3.1逻辑回归
#调参建模
pipe_lr=Pipeline([('scl',StandardScaler()),
                  ('clf',LogisticRegression(n_jobs=-1))]) 
param_grid={'clf__C':np.logspace(-2,2,5),'clf__penalty':['l1','l2']}
pipe_lr=gscv(x_train,y_train,pipe_lr,param_grid)
#交叉验证精度&学习曲线
cross_val_score(pipe_lr,x_train,y_train,cv=5,scoring='accuracy').mean()
plot_learning_curve(x_train,y_train,pipe_lr,'lr learning_curve')
#混淆矩阵
plot_confusion_matrix(x_train, y_train,pipe_lr,'lr_confusion_matrix')
#找出假正例，假反例
FP_index,FN_index=find_FP_FN(x_train, y_train,pipe_lr)
train_FP=train.iloc[FP_index,:]
train_FN=train.iloc[FN_index,:]

#3.2k近邻
#参数调优
pipe_knn=Pipeline([('scl',StandardScaler()),
                  ('clf',KNeighborsClassifier(weights='distance',algorithm='auto',n_jobs=-1))]) 
param_grid={'clf__n_neighbors':[2,3,4,5,6,7,8,9,10]}
pipe_knn=gscv(x_train,y_train,pipe_knn,param_grid)
#交叉验证精度&学习曲线
cross_val_score(pipe_knn,x_train,y_train,cv=5,scoring='accuracy').mean()
plot_learning_curve(x_train,y_train,pipe_knn,'knn learning_curve')

#3.3决策树
dt=DecisionTreeClassifier() 
param_grid={'max_depth':np.linspace(2,5,4)}
dt=gscv(x_train,y_train,dt,param_grid)
#交叉验证精度&学习曲线
cross_val_score(dt,x_train,y_train,cv=5,scoring='accuracy').mean()
plot_learning_curve(x_train,y_train,dt,'dt learning_curve')

#3.4支持向量机
pipe_svm=Pipeline([('scl',StandardScaler()),
                  ('clf',SVC(probability=True))]) 
param_range=np.logspace(-2,2,5)
param_grid=[{'clf__C':param_range,'clf__kernel':['linear']},
             {'clf__C':param_range,'clf__kernel':['rbf'],'clf__gamma':param_range}]
pipe_svm=gscv(x_train,y_train,pipe_svm,param_grid)
#交叉验证精度&学习曲线
cross_val_score(pipe_svm,x_train,y_train,cv=5,scoring='accuracy').mean()
plot_learning_curve(x_train,y_train,pipe_svm,'svm learning_curve')

#3.5随机森林
rf=RandomForestClassifier(n_jobs=-1,n_estimators=300,warm_start=True)
param_grid={'max_depth':[3,4,5],
            'max_features':[4,5,6,7]}
rf=gscv(x_train,y_train,rf,param_grid)
#交叉验证精度&学习曲线
cross_val_score(rf,x_train,y_train,cv=5,scoring='accuracy').mean()
plot_learning_curve(x_train,y_train,rf,'rf learning_curve')

#3.6xgboost
xgb= XGBClassifier(min_child_weight=1,
                   subsample=0.8,colsample_bytree=0.8,
                   objective='binary:logistic')
param_grid={'learning_rate':np.logspace(-2,0,3),
            'max_depth':range(1,3),
            'gamma':np.linspace(0,0.4,3)}
xgb=gscv(x_train,y_train,xgb,param_grid)
#交叉验证精度&学习曲线
cross_val_score(xgb,x_train,y_train,cv=5,scoring='accuracy').mean()
plot_learning_curve(x_train,y_train,xgb,'xgb learning_curve')

#3.7adaboost
ada=AdaBoostClassifier()
param_grid={'n_estimators':[200,500,700],
            'learning_rate':[0.05,0.1,0.2]}
ada=gscv(x_train,y_train,ada,param_grid)
#交叉验证精度&学习曲线
cross_val_score(ada,x_train,y_train,cv=5,scoring='accuracy').mean()
plot_learning_curve(x_train,y_train,ada,'ada learning_curve')

#3.8对比各类模型精度&相关度
clfs={'lr':pipe_lr,'svm':pipe_svm,'knn':pipe_knn,
      'dt':dt,'rf':rf,'xgb':xgb,'ada':ada}
multimodel_accuracy(clfs)
pred_matrix_corr(x_train,y_train,clfs)

#四、最终预测
#提取测试集
x_test=total.iloc[891:,2:]
x_test=x_test[features]

#stacking
#选择基础模型
clfs=[pipe_lr,dt,rf,pipe_svm,pipe_knn,ada,xgb]
#获取stack_train,stack_test
stack_train,stack_test=get_stack_data(clfs, x_train, y_train, x_test)
#stacking第二步调优
stack_lr=Pipeline([('scl',StandardScaler()),
                  ('clf',LogisticRegression(n_jobs=-1))]) 
param_grid={'clf__C':np.logspace(-2,2,5),'clf__penalty':['l1','l2']}
stack_lr=gscv(stack_train,y_train,stack_lr,param_grid)
#预测
stack_lr.fit(stack_train,y_train)
stack_pred=stack_lr.predict(stack_test)
submission['Survived']=stack_pred
submission.to_csv('d:/data_project/titanic/pred.csv', index=False,sep=',')

#voting
voting_clf = VotingClassifier(
        estimators=[('lr',pipe_lr),('dt',dt),('svm',pipe_svm),('xgb',xgb),('rf',rf),('knn',pipe_knn)],
        voting='soft')
cross_val_score(voting_clf,x_train,y_train,cv=5,scoring='accuracy').mean()
voting_clf.fit(x_train,y_train)
vote_pred=voting_clf.predict(x_test)
submission['Survived']=vote_pred
submission.to_csv('d:/data_project/titanic/pred.csv', index=False,sep=',')


