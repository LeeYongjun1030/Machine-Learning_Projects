import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


def normalized(x):## x is array
    normalized_x = (x - np.min(x)) / (np.max(x)-np.min(x))
    return normalized_x

df = pd.read_csv('drug.csv', encoding='utf-8')

le = preprocessing.LabelEncoder()

# 데이터 전처리
df['Age']=normalized(df['Age'])
df['Sex']= le.fit_transform(df['Sex']) / (len(le.classes_)-1)
df['BP']= le.fit_transform(df['BP']) / (len(le.classes_)-1)
df['Cholesterol']= le.fit_transform(df['Cholesterol']) / (len(le.classes_)-1)
df['Na_to_K']=normalized(df['Na_to_K'])
df['Drug']= le.fit_transform(df['Drug'])

# 데이터 쪼개기
train = df.loc[0:160,:] # 80% for train
test =  df.loc[160:200,:] # 20% for test

x_train = train[['Age','BP', 'Na_to_K','Cholesterol']]
y_train = train[['Drug']].values.ravel()

x_test = test[['Age','BP', 'Na_to_K','Cholesterol']]
y_test = test[['Drug']].values.ravel()



# # 단일 모델 정확도 측정

# decision tree
dtree = tree.DecisionTreeClassifier(max_depth=4,random_state = 0)
dtree.fit(x_train,y_train)
dtree_pred = dtree.predict(x_test)

# random forest
random_forest = RandomForestClassifier(n_estimators = 10)
random_forest.fit(x_train,y_train)
rf_pred = random_forest.predict(x_test)


# knn
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)

# svm
svm = SVC(C=10, gamma=1, probability=True,random_state=0)
svm.fit(x_train, y_train)
svm_pred = svm.predict(x_test)


print(" [accuarcy]")
print("tree     : ",accuracy_score(y_test, dtree_pred))
print("random forest : ",accuracy_score(y_test, rf_pred))
print("knn      : ",accuracy_score(y_test, knn_pred))
print("svm      : ",accuracy_score(y_test, svm_pred))


# 하드 보팅
voting_clf = VotingClassifier(estimators=[
    ('dtree', dtree), ('knn', knn), ('svm', svm)], 
    weights=[1,0.9,0.95], voting='hard').fit(x_train, y_train)
hard_voting_predicted = voting_clf.predict(x_test)
print(" ")
print(" [ensemble]")
print( " hard voting accuracy : ",accuracy_score(y_test, hard_voting_predicted))


# 소프트 보팅
voting_clf = VotingClassifier(estimators=[
    ('dtree', dtree), ('knn', knn), ('svm', svm)], 
    weights=[1,0.9,0.95], voting='soft').fit(x_train, y_train)
soft_voting_predicted = voting_clf.predict(x_test)
print( " soft voting accuracy : ",accuracy_score(y_test, soft_voting_predicted))



# 정확도 비교 시각화
x = np.arange(6)
plt.bar(x, height= [accuracy_score(y_test, dtree_pred),
                    accuracy_score(y_test, rf_pred),
                    accuracy_score(y_test, knn_pred),
                    accuracy_score(y_test, svm_pred),
                    accuracy_score(y_test, hard_voting_predicted),
                    accuracy_score(y_test, soft_voting_predicted)])
plt.xticks(x, ['dtree','rforest','knn','svm','hard voting','soft voting']);
plt.title('accuracy')
plt.show()

