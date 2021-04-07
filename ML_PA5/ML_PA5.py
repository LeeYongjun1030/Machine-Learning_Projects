from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

def normalized(x):## x is array
    normalized_x = (x - np.min(x)) / (np.max(x)-np.min(x))
    return normalized_x


def cross_validation(classifier,features, labels):
    cv_scores = []

    for i in range(10):
        scores = cross_val_score(classifier, features, labels, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())
    
    return cv_scores


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

features = x_train
labels = y_train

# 교차검증을 통한 decision tree 와 random forest 의 정확도 비교
dt_cv_scores = cross_validation(tree.DecisionTreeClassifier(), features, labels)
rf_cv_scores = cross_validation(RandomForestClassifier(n_estimators = 30), features, labels)

#랜덤포레스트 VS 의사결정트리 시각화
n_list = [1,2,3,4,5,6,7,8,9,10]
plt.title('accuracy')
plt.plot(n_list, dt_cv_scores, label = 'dt')
plt.plot(n_list,rf_cv_scores, label = 'rf')
plt.legend(loc='upper right')
plt.show()


#의사결정트리 정확도
print("validation")
print('decision tree accuracy : ',np.mean(dt_cv_scores))

#랜덤포레스트 정확도
print('random forest accuracy : ',np.mean(rf_cv_scores))



# decision tree model 작성 및 학습
decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(x_train,y_train)
dt_y_pred = decision_tree.predict(x_test)

# random forest 작성 및 학습
random_forest = RandomForestClassifier(n_estimators = 30)
random_forest.fit(x_train,y_train)
rf_y_pred = random_forest.predict(x_test)


# test accuracy 비교
print("")
print("test")
print("decision tree accuracy : " + str( accuracy_score(le.classes_[y_test], le.classes_[dt_y_pred])) )
print("random forest accuracy : " + str( accuracy_score(le.classes_[y_test], le.classes_[rf_y_pred])) )

