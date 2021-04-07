import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def normalized(x):## x is array
    normalized_x = (x - np.min(x)) / (np.max(x)-np.min(x))
    return normalized_x

def preprocessing(data): ## 데이터 전처리
    data['Sex'] = np.where(data['Sex']=='M',1.0,0.0)
    data['BP'] = np.where(data['BP']=='HIGH',1.0,np.where(data['BP']=='NORMAL',0.66,0.33))
    data['Cholesterol'] = np.where(data['Cholesterol']=='HIGH',1.0,np.where(data['Cholesterol']=='NORMAL',0.66,0.33))

    ##case of normalized (input data, output data)
    age = np.array(data.loc[:,['Age']])
    data['Age'] = normalized(age)
    na = np.array(data.loc[:,['Na_to_K']])
    data['Na_to_K'] = normalized(na)
    return data

def svc_param_selection(X, y, nfolds):
    svm_parameters = [
                        {'kernel': ['rbf'],
                         'gamma': [0.00001,0.0001, 0.001, 0.01, 0.1, 1],
                         'C': [0.01, 0.1, 1, 10, 100, 1000]
                        }
                       ]
    
    clf = GridSearchCV(SVC(), svm_parameters, cv=10)
    clf.fit(x_train, y_train.values.ravel())
    print(clf.best_params_)
    
    return clf

df = pd.read_csv('drug.csv', encoding='utf-8')
df = preprocessing(df)
train = df.loc[0:160,:] # 80% for train
test = df.loc[160:200,:] # 20% for test

## input 항목으로 3가지 사용(age, bp, na_to_k)
x_train, y_train = train[['BP', 'Na_to_K','Age','Cholesterol']], train[['Drug']]
x_test, y_test = test[['BP','Na_to_K','Age','Cholesterol']], test[['Drug']]

# 최적의 파라미터를 sklearn의 gridsearch를 통해 구함.
clf = svc_param_selection(x_train, y_train.values.ravel(), 10)

# SVM model로 예측값을 구함.
y_pred = clf.predict(x_test)

# 에측값과 실제값을 비교
print(classification_report(y_test, y_pred))
comparison = pd.DataFrame({'prediction':y_pred, 'ground_truth':y_test.values.ravel()}) 
print(comparison)

# accuracy
print("accuracy : "+ str(accuracy_score(y_test, y_pred)) )




