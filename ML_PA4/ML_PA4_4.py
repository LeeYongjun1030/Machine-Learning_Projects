import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

def normalized(x):## x is array
    normalized_x = (x - np.min(x)) / (np.max(x)-np.min(x))
    return normalized_x

def data_processing(data): ## 데이터 전처리
    data['Sex'] = np.where(data['Sex']=='M',1.0,0.0)
    data['BP'] = np.where(data['BP']=='HIGH',1.0,np.where(data['BP']=='NORMAL',0.66,0.33))
    data['Cholesterol'] = np.where(data['Cholesterol']=='HIGH',1.0,np.where(data['Cholesterol']=='NORMAL',0.66,0.33))

    ##case of normalized (input data, output data)
    age = np.array(data.loc[:,['Age']])
    data['Age'] = normalized(age)
    na = np.array(data.loc[:,['Na_to_K']])
    data['Na_to_K'] = normalized(na)
    return data


df = pd.read_csv('drug.csv', encoding='utf-8')
df = data_processing(df)
train = df.loc[0:160,:] # 80% for train
test =  df.loc[160:200,:] # 20% for test

## input 항목으로 4가지 사용(age, bp, na_to_k, cholesterol)
x_train, y_train = train[['Age','BP', 'Na_to_K','Cholesterol']], train[['Drug']]
x_test, y_test = test[['Age','BP', 'Na_to_K','Cholesterol']], test[['Drug']]


# LabelEncoder로 레이블을 숫자로 변경
le = preprocessing.LabelEncoder()
y_encoded = le.fit_transform(y_train)


# build tree model
clf = tree.DecisionTreeClassifier(max_depth=4, min_samples_split=2,
                                  min_samples_leaf=2, random_state=70)

# train
clf.fit(x_train,y_encoded.ravel())

# test
y_pred = clf.predict(x_test)
# accuracy
print("accuracy : " + str( accuracy_score(y_test.values.ravel(), le.classes_[y_pred])) )

# comparison
comparison = pd.DataFrame({'prediction':le.classes_[y_pred], 'ground_truth':y_test.values.ravel()}) 
print(comparison)