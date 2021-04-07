import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


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


df = pd.read_csv('drug.csv', encoding='utf-8')
df = preprocessing(df)
train = df.loc[0:160,:] ## 80% for train
test =  df.loc[160:199,:] ## 20% for test

## input 항목으로 3가지 사용(age, bp, na_to_k)
x_train, y_train = train[['Age', 'BP', 'Na_to_K']], train[['Drug']]
x_test, y_test = test[['Age', 'BP', 'Na_to_K']], test[['Drug']]


# 최적 k 찾기
max_k_range = train.shape[0] // 2
k_list = []
for i in range(3, max_k_range, 2):
    k_list.append(i)


cross_validation_scores = []
# 10-fold cross validation
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train.values.ravel(),
                             cv=10, scoring='accuracy')
    cross_validation_scores.append(scores.mean())

# visualize accuracy according to k
plt.plot(k_list, cross_validation_scores)
plt.xlabel('the number of k')
plt.ylabel('Accuracy')
plt.show()

# find best k
cvs = cross_validation_scores
k = k_list[cvs.index(max(cross_validation_scores))]
print("The best number of k : " + str(k) )

# knn model 만들기
knn = KNeighborsClassifier(n_neighbors=k)

# train
knn.fit(x_train, y_train.values.ravel())

# test
y_pred = knn.predict(x_test)

# check ground_truth with knn prediction
comparison = pd.DataFrame({'prediction':y_pred, 'ground_truth':y_test.values.ravel()}) 
print(comparison)

# check accuracy
print("accuracy : " + str( accuracy_score(y_test.values.ravel(), y_pred)) )

