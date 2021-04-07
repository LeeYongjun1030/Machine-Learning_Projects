import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt


train = pd.read_csv('ship_training_data.csv', encoding='utf-8')
valid = pd.read_csv('ship_validation_data.csv', encoding='utf-8')
test = pd.read_csv('ship_test_data.csv', encoding='utf-8')

x_train = train.loc[:,'speed':'engine_temperature']
x_train = x_train.drop(columns=['air_temperature','engine_pressure','engine_temperature'])
y_train = train.loc[:,'power']

x_valid = valid.loc[:,'speed':'engine_temperature']
x_valid = x_valid.drop(columns=['air_temperature','engine_pressure','engine_temperature'])
y_valid = valid.loc[:,'power']

x_test = test.loc[:,'speed':'engine_temperature']
x_test = x_test.drop(columns=['air_temperature','engine_pressure','engine_temperature'])
y_test = test.loc[:,'power']



model = RandomForestRegressor(max_depth = 20, random_state=10)

model.fit(x_train, y_train)


# Feature Importance
print(' ')
print(' [feature importance]')
for i in range(len(x_train.columns)):
    print(x_train.columns[i],'   ',model.feature_importances_[i])


# RMSE
def RMSE(model, x, y_truth): ## calculate RMSE
    y_pred = model.predict(x)
    d = y_pred - y_truth.values.ravel()
    rmse = np.sqrt(sum(pow(d,2))/len(x))
    comparison = pd.DataFrame({'prediction': y_pred, 'ground_truth':y_truth.values.ravel()}) 
    #print(comparison)
    return rmse

train_rmse = RMSE(model, x_train, y_train)
valid_rmse = RMSE(model, x_valid, y_valid)
test_rmse  = RMSE(model, x_test, y_test)

print(' ')
print(' [RMSE] ')
print('train RMSE : ',train_rmse)
print('valid RMSE : ',valid_rmse)
print('test  RMSE : ',test_rmse)


# predict power at calm sea
calm_sea_data = pd.read_csv('ship_calmsea.csv', encoding='utf-8')
calm_sea_data = calm_sea_data.loc[:,'speed':'engine_rpm']

y_calmsea = model.predict(calm_sea_data) # 입력데이터를 받아 power 추정

x_calmsea = calm_sea_data['speed'].values.ravel()

plt.plot(x_calmsea,y_calmsea) # speed - power 그래프 그리기


# approximation speed - power curve
# y = ax^b를 가정
# log(y) = log(a)+b*log(x)를 이용하여 일차함수(근사식)을 추정 => a,b를 구함
# a와 b를 이용하여 y = ax^b을 완성 => plot하여 비교


coefficients = np.polyfit(np.log(x_calmsea),np.log(y_calmsea),1) # 일차함수 근사 =>  a, b를 구함

def curve(x, coefficients):
    y = np.exp(coefficients[1])*np.power(x,coefficients[0]) # y = ax^b
    return y


curve_y = curve(x_calmsea, coefficients)
plt.plot(x_calmsea,curve_y) # speed - power 그래프 그리기
plt.title('y = {:.2f}x^{:.2f}'.format(np.exp(coefficients[1]),coefficients[0]))
plt.show()