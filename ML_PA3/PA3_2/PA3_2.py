import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import csv

def sigmoid(x):
    return 1/(1+np.exp(-x))

def identity_function(x):
    return x

def softmax(x):
    return np.exp(x) / (np.sum(np.exp(x)))

def normalized(x):## x is array
    normalized_x = (x - np.min(x)) / (np.max(x)-np.min(x))
    return normalized_x

def setting_data(data): ## for normalizing data
    gc = np.array(data.loc[:,['gc']])
    data['gc'] = normalized(gc)
    bp = np.array(data.loc[:,['bp']])
    data['bp'] = normalized(bp)
    st = np.array(data.loc[:,['st']])
    data['st'] = normalized(st)
    si = np.array(data.loc[:,['si']])
    data['si'] = normalized(si)
    bmi = np.array(data.loc[:,['bmi']])
    data['bmi'] = normalized(bmi)
    dpi = np.array(data.loc[:,['dpi']])
    data['dpi'] = normalized(dpi)
    age = np.array(data.loc[:,['age']])
    data['age'] = normalized(age)
    return data
#########################################################################
class hiddenlayer1 : ##hidden layer 개수 1
    def init_network(): ##hidden layer의 node 수 9
        network = {}
        network['w1'] = np.array([[0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.2,0.1],
                                 [0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.2,0.1],
                                 [0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.2,0.1],
                                 [0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.2,0.1],
                                 [0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.2,0.1],
                                 [0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.2,0.1],
                                 [0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.2,0.1],
                                 ])
        network['b1'] = np.array([[0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.2,0.1]])
        network['w2'] = np.array([[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2]])
        network['b2'] = np.array([[0.1,0.2]])
        return network

    def training(network, x, t):
        ##forward
        w1, w2 = network['w1'], network['w2']
        b1, b2 = network['b1'], network['b2']
        a1 = np.dot(x,w1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,w2)+b2
        y = sigmoid(a2)
        ##backward
        delta2 = y - t
        delta1 = np.dot(delta2,w2.T)*z1*(1-z1)
        ##update
        lr = 0.1 ##learning rate
        b2 -= lr*delta2
        w2 -= lr*np.dot(z1.T,delta2)
        b1 -= lr*delta1
        w1 -= lr*np.dot(x.T,delta1)
        temp={}
        temp['w1'], temp['w2'] = w1, w2
        temp['b1'],temp['b2'] = b1, b2
        return temp

    def test(network, x, t):
        w1, w2 = network['w1'], network['w2']
        b1, b2 = network['b1'], network['b2']
        a1 = np.dot(x,w1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,w2)+b2
        y = sigmoid(a2) ## y를 0과 1 사이의 갑으로 추출.
        if y[0][0] >= y[0][1] : ## y가 0.5 이상이 되면 predict를 1로 선언. 0.5미만의 경우 predict를 0으로 선언.
            predict = 1
        else :
            predict = 0

        if predict == t[0] : ## predict와 target이 같으면 예측 성공, 다르면 예측 실패로 간주.
            error = 0
        else:
            error = 1
        return error
###########################################################################
class hiddenlayer2 : ##hidden layer 개수 2
    def init_network(): ## 첫번째 hidden layer의 node 수 5, 두번째 hidden layer의 node 수 5
        network = {}
        network['w1'] = np.array([[0.1,0.1,0.1,0.1,0.1],
                                 [0.1,0.1,0.1,0.1,0.1],
                                 [0.1,0.1,0.1,0.1,0.1],
                                 [0.1,0.1,0.1,0.1,0.2],
                                 [0.1,0.1,0.1,0.1,0.2],
                                 [0.1,0.1,0.1,0.1,0.1],
                                 [0.1,0.1,0.1,0.1,0.2],
                                 ])
        network['b1'] = np.array([[0.1,0.2,0.1,0.1,0.2]])
        network['w2'] = np.array([[0.1,0.2,0.1,0.1,0.2],[0.1,0.2,0.1,0.1,0.2],[0.1,0.2,0.1,0.1,0.2],[0.1,0.2,0.1,0.1,0.2],[0.1,0.2,0.1,0.1,0.2]])
        network['b2'] = np.array([[0.1,0.2,0.1,0.1,0.2]])
        network['w3'] = np.array([[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2]])
        network['b3'] = np.array([[0.1,0.2]])
        return network

    def training(network, x, t):
        ##forward
        w1, w2, w3 =network['w1'], network['w2'], network['w3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']
        a1 = np.dot(x,w1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,w2)+b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2,w3)+b3
        y = sigmoid(a3) ## output layer의 활성홤수는 sigmoid
       
        ##backward
        delta3 = y - t ## error cost = CECF
        delta2 = np.dot(delta3,w3.T)*z2*(1-z2)
        delta1 = np.dot(delta2,w2.T)*z1*(1-z1)
        ##update
        lr = 0.1 ##learning rate
        b3 -= lr*delta3
        w3 -= lr*np.dot(z2.T,delta3)
        b2 -= lr*delta2
        w2 -= lr*np.dot(z1.T,delta2)
        b1 -= lr*delta1
        w1 -= lr*np.dot(x.T,delta1)
        temp={}
        temp['w1'], temp['w2'], temp['w3'] = w1, w2, w3
        temp['b1'],temp['b2'],temp['b3'] = b1, b2, b3
        return temp

    def test(network, x, t):
        w1, w2, w3 = network['w1'], network['w2'], network['w3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']
        a1 = np.dot(x,w1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,w2)+b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2,w3)+b3
        y = sigmoid(a3)
        
        if y[0][0] >= y[0][1] : 
            predict = 1
        else :
            predict = 0

        if predict == t[0] : ## predict와 target이 같으면 예측 성공, 다르면 예측 실패로 간주.
            error = 0
        else:
            error = 1
        return error
#############################################################################

#input
training_data = 612### 8 : 1 : 1 ##765
val_data = 76
test_data = 76
epoch = 50

network = hiddenlayer1.init_network() ##hidden layer 1개인 경우
##network = hiddenlayer2.init_network() ##hidden layer 2개인 경우

df = pd.read_csv('diabetes.csv', encoding='utf-8')
sdf = setting_data(df) ## for normalizing data

######################### training #######################################
for k in range(epoch):
    for i in range(training_data+val_data):
        x = np.array(sdf.loc[[i],'gc':'age'])
        target = np.array(sdf.loc[[i],'target'])
        if np.sum(target) == 1 :
            t_vector = np.array([1.0,0.0])
        else:
            t_vector = np.array([0.0,1.0])
        network = hiddenlayer1.training(network,x,t_vector) ##hidden layer 1개인 경우
        ##network = hiddenlayer2.training(network,x,t_vector)  ##hidden layer 2개인 경우


####################### test ###############################################
total_error = 0    
for i in range(training_data+val_data,training_data+val_data+test_data): ##test
    x = np.array(sdf.loc[[i],'gc':'age'])
    target = np.array(sdf.loc[[i],'target'])
    if np.sum(target) == 1 :
        t_vector = np.array([1.0,0.0])
    else:
        t_vector = np.array([0.0,1.0])
    error = hiddenlayer1.test(network,x,t_vector) ##hidden layer 1개인 경우
    ##error = hiddenlayer2.test(network,x,t_vector)  ##hidden layer 2개인 경우
    total_error += error

error_rate = float(total_error / test_data*100)
print('error = ',error_rate,'%')

