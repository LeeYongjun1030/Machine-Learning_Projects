import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import csv

def sigmoid(x):
    return 1/(1+np.exp(-x))

def identity_function(x):
    return x

def normalized(x):## x is array
    normalized_x = (x - np.min(x)) / (np.max(x)-np.min(x))
    return normalized_x

def setting_data(data):
    data['sex'] = np.where(data['sex']=='male',1.0,0.0)
    data['smoker'] = np.where(data['smoker']=='yes',1.0,0.0)
    data['region'] = np.where(data['region']=='southeast',0.0,np.where(data['region']=='southwest',0.33,np.where(data['region']=='northeast',0.66,1.0)))
    
    ##case of normalized (input data, output data)
    age = np.array(data.loc[:,['age']])
    data['age'] = normalized(age)
    bmi = np.array(data.loc[:,['bmi']])
    data['bmi'] = normalized(bmi)
    child = np.array(data.loc[:,['children']])
    data['children'] = normalized(child)
    charge = np.array(data.loc[:,['charges']])
    data['charges'] = normalized(charge)
    
    return data
#########################################################################
class hiddenlayer1 : ##hidden layer ๊ฐ์ 1
    def init_network(): ##hidden layer์ node ์ 9
        network = {}
        network['w1'] = np.array([[0.1,0.2,0.3,0.1,0.2,0.3,0.1,0.2,0.3],
                                 [0.1,0.2,0.3,0.1,0.2,0.3,0.1,0.2,0.3],
                                 [0.1,0.2,0.3,0.1,0.2,0.3,0.1,0.2,0.3],
                                 [0.1,0.2,0.3,0.1,0.2,0.3,0.1,0.2,0.3],
                                 [0.1,0.2,0.3,0.1,0.2,0.3,0.1,0.2,0.3],
                                 [0.1,0.2,0.3,0.1,0.2,0.3,0.1,0.2,0.3],
                                 ])
        network['b1'] = np.array([[0.1,0.2,0.3,0.1,0.2,0.3,0.1,0.2,0.3]])
        network['w2'] = np.array([[0.1],[0.3],[0.5],[0.1],[0.3],[0.5],[0.1],[0.3],[0.5]])
        network['b2'] = np.array([[0.1]])
        return network

    def training(network, x, t):
        ##forward
        w1, w2 = network['w1'], network['w2']
        b1, b2 = network['b1'], network['b2']
        a1 = np.dot(x,w1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,w2)+b2
        y = identity_function(a2)
       
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
        y = identity_function(a2)
        error = 0.5*( y - t)* (y - t)
        return error
###########################################################################
class hiddenlayer2 : ##hidden layer ๊ฐ์ 2
    def init_network(): ## ์ฒซ๋ฒ์งธ hidden layer์ node ์ 5, ๋๋ฒ์งธ hidden layer์ node ์ 5
        network = {}
        network['w1'] = np.array([[0.1,0.2,0.1,0.1,0.2],
                                 [0.1,0.2,0.1,0.1,0.2],
                                 [0.1,0.2,0.1,0.1,0.2],
                                 [0.1,0.2,0.1,0.1,0.2],
                                 [0.1,0.2,0.1,0.1,0.2],
                                 [0.1,0.2,0.1,0.1,0.2],
                                 ])
        network['b1'] = np.array([[0.0,0.2,0.0,0.1,0.0]])
        network['w2'] = np.array([[0.1,0.2,0.3,0.1,0.2],[0.1,0.2,0.3,0.1,0.2],[0.1,0.2,0.3,0.1,0.2],[0.1,0.2,0.3,0.1,0.2],[0.1,0.2,0.3,0.1,0.2]])
        network['b2'] = np.array([[0.0,0.0,0.0,0.1,0.0]])
        network['w3'] = np.array([[0.1],[0.1],[0.1],[0.1],[0.1]])
        network['b3'] = np.array([[0.0]])
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
        y = identity_function(a3)
        
        error = 0.5*( y - t)* (y - t)
        ##backward
        delta3 = y - t
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
        return temp, error

    def test(network, x, t):
        w1, w2, w3 = network['w1'], network['w2'], network['w3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']
        a1 = np.dot(x,w1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,w2)+b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2,w3)+b3
        y = identity_function(a3)
        error = 0.5*( y - t)* (y - t)
        return error
#############################################################################

#input
training_data = 1070### 8 : 1 : 1
val_data = 133
test_data = 133
epoch = 50

##network = hiddenlayer1.init_network() ##hidden layer 1๊ฐ์ธ ๊ฒฝ์ฐ
network = hiddenlayer2.init_network() ##hidden layer 2๊ฐ์ธ ๊ฒฝ์ฐ

df = pd.read_csv('insurance.csv', encoding='utf-8')

sdf = setting_data(df) ##normalized data

######################### training #######################################
num_train = np.arange(0,(training_data+val_data)*epoch,1)
train_error = np.array([])
t=0
for k in range(epoch):
    for i in range(training_data+val_data):
        x = np.array(sdf.loc[[i],'age':'region'])
        target = np.array(sdf.loc[[i],'charges'])
        ##network = hiddenlayer1.training(network,x,target) ##hidden layer 1๊ฐ์ธ ๊ฒฝ์ฐ
        network,error = hiddenlayer2.training(network,x,target)  ##hidden layer 2๊ฐ์ธ ๊ฒฝ์ฐ
        train_error = np.append(train_error, error)
        t +=1
plt.title('error function(train)')
plt.plot(num_train,train_error)
plt.show()

####################### test ###############################################
num_test = np.arange(0,test_data,1)
test_error = np.array([])
t=0

for i in range(training_data+val_data,training_data+val_data+test_data): ##test
    x = np.array(sdf.loc[[i],'age':'region'])
    target = np.array(sdf.loc[[i],'charges'])
    error = hiddenlayer2.test(network,x,target)
    test_error = np.append(test_error,error)
    t+1
    
plt.title('error function(test)')
plt.plot(num_test,test_error)
plt.show()


