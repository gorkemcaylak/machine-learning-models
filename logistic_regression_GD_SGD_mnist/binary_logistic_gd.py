# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import random

mndata = MNIST('./data/')
X_train, labels_train = map(np.array, mndata.load_training()) 
X_test, labels_test = map(np.array, mndata.load_testing()) 
idx_train = np.isin(labels_train, [2,7])
idx_test  = np.isin(labels_test,  [2,7])

X_train = X_train[idx_train]
X_test = X_test[idx_test]
labels_train = labels_train[idx_train]
labels_test = labels_test[idx_test]

X_train = X_train/255.0 # nTr x d    60000 x 784
X_test = X_test/255.0   # nTs x d    10000 x 784    


nTrain, d = X_train.shape # 12223 , 784
nTest = len(labels_test)  # 2060

#replace 2s with -1, 7s with 1
Y_train = np.array([int((i-4.5)/2.5) for i in labels_train])
Y_test  = np.array([int((i-4.5)/2.5) for i in labels_test])

print(Y_train.shape)

# In[2]:

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def J(x,y):
    n, d = x.shape
    k1 = np.matmul(x, w) + np.zeros(n)* b
    k2 = np.multiply(y, k1) * -1
    xp = np.exp(k2) + np.ones(n)
    J_ = np.mean(np.log(xp)) + lbda * np.sum(np.square(w))
    return J_

def dw():
    n, d = x.shape
    Dw = np.zeros(d)
    k = np.multiply(y, (Mu - np.ones(n)))
    Dw = np.matmul(x.T, k) / n + 2 * lbda * w
    return Dw

def db():
    n, d = x.shape
    k = (Mu - np.ones(n))
    Db = np.mean(np.multiply(k, y))
    return Db

def updateModel():
    global b,w
    updateMu()
    Dw = dw()
    Db = db()
    w = w - rate * Dw
    b = b - rate * Db 

def updateMu():
    n, d = x.shape
    global Mu
    Mu = np.zeros((n))
    B = b * np.ones(n)
    k1 = np.matmul(x, w) + B
    k2 = np.multiply(y, k1) 
    Mu = sigmoid(k2)

def getBatch(x_, y_, sample_size):
    n = x_.shape[0]
    ind = range(0,n,1)
    rd_ind = random.sample(ind, sample_size)
    return x_[rd_ind], y_[rd_ind]


# In[]:

# SGD (batch = 100)

batch_size = 100
Mu = np.zeros((nTrain))
lbda = 0.1 #* batch_size / nTrain
w = np.zeros((d))
b = 0
rate = 0.3

prev = 0
diff = 10
NTR = 100/nTrain
J_train = {}
mc_train = {}
NTS = 100/nTest
J_test = {}
mc_test = {}

i=0
inf_norm=1
x,y = getBatch(X_train, Y_train, batch_size)
while inf_norm > 1.5e-2:
    i += 1
    J_tr = J(X_train, Y_train)
    diff = prev - J_tr
    prev = J_tr
    J_ts = J(X_test, Y_test)
    J_train[i] = J_tr
    J_test[i] = J_ts
    
    yhat = np.matmul(X_train, w) + np.zeros(nTrain)* b
    res = np.multiply(yhat, Y_train) 
    corrects = np.count_nonzero(np.abs(res) + res) #counts 0s as wrong prediction!
    misc = (nTrain-corrects)/nTrain
    mc_train[i] = misc
    # print("TR : misc=", misc, "Accuracy", corrects*NTR, "loss", J_tr)

    yhat_ts = np.matmul(X_test, w) + np.zeros(nTest)* b
    res_ts = np.multiply(yhat_ts, Y_test)
    corrects_ts = np.count_nonzero(np.abs(res_ts) + res_ts)
    misc_ts = (nTest-corrects_ts)/nTest
    mc_test[i] = misc_ts
    # print("TS : misc=", misc, "Accuracy", corrects_ts*NTS, "loss", J_ts, "\n")

    updateModel()
    dW = dw()
    dB = db()
    dW = np.append(dW,dB)
    inf_norm =  np.max(np.abs(dW))
    # print(inf_norm, np.abs(diff))
    x,y = getBatch(X_train, Y_train, batch_size)

# In[]
S_jt = sorted(J_train.items())
S_js = sorted(J_test.items())

S_xx, S_yy = zip(*S_jt)
S_xs, S_ys = zip(*S_js)

plt.title('SGD100 - J vs Iteration for Train and Test')
plt.xlabel('Iteration')
plt.ylabel('J')
plt.plot(S_xx[:], S_yy[:], 'b-', label='Training Set')
plt.plot(S_xs[:], S_ys[:], 'g-', label='Testing Set')
plt.legend()
plt.savefig('SGD100_Jtrain_test_rate'+str(rate)+'_conv_reg.png')
plt.show()

# In[]
mt = sorted(mc_train.items())
ms = sorted(mc_test.items())

print(len(mt))
mx, my = zip(*mt)
msx, msy = zip(*ms)

plt.title('SGD100 - Misclassification vs Iteration for Train and Test')
plt.xlabel('Iteration')
plt.ylabel('Misclassification')
plt.plot(mx[1:], my[1:], 'b-', label='Training Set')
plt.plot(msx[1:], msy[1:], 'g-', label='Testing Set')
plt.legend()
plt.savefig('SGD100_Misclass_train_test_rate'+str(rate)+'_conv_reg.png')
plt.show()

# In[]:

# SGD (batch = 1)

Mu = np.zeros((nTrain))
w = np.zeros((d))
b = 0
rate = 0.001
lbda = 0.1  #  / nTrain 
prev = 10
diff = 10
J_train = {}
mc_train = {}
J_test = {}
mc_test = {}
NTR = 100/nTrain
NTS = 100/nTest

i=0
x,y = getBatch(X_train, Y_train, 1)
diff = 1
inf_norm = 1
while inf_norm > 2e-2:
    i += 1
    J_tr = J(X_train, Y_train)
    diff = prev - J_tr
    prev = J_tr
    J_ts = J(X_test, Y_test)
    J_train[i] = J_tr
    J_test[i] = J_ts
    
    yhat = np.matmul(X_train, w) + np.zeros(nTrain)* b
    res = np.multiply(yhat, Y_train) 
    corrects = np.count_nonzero(np.abs(res) + res) #counts 0s as wrong prediction!
    misc = (nTrain-corrects)/nTrain
    mc_train[i] = misc
    # print("TR : misc=", misc, "Accuracy", corrects*NTR, "loss", J_tr)

    yhat_ts = np.matmul(X_test, w) + np.zeros(nTest)* b
    res_ts = np.multiply(yhat_ts, Y_test)
    corrects_ts = np.count_nonzero(np.abs(res_ts) + res_ts)
    misc_ts = (nTest-corrects_ts)/nTest
    mc_test[i] = misc_ts
    # print("TS : misc=", misc, "Accuracy", corrects_ts*NTS, "loss", J_ts, "\n")

    updateModel()
    dW = dw()
    dB = db()
    dW = np.append(dW,dB)
    inf_norm =  np.max(np.abs(dW))
    # print(inf_norm, np.abs(diff))
    x,y = getBatch(X_train, Y_train, 1)

# In[]
s_jt = sorted(J_train.items())
s_js = sorted(J_test.items())

s_xx, s_yy = zip(*s_jt)
s_xs, s_ys = zip(*s_js)

plt.title('SGD1 - J vs Iteration for Train and Test')
plt.xlabel('Iteration')
plt.ylabel('J')
plt.plot(s_xx[:], s_yy[:], 'b-', label='Training Set')
plt.plot(s_xs[:], s_ys[:], 'g-', label='Testing Set')
plt.legend()
plt.savefig('SGD_Jtrain_test_rate'+str(rate)+'_conv_reg.png')
plt.show()

# In[]

mt = sorted(mc_train.items())
ms = sorted(mc_test.items())

print(len(mt))
mx, my = zip(*mt)
msx, msy = zip(*ms)

plt.title('SGD1 - Misclassification vs Iteration for Train and Test')
plt.xlabel('Iteration')
plt.ylabel('Misclassification')
# plt.xscale('log')
plt.plot(mx[1:], my[1:], 'b-', label='Training Set')
plt.plot(msx[1:], msy[1:], 'g-', label='Testing Set')
plt.legend()
#plt.annotate('%f',xx)
plt.savefig('SGD_Misclass_train_test_rate'+str(rate)+'_conv_reg.png')
plt.show()

# In[]:

## gradient descent :

Mu = np.zeros((nTrain))
lbda = 0.1
w = np.zeros((d))
b = 0
rate = 0.3

prev = 0
diff = 10
NTR = 100/nTrain
J_train = {}
mc_train = {}
NTS = 100/nTest
J_test = {}
mc_test = {}

x = X_train # n x d
y = Y_train # n x 1
i=0
inf_norm = 1
while inf_norm > 5e-3: 
    i += 1
    J_tr = J(X_train, Y_train)
    diff = prev - J_tr
    print(diff)
    prev = J_tr
    J_ts = J(X_test, Y_test)

    J_train[i] = J_tr
    J_test[i] = J_ts
    
    yhat = np.matmul(X_train, w) + np.zeros(nTrain)* b
    res = np.multiply(yhat, Y_train) 
    corrects = np.count_nonzero(np.abs(res) + res) #counts 0s as wrong prediction!
    misc = (nTrain-corrects)/nTrain
    mc_train[i] = misc
    # print("TR : misc=", misc, "Accuracy", corrects*NTR, "loss", J_tr)

    yhat_ts = np.matmul(X_test, w) + np.zeros(nTest)* b
    res_ts = np.multiply(yhat_ts, Y_test)
    corrects_ts = np.count_nonzero(np.abs(res_ts) + res_ts)
    misc_ts = (nTest-corrects_ts)/nTest
    mc_test[i] = misc_ts
    # print("TS : misc=", misc, "Accuracy", corrects_ts*NTS, "loss", J_ts, "\n")

    updateModel()
    dW = dw()
    dB = db()
    dW = np.append(dW,dB)
    inf_norm =  np.max(np.abs(dW))
    # print(inf_norm, np.abs(diff))

# In[3]:

jt = sorted(J_train.items())
js = sorted(J_test.items())

xx, yy = zip(*jt)
xs, ys = zip(*js)

plt.title('J vs Iteration for Train and Test')
plt.xlabel('Iteration')
plt.ylabel('J')
# plt.xscale('log')
plt.plot(xx[1:], yy[1:], 'b-', label='Training Set')
plt.plot(xs[1:], ys[1:], 'g-', label='Testing Set')
plt.legend()
#plt.annotate('%f',xx)
plt.savefig('Jtrain_test_rate'+str(rate)+'_conv.png')
plt.show()

# In[4]:

mt = sorted(mc_train.items())
ms = sorted(mc_test.items())

print(len(mt))
mx, my = zip(*mt)
msx, msy = zip(*ms)

plt.title('Misclassification vs Iteration for Train and Test')
plt.xlabel('Iteration')
plt.ylabel('Misclassification')
# plt.xscale('log')
plt.plot(mx[1:], my[1:], 'b-', label='Training Set')
plt.plot(msx[1:], msy[1:], 'g-', label='Testing Set')
plt.legend()
#plt.annotate('%f',xx)
plt.savefig('GD_Misclass_train_test_rate'+str(rate)+'_conv.png')
plt.show()


# %%
