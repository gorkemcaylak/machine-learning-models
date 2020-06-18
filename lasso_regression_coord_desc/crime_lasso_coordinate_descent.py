# In[]
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df_train = pd.read_table("crime-train.txt")
df_test = pd.read_table("crime-test.txt")

nTrain, d = df_train.shape  #1595 x 96
nTest = df_test.shape[0]   #399

d = d-1 
# print(nTrain, nTest, d)
# print(df_train.index)
# print(df_train.columns)
df_train.head()

y_train = df_train['ViolentCrimesPerPop'].values
y_test = df_test['ViolentCrimesPerPop'].values

x_train = df_train.iloc[:,1:].values
x_test = df_test.iloc[:,1:].values
print(x_train.shape) #1595 x 95

# initial lambda to decrease
s1 = x_train.T@y_train  #95
l = 2*np.abs(s1)
lbda = np.amax(l)
# print(lbda)

# In[]
x = x_train
y = y_train
n = nTrain
lbda_nonzero = {} 
saved_w = []

count = 0
train_error = {}
test_error = {}
w = np.zeros((d))
lbda = 30
w30 = np.zeros((d))
a = 2 * np.sum(np.square(x) , axis = 0)
while lbda >= 0.01:
    max_update = 1
    w_prev = np.zeros((d)) 
    while max_update > 0.001:
        b = np.average(y - np.matmul(x,w))
        c = np.zeros((d))
        w_prev = w.copy()
        for k in range(d):
            for i in range(n):
                t = b + np.sum(w[:]*x[i,:]) - w[k]*x[i,k]
                c[k] += 2 * (x[i,k] * (y[i] - t))
            if (c[k] < -lbda):
                w[k] = (c[k]+lbda)/a[k]
            elif (c[k] > lbda):
                w[k] = (c[k]-lbda)/a[k] 
            else:
                w[k] = 0
        max_update = np.amax(np.abs(w - w_prev))
        obj = np.sum(np.square(x.dot(w) + b - y) + lbda * np.sum(np.abs(w)))
        # print(obj)
    w30 = w.copy()
    count = np.count_nonzero(w)
    lbda_nonzero[lbda] = count
    lbda /= 2
    pred_tr = np.matmul(x, w)
    pred_ts = np.matmul(x_test, w)
    error_tr = np.average(np.square(y_train-pred_tr)) 
    error_ts = np.average(np.square(y_test-pred_ts)) 
    test_error[lbda] = error_ts
    train_error[lbda] = error_tr
    # print("tr error=", error_tr)
    # print("ts error=", error_ts)
# print(lbda_nonzero)

# In[]

tr = sorted(train_error.items())
ld, e_tr = zip(*tr)
ts = sorted(test_error.items())
ld2, e_ts = zip(*ts)
plt.title('Training & Testing Error vs Lambda')
plt.xlabel('Lambda')
plt.ylabel('Error')
plt.xscale('log')
plt.plot(ld, e_tr, 'b-', label= 'Training Error')
plt.plot(ld, e_ts, 'g-', label= 'Testing Error')
plt.legend()
plt.savefig('Crime_errors.png')
plt.show()

kk = sorted(lbda_nonzero.items())
xx, yy = zip(*kk)
plt.title('Crimes - Nonzero Elements vs Lambda')
plt.xlabel('Lambda')
plt.ylabel('#non-zero')
plt.xscale('log')
plt.plot(xx, yy, 'b-')
plt.savefig('Crime_lasso_nonzero.png')
plt.show()


# %%
