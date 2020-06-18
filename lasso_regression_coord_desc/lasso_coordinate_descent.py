# In[0]:

import numpy as np
import matplotlib.pyplot as plt

'''
create synthetic data
'''
n = 500
K = 100
sigmaSq = 1
d = 1000
w_synth = np.zeros((d))
w_synth[:K] = [j/K for j in range(1,K+1)]

xV = np.random.normal(0, 1, n*d)
x = np.reshape(xV, (n,d))
e = np.random.normal(0, sigmaSq, n)
y = x.dot(w_synth) + e

#find maximum lambda to start decreasing
y_ = y - np.average(y) # 500 x 1
s1 = np.zeros((d))
s1 = x.T@y_
l = 2*np.abs(s1)
lbda = np.amax(l) 
#print(lbda)

# In[1]:
a = 2 * np.sum(np.square(x) , axis = 0) 
lbda_nonzero = {} 
fdr_tpr = []
count = 0
orig_nonzeros = np.nonzero(w_synth)
while count<999: #make 1000
    max_update = 1
    w_prev = np.zeros((d))
    w = np.zeros((d)) 
    while max_update > 0.01:
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
    w_nonzeros = np.nonzero(w)
    true_nonzeros_i = np.intersect1d(w_nonzeros, orig_nonzeros)
    true_nonzeros = len(true_nonzeros_i)
    count = np.count_nonzero(w)
    false_nonzeros = count - true_nonzeros
    fdr = 0 if count == 0 else false_nonzeros / count
    tpr = true_nonzeros / K
    print("fdr: ", fdr, " tpr: ", tpr)
    fdr_tpr.append((fdr,tpr))
    # print(lbda, " -> ", count)
    lbda_nonzero[lbda] = count
    lbda /= 1.5
# print(fdr_tpr)
# print(lbda_nonzero)

# In[2]:

kk = sorted(lbda_nonzero.items())
xx, yy = zip(*kk)
plt.title('Nonzero Elements for Decreasing Lambda')
plt.xlabel('Lambda')
plt.ylabel('#non-zero')
plt.xscale('log')
plt.plot(xx, yy, 'b-')
plt.savefig('lasso_nonzero.png')
plt.show()

plt.title('TPR / FDR Graph for Different Lambdas')
plt.xlabel('FDR')
plt.ylabel('TPR')
plt.scatter(*zip(*fdr_tpr))
plt.savefig('fdr_tpr.png')
plt.show()


# %%
