# In[]
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import numpy.linalg as lg

mndata = MNIST('./data/')
X_train, labels_train = map(np.array, mndata.load_training()) 
X_test, labels_test = map(np.array, mndata.load_testing()) 

# In[]
# X_train = X_train/255.0 # nTr x d    60000 x 784
# X_test = X_test/255.0   # nTs x d    10000 x 784    


def MSE(pred,true):
    errors = np.square(pred-true)
    return np.mean(np.mean(errors))

nTrain, d = X_train.shape # 60000 , 784
nTest = len(labels_test)  # 10000

mu = np.average(X_train, axis=0)
Mu = np.broadcast_to(mu,(nTrain,d))

X = X_train - Mu
# Cov = np.matmul(np.transpose(X), X)/nTrain
# print(np.trace(Cov))
# eigVal,eigVec = lg.eig(Cov)
# print("eigvec",eigVec[:30])
# Cov_re =  eigVec.dot(np.diag(eigVal)).dot(eigVec.T) #correct

U, D, V = lg.svd(Cov, full_matrices = True)

#first two eigenvalues
print(D[0])   
print(D[1])       

#sum of eigenvalues
print(np.sum(D))
'''
332719.12203544343
243279.8843381836

3428445.433070623

'''
# In[]
Score = np.dot(X,U)
X_hat = Score.dot(U.T)
print("MSE:",MSE(X_hat+Mu,X+Mu)) 

error_train = {}
for k in range(1,101):
    Uk = U[:,:k]
    Scorek = np.dot(X,Uk)
    X_hatk = Scorek.dot(Uk.T)
    error = MSE(X_hatk+Mu,X+Mu)
    print(f"MSE{k}: {error}")
    error_train[k] = error

# In[]
MuT = np.broadcast_to(mu,(nTest,d))
X_T = X_test - MuT

Score = np.dot(X_T,U)
X_hat = Score.dot(U.T)
print("MSE:",MSE(X_hat+MuT,X_T+MuT)) 
error_test = {}
for k in range(1,101):
    Uk = U[:,:k]
    Scorek = np.dot(X_T,Uk)
    X_hatk = Scorek.dot(Uk.T)
    error = MSE(X_hatk+MuT,X_T+MuT)
    print(f"MSE{k}: {error}")
    error_test[k] = error
# In[]
key, val = zip(*(error_train.items()))
keys, vals = zip(*(error_test.items()))

plt.xlabel("k")
plt.ylabel("error")
plt.title("Reconstruction MSE Error vs k")
plt.plot(key,val,label = 'Training error')
plt.plot(keys,vals,label = 'Testing error')
plt.legend()
plt.show()

# In[]
ratio = np.zeros(100)
all_sum = np.sum(D)
for k in range(1,101):
    ratio[k-1] = np.sum(D[:k]) / all_sum

plt.plot(np.arange(1,101),1-ratio)
plt.xlabel("k")
plt.ylabel("error")
plt.title("1 - Utilized Eigenvalues vs k")
plt.show()


# In[]
fig=plt.figure(figsize=(140,60))
fig.suptitle("Top 10 Eigenvectors Visualization", fontsize=16)
for i in range(10):
    vec = U[:,i]
    vecc = np.reshape(vec, (28,28))
    fig.add_subplot(2,5, i+1)
    plt.xlabel(f"eigenVec-{i+1}")
    plt.imshow(vecc)  

fig.show()  
# In[]

k_list = [5,15,40,100]
inds_2 = labels_train == 2
sample2 = X_train[inds_2][0]
inds_7 = labels_train == 7
sample7 = X_train[inds_7][0]
inds_6 = labels_train == 6
sample6 = X_train[inds_6][0]

print(sample2.shape)
vecc = np.reshape(sample2, (28,28))
plt.xlabel(f"Original")
plt.imshow(vecc)
plt.show()

fig=plt.figure()
i = 1
for k in k_list:
    Uk = U[:,:k]
    x_ = sample2 * (1.0)
    x_ -= mu
    Scorek = np.dot(x_,Uk)
    X_hatk = Scorek.dot(Uk.T)
    vecc = np.reshape(X_hatk, (28,28))
    fig.add_subplot(1,4, i)
    plt.xlabel(f"k={k}")
    plt.imshow(vecc)  
    i+=1
fig.show()
fig.clear()
vecc = np.reshape(sample6, (28,28))
plt.xlabel(f"Original")
plt.imshow(vecc)
plt.show()

fig=plt.figure()
# fig.suptitle("Reconstruction of '2' for different k")
i = 1
for k in k_list:
    Uk = U[:,:k]
    x_ = sample6 * (1.0)
    x_ -= mu
    Scorek = np.dot(x_,Uk)
    X_hatk = Scorek.dot(Uk.T)
    vecc = np.reshape(X_hatk, (28,28))
    fig.add_subplot(1,4, i)
    plt.xlabel(f"k={k}")
    plt.imshow(vecc)  
    i+=1

fig.show()
fig.clear()
vecc = np.reshape(sample7, (28,28))
plt.xlabel(f"Original")
plt.imshow(vecc)
plt.show()

fig=plt.figure()
# fig.suptitle("Reconstruction of '2' for different k")
i = 1
for k in k_list:
    Uk = U[:,:k]
    x_ = sample7 * (1.0)
    x_ -= mu
    Scorek = np.dot(x_,Uk)
    X_hatk = Scorek.dot(Uk.T)
    vecc = np.reshape(X_hatk, (28,28))
    fig.add_subplot(1,4, i)
    plt.xlabel(f"k={k}")
    plt.imshow(vecc)  
    i+=1
fig.show()

# %%
