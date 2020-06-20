# In[]
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import torch
import torch.optim as optim
import torch.nn.functional as func
mndata = MNIST('./data/')
X_train, labels_train = map(np.array, mndata.load_training()) 
X_test, labels_test = map(np.array, mndata.load_testing()) 

X_train = X_train/255.0 # nTr x d    60000 x 784
X_test = X_test/255.0   # nTs x d    10000 x 784    

nTrain, d = X_train.shape # 60000 , 784
nTest = len(labels_test)  # 10000

k = 10

#In[]

def linearLayer(x, W, b):
    # Wx + b
    A = torch.matmul(x, torch.transpose(W,0,1)) 
    n_ = b.size()[0]
    B = b.view(1,n_)
    return torch.add(A,B)

def forwardA(x, W0, b0, W1, b1):
    x1 = linearLayer(x, W0, b0)
    x2 = func.relu(x1)
    x3 = linearLayer(x2, W1, b1)
    return x3

def forwardB(x, W0, b0, W1, b1, W2, b2):
    x1 = linearLayer(x, W0, b0)
    x2 = func.relu(x1)
    x3 = linearLayer(x2, W1, b1)
    x4 = func.relu(x3)
    x5 = linearLayer(x4, W2, b2)
    return x5

def a(m):
    return 1/np.sqrt(m)

Y_train = np.zeros((nTrain, k))
Y_test = np.zeros((nTest, k))

#one-hot coding
for i in range(nTrain):
    hot = labels_train[i]
    Y_train[i][hot] = 1
for i in range(nTest):
    hot = labels_test[i]
    Y_test[i][hot] = 1

x_train = torch.from_numpy(X_train).float()
x_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(Y_train).float()
y_test = torch.from_numpy(Y_test).float()

labels_tr = torch.from_numpy(labels_train).long()
labels_ts = torch.from_numpy(labels_test).long()

d = 784
#In[]
W0_a = torch.empty(64, d).uniform_(-a(d), a(d)) 
W1_a = torch.empty(10, 64).uniform_(-a(64), a(64)) 

b0_a = torch.empty(64).uniform_(-a(d), a(d)) 
b1_a = torch.empty(10).uniform_(-a(64), a(64)) 

W0_a.requires_grad_(True)
W1_a.requires_grad_(True)

b0_a.requires_grad_(True)
b1_a.requires_grad_(True)

#   PART    A

rate = 0.001
optimizerA = optim.Adam([W0_a,b0_a,W1_a,b1_a], lr=rate)
batch_size = 500
mixed_ind = torch.randperm(nTrain)

iter_per_epoch = int(nTrain/batch_size) #60

lossF = func.cross_entropy
i=0
acc = 0.00
loss_list = []
while acc<0.99:
    i += 1
    for iter in range(iter_per_epoch):
        ind = mixed_ind [(iter*batch_size) : ((iter+1)*batch_size)] 
        x      = x_train[ind]
        lab_tr = labels_tr[ind]
        y_hat = forwardA(x, W0_a, b0_a, W1_a, b1_a)
        loss = lossF(y_hat, lab_tr)
        # print(loss)
        loss.backward()
        optimizerA.step()
        optimizerA.zero_grad()
    print(f"epoch {i}")
    y_hat_tr = forwardA(x_train, W0_a, b0_a, W1_a, b1_a)
    pred_tr, ind_tr = torch.max(y_hat_tr, dim=1)
    mismatch_tr = torch.nonzero(ind_tr - labels_tr)
    print("Training Accuracy: " , (nTrain - mismatch_tr.shape[0])/nTrain)
    acc = (nTrain - mismatch_tr.shape[0])/nTrain 
    loss_list.append(loss.item())
print("Epoch count:",i)
#In[]
plt.plot(loss_list)
plt.title(f'Model a Loss - Epoch for LR = {rate}, batch size = {batch_size}')
plt.xlabel('Epoch')
plt.ylabel('Loss ')
# plt.legend()
plt.savefig(f'A_lossperepoch_lr{rate}_batch{batch_size}_allrequire.png')
plt.show()
print("Final loss training: " ,loss_list[-1])

y_hat_ts = forwardA(x_test, W0_a, b0_a, W1_a, b1_a)
pred_ts, ind_ts = torch.max(y_hat_ts, dim=1)
mismatch_ts = torch.nonzero(ind_ts - labels_ts)
print("Testing Accuracy: " , (nTest - mismatch_ts.shape[0])/nTest)
loss = lossF(y_hat_ts, labels_ts)
print("Test Loss: ", loss.item())
#In[]

#       PART B

W0_b = torch.empty(32, d).uniform_(-a(d), a(d)) 
W1_b = torch.empty(32, 32).uniform_(-a(32), a(32)) 
W2_b = torch.empty(10, 32).uniform_(-a(32), a(32)) 

b0_b = torch.empty(32).uniform_(-a(d), a(d)) 
b1_b = torch.empty(32).uniform_(-a(32), a(32)) 
b2_b = torch.empty(10).uniform_(-a(32), a(32))


W0_b.requires_grad_(True)
W1_b.requires_grad_(True)
W2_b.requires_grad_(True)

b0_b.requires_grad_(True)
b1_b.requires_grad_(True)
b2_b.requires_grad_(True)

rate = 0.001
optimizerB = optim.Adam([W0_b,b0_b,W1_b,b1_b,W2_b,b2_b], lr=rate)
batch_size = 2000
mixed_ind = torch.randperm(nTrain)

iter_per_epoch = int(nTrain/batch_size) #60

lossF = func.cross_entropy
i=0
acc = 0.00
loss_list_b = []
while acc<0.99: 
    i += 1
    for iter in range(iter_per_epoch):
        ind = mixed_ind [(iter*batch_size) : ((iter+1)*batch_size)] 
        x      = x_train[ind]
        lab_tr = labels_tr[ind]
        y_hat = forwardB(x, W0_b, b0_b, W1_b, b1_b, W2_b, b2_b)
        loss = lossF(y_hat, lab_tr)
        # print(loss)
        loss.backward()
        optimizerB.step()
        optimizerB.zero_grad()
    print(f"epoch {i}")
    y_hat_tr = forwardB(x_train, W0_b, b0_b, W1_b, b1_b, W2_b, b2_b)
    pred_tr, ind_tr = torch.max(y_hat_tr, dim=1)
    mismatch_tr = torch.nonzero(ind_tr - labels_tr)
    print("Training Accuracy: " , (nTrain - mismatch_tr.shape[0])/nTrain)
    acc = (nTrain - mismatch_tr.shape[0])/nTrain 
    loss_list_b.append(loss.item())
print("Epoch count:",i)

#In[]
plt.plot(loss_list_b)
plt.title(f'Model b Loss - Epoch for LR = {rate}, batch size = {batch_size}')
plt.xlabel('Epoch')
plt.ylabel('Loss ')
# plt.legend()
plt.savefig(f'B_lossperepoch_lr{rate}_batch{batch_size}.png')
plt.show()
print("Final loss training: " ,loss_list_b[-1])

y_hat_ts = forwardB(x_test, W0_b, b0_b, W1_b, b1_b, W2_b, b2_b)
pred_ts, ind_ts = torch.max(y_hat_ts, dim=1)
mismatch_ts = torch.nonzero(ind_ts - labels_ts)
print("Testing Accuracy: " , (nTest - mismatch_ts.shape[0])/nTest)
loss = lossF(y_hat_ts, labels_ts)
print("Test Loss: ", loss.item())
# %%