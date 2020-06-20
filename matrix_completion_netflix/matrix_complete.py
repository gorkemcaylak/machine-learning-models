# In[]
import numpy as np
import matplotlib.pyplot as plt
import csv
import torch
import torch.optim as optim
import torch.nn.functional as func
import scipy as sc
import scipy.sparse.linalg as linalg

data = []
with open("u.data") as csvfile:
    spamreader = csv.reader(csvfile, delimiter="\t")
    for row in spamreader:
        data.append([int(row[0])-1, int(row[1])-1, int(row[2])])

data = np.array(data)
num_observations = len(data) # num_observations = 100,000
num_users = max(data[:,0])+1 # num_users = 943, indexed 0,...,942    #m
num_items = max(data[:,1])+1 # num_items = 1682 indexed 0,...,1681   #n

np.random.seed(1)
num_train = int(0.8*num_observations)
perm = np.random.permutation(data.shape[0])
train = data[perm[0:num_train],:]   #80000,3   80%
test = data[perm[num_train::],:]    #20000,3   20%

'''
[user, movie, score]
[  j ,   i  ,   s  ]
'''
m = num_items #1682
n = num_users #943 
#100,000 total reviews

'''
Goal:recommend movies user hasn't seen
Low-rank matrix factorization

it learns a vector representation ui ∈ Rd
for each movie and a vector representation vj ∈ Rd
for each user, such that the inner product
<ui,vj> approximates the rating Ri,j .

R , m x n matrix, each element score 1 to 5  
R_i,j = jth user on ith movie

'''


#In[]

def errorR(Rhat, Rreal):
    r, c = Rreal.shape
    # sum = 0
    # count = 0
    ind = np.nonzero(Rreal) #20000 indices for test
    # print(Rreal[ind].shape)
    # print(np.count_nonzero(Rreal[ind]))
    # print(Rhat[ind].shape)
    # print(np.count_nonzero(Rhat[ind]))

    diff = np.square(Rhat[ind] - Rreal[ind])

    return diff.mean()


def allocateR(R, data):
    for row in data:
        R[row[1],row[0]] = row[2]


'''
[43, 0, 2]
[23, 0, 4]
[12, 0, 1]
[13, 1, 3]
[42, 1, 3]
[11, 1, 5]
[35, 1, 1]

'''
MuD = {}
justMovies = train[:,1:]
# print(justMovies[:20])
# trainMovies = sorted(justMovies, axis=0)
# trainMovies = justMovies[justMovies[:,0].argsort()]
# print(trainMovies[:50])

for row in train:
    if row[1] not in MuD.keys():
        MuD[row[1]] = [1,row[2]]
    else:
        MuD[row[1]][0] += 1
        MuD[row[1]][1] += row[2]
 
mu = np.zeros(num_items)

for i in range(num_items):
    if i not in MuD.keys():
        mu[i] = 0
    else:
        count = MuD[i][0]
        sum = MuD[i][1]
        mu[i] = sum/count

ons = np.ones((n))
R_mu = np.outer(mu, ons)

Rtrain = np.zeros((m,n))
allocateR(Rtrain,train)
# print(R.shape) #(1682, 943)

Rtest = np.zeros((m,n))
allocateR(Rtest,test)

mu_error = errorR(R_mu, Rtest)
print(mu_error)  #1.063564200567445



# In[]
d_list = [1,2,5,10,20,50]
train_error = np.zeros(6)
test_error = np.zeros(6)
i=0
for d in d_list:
    U,S,V = linalg.svds(Rtrain,d)
    Score = np.dot(Rtrain.T,U)
    R_hat = (Score.dot(U.T)).T
    tr_error = errorR(R_hat, Rtrain)
    ts_error = errorR(R_hat, Rtest)
    train_error[i] = tr_error
    test_error[i] = ts_error
    i+=1
    print(f"d={d}, train error={tr_error}, test error={ts_error}")

plt.title('Error vs top d eigenvalues used')
plt.xlabel('d')
plt.ylabel('error')
plt.plot(d_list, train_error, label='Training Error')
plt.plot(d_list, test_error, label='Testing Error')
# plt.xscale('log')
plt.legend()
plt.savefig('top_d.png')
plt.show()



# In[]

def loss(uhat, vhat, R, lbda):
    r, c = R.size()
    r_ = uhat.size()
    c_ = vhat.size()

    Rhat = torch.matmul(uhat, vhat.T)
    
    ind = torch.nonzero(R)
    diff = Rhat[ind] - R[ind]
    # diff = R - Rhat
    sqdiff = torch.square(diff)
    unorm = torch.sum(torch.square(uhat))
    vnorm = torch.sum(torch.square(vhat))
    
    return torch.sum(torch.sum(sqdiff)) + lbda * unorm + lbda * vnorm



#In[]
'''

    closed form method

    alternating least squares

'''

def solve_for_u(u, v, R, lbda):
    r = v.size()[1] #use d!
    M = u.size()[0]
    vT = torch.transpose(v,0,1)
    # uT = torch.transpose(u,0,1)
    reg_matrix = lbda * np.eye(r)
    k = torch.matmul(vT,v) + reg_matrix
    kinv = torch.inverse(k).float()
    Uhat = torch.matmul(torch.matmul(R,v) , kinv)
    return Uhat


def solve_for_v(u, v, R, lbda):
    r = u.size()[1]
    N = v.size()[0]
    RT = torch.transpose(R,0,1)
    uT = torch.transpose(u,0,1)
    reg_matrix = lbda * np.eye(r)
    k = torch.matmul(uT,u) + reg_matrix
    kinv = torch.inverse(k).float()
    Vhat = torch.matmul(torch.matmul(RT,u) , kinv)
    return Vhat


def errorFromV(uhat, vhat, Rreal):
    sum = 0
    count = 0
    for j in range(m): #m
        for i in range(n): #n
            v = vhat[i, :]
            u = uhat[j, :]
            real = Rreal[j,i]
            if(real > 0):
                pred_ji =  torch.dot(u,v)
                dif = real-pred_ji
                sum += dif * dif
                count += 1
    
    return sum/count if count>0 else -1



lbda_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
sigma_list = [0.001, 0.002, 0.005, 0.02, 0.01, 0.1, 0.3]
d_list = [1,2,5,10,20,50]
train_error = np.zeros(6)
test_error = np.zeros(6)
i=0
for lbda in lbda_list:
    for sigma in sigma_list:
        d_ind = 0
        for d in d_list:
            i=0
            R_torch = torch.from_numpy(Rtrain).float()

            uhat = torch.rand((m,d)) * sigma
            vhat = torch.rand((n,d)) * sigma


            ls = np.infty
            lsv = np.infty
            prev_ls = torch.zeros(1)
            prev_lsv = torch.zeros(1)
            now_ls = torch.ones(1)
            now_lsv = torch.ones(1)
            change = 1
            loss_list = []
            while change > 0.005: 
                uhatprev = uhat.clone()
                vhatprev = vhat.clone()
                # print("in")
                i += 1
                uhat = solve_for_u(uhat, vhat, R_torch, lbda)
                vhat = solve_for_v(uhat, vhat, R_torch, lbda)


                change = torch.mean(torch.abs(uhatprev - uhat)) + torch.mean(torch.abs(vhatprev - vhat))
     

            train_er = errorFromV(uhat,vhat,Rtrain)#errorR(RHat, Rtrain)
            test_er = errorFromV(uhat,vhat,Rtest)#errorR(RHat, Rtest)
            print(f"\nd:{d} L:{lbda} S:{sigma} e:{i} TrEr:{train_er} TsEr:{test_er}")
            train_error[d_ind] = train_er
            test_error[d_ind] = test_er
            d_ind += 1
            

plt.title('Alternating Minimization - error vs d')
plt.xlabel('d')
plt.ylabel('error')
plt.plot(d_list, train_error, label='Training Error')
plt.plot(d_list, test_error, label='Testing Error')
plt.legend()
plt.savefig('als.png')
plt.show()



'''



SGD


'''


def grad_loss_u(uhat, vhat, R, lbda):
    Rhat = torch.matmul(uhat, vhat.T)
    diff = Rhat - R
    return 2 * torch.matmul(diff,vhat) + 2 * lbda * uhat

def grad_loss_v(uhat, vhat, R, lbda):
    Rhat = torch.matmul(uhat, vhat.T)
    diff = Rhat - R
    return 2 * torch.matmul(diff.T,uhat) + 2 * lbda * vhat


rate_list = [0.0001, 0.001, 0.01, 0.1, 1]
lbda_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
sigma_list = [0.001, 0.002, 0.005, 0.01, 0.1, 0.3]
d_list = [1,2,5,10,20,50]
b_list = [250, 500, 2000, 4000]

Train = torch.from_numpy(train).float()
R_torch = torch.from_numpy(Rtrain).float()
mixed_ind = torch.randperm(num_train)

it=0
train_ac=[]
test_ac=[]
train_error = np.zeros(6)
test_error = np.zeros(6)
for base_rate in rate_list:
    for lbda in lbda_list:
        for sigma in sigma_list:
            for b in b_list:
                d_ind = 0
                for d in d_list:
                    rate = base_rate
                    it=0

                    iter_per_epoch = int(num_train/b) 

                    uhat = torch.rand((m,d)) * sigma
                    vhat = torch.rand((n,d)) * sigma

                    # uhat.requires_grad_(True)
                    # vhat.requires_grad_(True)

                    # optimizerU = optim.Adam([uhat], lr=rate)
                    # optimizerV = optim.Adam([vhat], lr=rate)

                    ls = 1
                    prev_ls = 0
                    prev_acc = np.ones(1)
                    acc = np.zeros(1)
                    # loss_list = []
                    avegrad = 100
                    change = 1
                    
                    while it<15: #np.abs(ls-prev_ls) > 0.1:  #np.abs(prev_acc - acc) > 0.001 :
                        rate *= 0.99
                        # prev_ls = ls
      
                        # uhatprev = uhat.clone()
                        # vhatprev = vhat.clone()

                        it += 1
                        avegrad = 0
                        for iter in range(iter_per_epoch):
                            ind = mixed_ind [(iter*b) : ((iter+1)*b)] 
                            x = Train[ind]

                            
                            real = x[:,2]
                            i = x[:,1].long()
                            j = x[:,0].long()
            
                            u = uhat[i]
                            v = vhat[j]
                            diff = torch.matmul(u, torch.transpose(v,0,1)) - real
              
                            grad_u = (2*torch.matmul(diff,v) + 2*lbda * u)
                            grad_v = (2*torch.matmul(diff.T,u) + 2*lbda * v)
                          
                            uhat[i] = u - rate*grad_u
                            vhat[j] = v - rate*grad_v
                           
                        # ls = loss(uhat, vhat, R_torch, lbda)
                        # print(ls)
                    
                        Rhat = torch.matmul(uhat, vhat.T)
                        RHat = Rhat.detach().numpy()
                        # print(f"d:{d} L:{lbda} S:{sigma} R:{rate} b:{b} e:{it} TrEr:{errorR(RHat, Rtrain)} TsEr:{errorR(RHat, Rtest)}")
                        trer = errorR(RHat, Rtrain)
                        tser = errorR(RHat, Rtest)
                        train_ac.append(trer)
                        test_ac.append(tser)
                        print(f"d:{d} L:{lbda} S:{sigma} R:{rate} b:{b} e:{it} TrEr:{trer} TsEr:{tser}")
                        change = torch.mean(torch.abs(uhatprev - uhat)) + torch.mean(torch.abs(vhatprev - vhat))

                    # train_er = errorFromV(uhat,vhat,Rtrain)#errorR(RHat, Rtrain)
                    # test_er = errorFromV(uhat,vhat,Rtest)#errorR(RHat, Rtest)
                    # print(f"\nd:{d} L:{lbda} S:{sigma} e:{i} TrEr:{train_er} TsEr:{test_er}")
                    Rhat = torch.matmul(uhat, vhat.T)
                    RHat = Rhat.detach().numpy()
                    train_er = errorR(RHat, Rtrain)
                    test_er = errorR(RHat, Rtest)
                    train_error[d_ind] = train_er
                    test_error[d_ind] = test_er
                    d_ind += 1
                    
                    print(f"Final:  d:{d} L:{lbda} S:{sigma} R:{base_rate} b:{b} e:{it} TrEr:{train_er} TsEr:{test_er}")
            
plt.title('SGD - error vs d')
plt.xlabel('d')
plt.ylabel('error')
plt.plot(d_list, train_error, label='Training Error')
plt.plot(d_list, test_error, label='Testing Error')
# plt.xscale('log')
plt.legend()
plt.savefig('sgd.png')
plt.show()

# # %%


'''




SGD with biases added




'''

base_rate = 0.006
lbda = 0.000001
sigma = 0.002
d = 1
b = 1

Train = torch.from_numpy(train).float()
R_torch = torch.from_numpy(Rtrain).float()
mixed_ind = torch.randperm(num_train)

it=0
train_ac=[]
test_ac=[]
train_error = np.zeros(6)
test_error = np.zeros(6)

rate = base_rate
it=0

iter_per_epoch = int(num_train/b) 

uhat = torch.rand((m,d)) * sigma
vhat = torch.rand((n,d)) * sigma

ubias = torch.rand((m)) * sigma
vbias = torch.rand((n)) * sigma

ls = 1
prev_ls = 0
prev_acc = np.ones(1)
acc = np.zeros(1)
Rhat = torch.matmul(uhat, vhat.T)
Rhat += ubias.repeat(n,1).T
Rhat += vbias.repeat(m,1)

while it<40:
    rate *= 0.99

    it += 1
    for iter in range(iter_per_epoch):
        ind = mixed_ind [(iter*b) : ((iter+1)*b)] 
        x = Train[ind]
        
        real = x[:,2]
        i = x[:,1].long() #users
        j = x[:,0].long() #movies
      
        u = uhat[i] #d size
        v = vhat[j]
        ub = ubias[i]
        vb = vbias[j]

        diff = (ub + vb + torch.matmul(u.T, v) - real)
        ubias[i] = ub - rate * (diff + lbda * ub)
        vbias[j] = vb - rate * (diff + lbda * vb)
        uhat[i] = u - rate * (torch.matmul(diff,v) + lbda * u)
        vhat[j] = v - rate * (torch.matmul(diff.T,u) + lbda * v)
        
    Rhat = torch.matmul(uhat, vhat.T)
    Rhat += ubias.repeat(n,1).T
    Rhat += vbias.repeat(m,1)                        
    RHat = Rhat.detach().numpy()
    trer = errorR(RHat, Rtrain)
    tser = errorR(RHat, Rtest)
    train_ac.append(trer)
    test_ac.append(tser)
    print(f"d:{d} L:{lbda} S:{sigma} R:{rate} b:{b} e:{it} TrEr:{trer} TsEr:{tser}")
    change = torch.mean(torch.abs(uhatprev - uhat)) + torch.mean(torch.abs(vhatprev - vhat))


Rhat = torch.matmul(uhat, vhat.T)
Rhat += ubias.repeat(n,1).T
Rhat += vbias.repeat(m,1)
RHat = Rhat.detach().numpy()
train_er = errorR(RHat, Rtrain)
test_er = errorR(RHat, Rtest)
train_error[d_ind] = train_er
test_error[d_ind] = test_er
d_ind += 1

print(f"Final:  d:{d} L:{lbda} S:{sigma} R:{base_rate} b:{b} e:{it} TrEr:{train_er} TsEr:{test_er}")

plt.title('SGD with bias - error vs epoch d=1')
plt.xlabel('epoch')
plt.ylabel('error')
plt.plot(train_ac[1:], label='Training Error')
plt.plot(test_ac[1:], label='Testing Error')
plt.legend()
plt.savefig(f'sgd_bias_perepoch_lr{base_rate}.png')
plt.show()

# # %%


