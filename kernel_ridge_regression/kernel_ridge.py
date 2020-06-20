# In[]

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

n = 30
x = np.sort(np.random.rand(n,1), axis = 0)
y = (4 * np.sin(np.pi * x) * np.cos(6 * np.pi * np.square(x))).ravel()
eps = np.random.normal(0,1,y.shape)
y = y + eps

print("random data generated")

# In[]


def train(K, Y, lbda):
    n = len(K)
    reg_matrix = lbda * np.eye(n)

    trained = linalg.solve((K + reg_matrix), (Y))   
    return trained

def predict_poly(alpha, X, Xorg, d):
    n1 = len(X)
    n2 = len(Xorg)
    f_ = np.zeros(n1)
    K = kPolyxz(Xorg,X,d)
    f_ = np.matmul(alpha, K)
    return f_

def predict_rbf(alpha, X, Xorg, g):
    n1 = len(X)
    n2 = len(Xorg)
    f_ = np.zeros(n1)
    K = kRBFxz(Xorg,X,g)
    f_ = np.matmul(alpha, K) 
    return f_

def error(f_, y):
    return np.average(np.square(f_ - y))

def kPolyx(x,d):
    n = len(x)
    k = np.outer(x,x)
    k = np.ones((n,n)) + k
    k = np.power(k,d)
    return k

def kPolyxz(x,z,d):
    n1 = len(x)
    n2 = len(z)
    k = np.outer(x,z)
    k = np.ones((k.shape)) + k
    k = np.power(k,d)
    return k

def kRBFx(x,g):
    n = len(x)
    X = np.broadcast_to(x,(n,n))
    k = np.square(X.T - X) * -1 * g
    return np.exp(k)

def kRBFxz(x,z,g):
    n = len(x)
    nz = len(z)
    z_ = np.zeros((nz,1))
    for i in range(nz):
        z_[i] = z[i]
    X = np.broadcast_to(x,(n,nz))
    Z = np.broadcast_to(z_,(nz,n))
    k = np.square(X - Z.T) * -1 * g
    return np.exp(k)


def loss(K, alpha, y, lbda):

    s = y - np.matmul(K,alpha)
    S = np.square(linalg.norm(s,2))

    return S + lbda * alpha.T.dot(K.dot(alpha))

# In[]

# Leave-One-Out

d_list = [1,4,10,15,20,25,30,35,40,45,50,55,60,70,80]
g_list = [1,4,10,15,20,25,30,35,40,45,50,55,60,70,80]
lbda_list = [1e-5,1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4]


i_array = np.arange(n)

poly_loss_grid = {}
rbf_loss_grid = {}
poly_error_grid = {}
rbf_error_grid = {}

for lbda in lbda_list:
    for d in d_list:
        poly_errors = []
        poly_losses = []
        for one in range(n):
            i_tr = np.delete(i_array, one)
            i_va = one
            Xtr = x[i_tr]
            Xva = x[i_va]
            Ytr = y[i_tr]
            Yva = y[i_va]

            Ntr = len(Xtr)
            Ktr_poly = kPolyx(Xtr,d)
            Nva = len(Xva)
            Kva_poly = kPolyx(Xva,d)
            alpha_poly = train(Ktr_poly, Ytr, lbda)
            loss_poly = loss(Ktr_poly, alpha_poly, Ytr, lbda)
            poly_losses.append(loss_poly)
            fhat_poly_va = predict_poly(alpha_poly, Xva, Xtr, d)
            error_poly = error(fhat_poly_va, Yva)
            poly_errors.append(error_poly)
        poly_error_grid[(lbda,d)] = np.mean(poly_errors)
        poly_loss_grid[(lbda,d)] = np.mean(poly_losses)
        
    for g in g_list:
        rbf_errors = []
        rbf_losses = []
        for one in range(n):
            i_tr = np.delete(i_array, one)
            i_va = one        
            Xtr = x[i_tr]
            Xva = x[i_va]
            Ytr = y[i_tr]
            Yva = y[i_va]

            Ntr = len(Xtr)
            Ktr_rbf = kRBFx(Xtr,g)
            Nva = len(Xva)
            Kva_rbf = kRBFx(Xva,g)
            alpha_rbf = train(Ktr_rbf, Ytr, lbda)
            loss_rbf = loss(Ktr_rbf, alpha_rbf, Ytr, lbda)
            rbf_losses.append(loss_rbf)
            fhat_rbf_va  = predict_rbf(alpha_rbf , Xva, Xtr, g)
            error_rbf  = error(fhat_rbf_va,  Yva)
            rbf_errors.append(error_rbf)
        rbf_error_grid[(lbda,g)] = np.mean(rbf_errors)
        rbf_loss_grid[(lbda,g)] = np.mean(rbf_losses)


# In[]

d_list = [1,4,10,15,20,25,30,35,40,45,50,55,60,70,80]
g_list = [1,4,10,15,20,25,30,35,40,45,50,55,60,70,80]
lbda_list = [1e-5,1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4]

x_r = np.linspace(0.0,1.0,100)
f_true = 4 * np.sin(np.pi * x_r) * np.cos(6 * np.pi * np.square(x_r))
lw = 2
kernel_label = ['RBF', 'Poly']
model_color = ['m', 'g','b']

for lbda in lbda_list:
    for d in d_list:
        Kpoly = kPolyx(x,d)
        alphap = train(Kpoly, y, lbda)
        fhat_poly = predict_poly(alphap, x_r, x, d)
        plt.plot(x_r, fhat_poly, color=model_color[0], label='fhat(x)')
        plt.plot(x, y, 'o', color=model_color[1], label='y')
        plt.plot(x_r, f_true, color=model_color[2], label='ftrue(x)')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.ylim((-6,6))
        plt.title(f"Poly Kernel with lbda={lbda} d={d}")
        plt.legend()
        plt.savefig(f"A3_B_plots/HW3_A3B_poly_d{d}_{lbda}_{n}.png")
        plt.show()

    for g in g_list:
        Krbf = kRBFx(x,g)
        alphar = train(Krbf, y, lbda)
        fhat_rbf = predict_rbf(alphar, x_r, x, g)   
        plt.plot(x_r, fhat_rbf, color=model_color[0], label='fhat(x)')
        plt.plot(x, y, 'o', color=model_color[1], label='y')
        plt.plot(x_r, f_true, color=model_color[2], label='ftrue(x)')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.ylim((-6,6))
        plt.title(f"RBF Kernel with lbda={lbda} g={g}")
        plt.legend()
        plt.savefig(f"A3_B_plots/HW3_A3B_rbf_g{g}_{lbda}_{n}.png")
        plt.show()

# In[]
B = 300
g = 28
d = 20
lbda_rbf = 0.001
lbda_poly = 0.0001
finegrid = 300
ind = np.linspace(0,n-1,n, dtype=int)
x_r = np.linspace(0.0,1.0,finegrid)
f_true = 4 * np.sin(np.pi * x_r) * np.cos(6 * np.pi * np.square(x_r))
lw = 2
kernel_label = ['RBF', 'Poly']
model_color = ['m', 'g','b']
f_hat_poly_b = np.zeros((B, finegrid))
f_hat_rbf_b = np.zeros((B, finegrid))
lower95 = np.zeros((finegrid))
upper95 = np.zeros((finegrid))
lower5 = np.zeros((finegrid))
upper5 = np.zeros((finegrid))
lower95_r = np.zeros((finegrid))
upper95_r = np.zeros((finegrid))
lower5_r = np.zeros((finegrid))
upper5_r = np.zeros((finegrid))

for b in range(B):
    choices = np.sort(np.random.choice(ind,n,replace=True))
    x_boot = x[choices]
    y_boot = y[choices]
    
    Kpoly = kPolyx(x_boot,d)
    alphap = train(Kpoly, y_boot, lbda_poly)
    fhat_poly = predict_poly(alphap, x_r, x_boot, d)
    f_hat_poly_b[b] = fhat_poly

    Krbf = kRBFx(x_boot,g)
    alphar = train(Krbf, y_boot, lbda_rbf)
    fhat_rbf = predict_rbf(alphar, x_r, x_boot, g)   
    f_hat_rbf_b[b] = fhat_rbf

for i in range(finegrid):
    colm_poly = np.sort(f_hat_poly_b[:,i]) #length 300
    lower95[i] = colm_poly[8]
    upper95[i] = colm_poly[292]
    lower5[i]  = colm_poly[143]
    upper5[i]  = colm_poly[156]

    colm_rbf = np.sort(f_hat_rbf_b[:,i]) #length 300
    lower95_r[i] = colm_rbf[8]
    upper95_r[i] = colm_rbf[292]
    lower5_r[i]  = colm_rbf[143]
    upper5_r[i]  = colm_rbf[156]
# In[]

Krbf = kRBFx(x,g)
alphar = train(Krbf, y, lbda_rbf)
fhat_rbf = predict_rbf(alphar, x_r, x, g)   

plt.plot(x_r, fhat_rbf, label='fhat(x)')
plt.plot(x, y, 'o', label='y')
plt.plot(x_r, lower95_r, label='lower95')
plt.plot(x_r, upper95_r, label='upper95')
plt.legend()
plt.ylim((-7,7))
plt.title(f"RBF Kernel with 95% confidence interval lbda={lbda_rbf} g={g}")
plt.savefig(f"HW3_A3D_95rbf_g{g}_{lbda_rbf}_{n}_fine300.png")

plt.show()

plt.plot(x_r, fhat_rbf, label='fhat(x)')
plt.plot(x, y, 'o', label='y')
plt.plot(x_r, lower5_r, label='lower5')
plt.plot(x_r, upper5_r, label='upper5') 
plt.legend()
plt.ylim((-5,5))

plt.title(f"RBF Kernel with 5% confidence interval lbda={lbda_rbf} g={g}")
plt.savefig(f"HW3_A3D_5rbf_g{g}_{lbda_rbf}_{n}_fine300.png")

plt.show()

plt.plot(x_r, fhat_poly, label='fhat(x)')
plt.plot(x, y, 'o', label='y')
plt.plot(x_r, lower95, label='lower95')
plt.plot(x_r, upper95, label='upper95')
plt.legend()
plt.ylim((-7,7))
plt.title(f"Poly Kernel with 95% confidence interval lbda={lbda_poly} d={d}")
plt.savefig(f"HW3_A3D_95poly_d{d}_{lbda_poly}_{n}_fine300.png")

plt.show()

plt.plot(x_r, fhat_poly, label='fhat(x)')
plt.plot(x, y, 'o', label='y')
plt.plot(x_r, lower5, label='lower5')
plt.plot(x_r, upper5, label='upper5') 
plt.legend()
plt.ylim((-5,5))

plt.title(f"Poly Kernel with 5% confidence interval lbda={lbda_poly} d={d}")
plt.savefig(f"HW3_A3D_5poly_d{d}_{lbda_poly}_{n}_fine300.png")
plt.show()


# In[]

# 10-fold CV

d_list = [1,4,10,15,20,25,30,35,40,45,50,55,60,70,80]
g_list = [1,4,10,15,20,25,30,35,40,45,50,55,60,70,80]
lbda_list = [1e-5,1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4]

i_array = np.arange(n)
batch_size=30
i_array = np.random.permutation(i_array)
print(f"i_array:{i_array}")

poly_loss_grid = {}
rbf_loss_grid = {}
poly_error_grid = {}
rbf_error_grid = {}

for lbda in lbda_list:
    for d in d_list:
        poly_errors = []
        poly_losses = []
        for fold in range(10):
            indd = i_array[fold*batch_size : (fold+1)*batch_size]
            i_tr = np.concatenate((i_array[:fold*batch_size], i_array[(fold+1)*batch_size:]))
            i_va = indd        
            Xtr = x[i_tr]
            Xva = x[i_va]
            Ytr = y[i_tr]
            Yva = y[i_va]

            Ntr = len(Xtr)
            Ktr_poly = np.zeros((Ntr,Ntr)) #29x29
            Ktr_poly = kPolyx(Xtr,d)

            Nva = len(Xva)
            Kva_poly = np.zeros((Nva,Nva))
            Kva_poly = kPolyx(Xva,d)

            alpha_poly = train(Ktr_poly, Ytr, lbda)

            loss_poly = loss(Ktr_poly, alpha_poly, Ytr, lbda)

            poly_losses.append(loss_poly)

            fhat_poly_va = predict_poly(alpha_poly, Xva, Xtr, d)

            error_poly = error(fhat_poly_va, Yva)
            
            poly_errors.append(error_poly)

        poly_error_grid[(lbda,d)] = np.mean(poly_errors)
        poly_loss_grid[(lbda,d)] = np.mean(poly_losses)
    for g in g_list:
        rbf_errors = []
        rbf_losses = []
        for fold in range(10):
            indd = i_array[fold*batch_size : (fold+1)*batch_size]
            i_tr = np.concatenate((i_array[:fold*batch_size], i_array[(fold+1)*batch_size:]))
            i_va = indd        
            Xtr = x[i_tr]
            Xva = x[i_va]
            Ytr = y[i_tr]
            Yva = y[i_va]


            Ntr = len(Xtr)
            Ktr_rbf = np.zeros((Ntr,Ntr))
            Ktr_rbf = kRBFx(Xtr,g)

            Nva = len(Xva)
            Kva_rbf = np.zeros((Nva,Nva))
            Kva_rbf = kRBFx(Xva,g)

            alpha_rbf = train(Ktr_rbf, Ytr, lbda)

            loss_rbf = loss(Ktr_rbf, alpha_rbf, Ytr, lbda)

            rbf_losses.append(loss_rbf)

            fhat_rbf_va  = predict_rbf(alpha_rbf , Xva, Xtr, g)

            error_rbf  = error(fhat_rbf_va,  Yva)
            
            rbf_errors.append(error_rbf)

        rbf_error_grid[(lbda,g)] = np.mean(rbf_errors)
        rbf_loss_grid[(lbda,g)] = np.mean(rbf_losses)



#In[]

# A3.e
B=300

d=20
g=28
lbda_rbf = 0.001
lbda_poly = 0.0001

m = 1000
x_ = np.sort(np.random.rand(m,1), axis = 0)
y_ = (4 * np.sin(np.pi * x_) * np.cos(6 * np.pi * np.square(x_))).ravel()
eps_ = np.random.normal(0,1,y_.shape)
y_ = y_ + eps_

finegrid = 1000
x_r = np.linspace(0.0,1.0,finegrid)
import time

indic = np.linspace(0,m-1,m, dtype=int)

mse_diff = np.zeros((B))
f_hat_poly_b = np.zeros((B,m))
f_hat_rbf_b = np.zeros((B,m))
print("bootstrapping")
start = time.time()
for b in range(B):
    choices = np.sort(np.random.choice(indic,m,replace=True))
    x_boot = x_[choices]
    y_boot = y_[choices]
    
    Kpoly = kPolyx(x_boot,d)
    alphap = train(Kpoly, y_boot, lbda_poly)
    fhat_poly = predict_poly(alphap, x_r, x_boot, d)
    f_hat_poly_b[b] = fhat_poly

    print(f"time per b end of poly: {time.time()-start}")
    start = time.time()

    Krbf = kRBFx(x_boot,g)
    alphar = train(Krbf, y_boot, lbda_rbf)
    fhat_rbf = predict_rbf(alphar, x_r, x_boot, g)   
    f_hat_rbf_b[b] = fhat_rbf

    print(f"time per b end of rbf: {time.time()-start}")
    start = time.time()

    mse_diff[b] = np.average(np.square(y_boot - fhat_poly)-np.square(y_boot - fhat_rbf))
    print(f"time per b end of diff: {time.time()-start}, mse_diff[{b}]={mse_diff[b]}")
    start = time.time()


mse_diff = np.sort(mse_diff)

dlower95 = mse_diff[8]
dupper95 = mse_diff[292]
dlower5  = mse_diff[143]
dupper5  = mse_diff[156]

print(dlower95,dupper95,dlower5,dupper5)
# 0.03999296960733489 0.8226023409500371 0.5292052841130107 0.5495539163603539