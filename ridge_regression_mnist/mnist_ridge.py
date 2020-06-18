import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from polyreg import PolynomialRegression
from linreg_closedform import LinearRegressionClosedForm
from mnist import MNIST

def train(X, Y, Lambda):
    r,c = X.shape
    reg_matrix = Lambda * np.eye(c)
    W_ = linalg.solve((X.T.dot(X) + reg_matrix), X.T.dot(Y))
    return W_

def predict(W, X):
    Y = X.dot(W)
    n,k = Y.shape
    R = np.zeros(n)
    for i in range(n):
        max = 0
        max_index = 0
        for j in range(k):
            if Y[i][j] >= max:
                max_index = j
                max = Y[i][j]
        R[i] = int(max_index)
    return R

def h(G, x, b):
    h = np.cos(G.dot(x) + b)
    return h # size p

if __name__ == "__main__":

    # load the data
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training()) 
    X_test, labels_test = map(np.array, mndata.load_testing()) 
    # print(X_train[0])
    X_train = X_train/255.0 # n x d    
    X_test = X_test/255.0
    # print(X_train.shape) #60000 x 784
    # print(labels_train.shape) #60000
    # print(X_test.shape)  #10000 x 784
    # print(labels_test.shape)  #10000

    print(X_train[0:30])
    print(labels_train[0:30])

    nTrain, d = X_train.shape
    nTest = len(labels_test)

    k = 10
    Lambda = 1e-6

    Y = np.zeros((nTrain, k))
    #one-hot coding
    for i in range(nTrain):
        hot = labels_train[i]
        Y[i][hot] = 1

    p_count = 7
    p_arr = np.zeros(p_count)
    p_arr = 50 * pow(2,np.arange(p_count))
    
    # h(x) = cos(Gx + b)
    # G : p x d   - random with iid samples N(0, 0.1)
    # b : p       - random with iid samples uniform[0,2pi]
    ErrorTrP = np.zeros(p_count)
    ErrorVaP = np.zeros(p_count)
  
    for t in range(p_count):
        p = p_arr[t]
        Gvec = np.random.normal(0, 0.1, p*d)
        G = np.reshape(Gvec, (p,d))

        b = np.random.uniform(0, 2*np.pi, p)

        X_p = np.zeros((nTrain, p))
        X_pTest = np.zeros((nTest, p))

        for i in range(nTrain):
            X_p[i] = h(G, X_train[i], b)
        for i in range(nTest):
            X_pTest[i] = h(G, X_test[i], b)

        i_array = np.random.permutation(nTrain)
        nSplit = int( 0.8 * nTrain )
        i_tr = i_array[:nSplit]
        i_va = i_array[nSplit:]

        Xtr = X_p[i_tr, :]
        Xva = X_p[i_va, :]
        Ytr = Y[i_tr, :]
        lbTr = labels_train[i_tr,]
        lbVa = labels_train[i_va,]

        W_ = train(Xtr, Ytr, Lambda)
        
        R = predict(W_, Xtr)
        n = len(R)
        count = 0
        for i in range(n):
            if R[i] == lbTr[i]:
                count = count + 1
        print(f"Training Error= {100 - count/n*100}%")
        ErrorTr = 100 - count/n*100

        R = predict(W_, Xva)
        n = len(R)
        count = 0
        for i in range(n):
            if R[i] == lbVa[i]:
                count = count + 1
        print(f"Validation Error= {100 - count/n*100}%")
        ErrorVa = 100 - count/n*100
        ErrorTrP[t] = ErrorTr
        ErrorVaP[t] = ErrorVa
        
        
    print(ErrorTrP)
    plt.plot(p_arr, ErrorTrP, label = 'Training Error')
    plt.plot(p_arr, ErrorVaP, label = 'Validation Error')
    plt.title('Error for regL = '+str(Lambda))
    plt.xlabel('p')
    plt.ylabel('Error(%)')
    plt.legend()
    plt.savefig('Ridge_L='+str(Lambda)+'.png')

    plt.show()
    