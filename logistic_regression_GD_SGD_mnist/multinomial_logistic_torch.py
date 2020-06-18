    # In[]
    import numpy as np
    import matplotlib.pyplot as plt
    from mnist import MNIST
    import torch

    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training()) 
    X_test, labels_test = map(np.array, mndata.load_testing()) 

    X_train = X_train/255.0 # nTr x d    60000 x 784
    X_test = X_test/255.0   # nTs x d    10000 x 784    

    nTrain, d = X_train.shape # 60000 , 784
    nTest = len(labels_test)  # 10000

    k = 10
    Lambda = 1e-6

    Y_train = np.zeros((nTrain, k))
    Y_test = np.zeros((nTest, k))

    #one-hot coding
    for i in range(nTrain):
        hot = labels_train[i]
        Y_train[i][hot] = 1
    # print(Y_train[:30])
    for i in range(nTest):
        hot = labels_test[i]
        Y_test[i][hot] = 1

    x_train = torch.from_numpy(X_train).float()
    x_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(Y_train).float()
    y_test = torch.from_numpy(Y_test).float()

    labels_tr = torch.from_numpy(labels_train).long()
    labels_ts = torch.from_numpy(labels_test).long()

    step_size = 0.4
    W = torch.zeros(784, 10, requires_grad=True)
    W.grad = torch.zeros(784, 10)
    update = 1
    i=0
    conv_check = 0.001

    print("Cross entropy, rate " + str(step_size) + ", conv check " + str(conv_check)+ " running...")
    while update > conv_check and i<1000:
        i += 1
        y_hat = torch.matmul(x_train, W)
        loss = torch.nn.functional.cross_entropy(y_hat, labels_tr)
        loss.backward()
        W.data = W.data - step_size * W.grad
        update = torch.max(torch.abs(W.grad))
        W.grad.zero_() 

    print("Cross entropy iteration count:",i)

    y_hat_tr = torch.matmul(x_train, W)
    pred_tr, ind_tr = torch.max(y_hat_tr, dim=1)
    mismatch_tr = torch.nonzero(ind_tr - labels_tr)
    print("Training Accuracy: " , (nTrain - mismatch_tr.shape[0])/nTrain)

    y_hat_ts = torch.matmul(x_test, W)
    pred_ts, ind_ts = torch.max(y_hat_ts, dim=1)
    mismatch_ts = torch.nonzero(ind_ts - labels_ts)
    print("Testing Accuracy: " , (nTest - mismatch_ts.shape[0])/nTest)

    # In[]

    step_size2 = 0.2
    W2 = torch.zeros(784, 10, requires_grad=True)
    W2.grad = torch.zeros(784, 10)
    update = 1
    i2=0
    conv_check2 = 0.0005

    print("MSE, rate " + str(step_size2) + ", conv check " + str(conv_check2)+ " running...")
    while update > conv_check2 and i2 < 1000:
        i2 += 1
        y_hat2 = torch.matmul(x_train, W2)
        loss2 = torch.nn.functional.mse_loss(y_hat2, y_train, size_average=True)
        loss2.backward()
        W2.data = W2.data - step_size2 * W2.grad
        update = torch.max(torch.abs(W2.grad))
        W2.grad = torch.zeros(784, 10)

    print("MSE iteration count:",i2)

    y_hat_tr = torch.matmul(x_train, W2)
    pred_tr, ind_tr = torch.max(y_hat_tr, dim=1)
    mismatch_tr = torch.nonzero(ind_tr - labels_tr)
    print("Training Accuracy: " , (nTrain - mismatch_tr.shape[0])/nTrain)

    y_hat_ts = torch.matmul(x_test, W2)
    pred_ts, ind_ts = torch.max(y_hat_ts, dim=1)
    mismatch_ts = torch.nonzero(ind_ts - labels_ts)
    print("Testing Accuracy: " , (nTest - mismatch_ts.shape[0])/nTest)


    # %%