# In[]

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from mnist import MNIST
import time

'''
1. Once a set of centroids \mu_k is available, the clusters are updated to contain the points closest in distance to each centroid
2. Given a set of clusters, the centroids are recalculated as the means of all points belonging to a cluster.
'''
# load the data
mndata = MNIST('./data/')
X_train, labels_train = map(np.array, mndata.load_training()) 
X_test, labels_test = map(np.array, mndata.load_testing()) 
X_train = X_train/255.0 # n x d    
X_test = X_test/255.0
run=0
# In[]

def dist(x,y):
    return np.sum(np.abs(x-y))

def display_means(means):
    means_np = np.array(means)
    # print("\nNew Iteration\n\n")
    if means_np.shape[0] == 784:
        plt.imshow(np.reshape(means,(28,28)))
        plt.show()
        return
    k = 0
    for mean in means_np: 
        plt.imshow(np.reshape(mean,(28,28)))
        plt.savefig("./centers/run_"+str(run)+"_iter_"+str(iter)+"_center_"+str(k)+".png")
        k += 1
    return

def calc_loss():
    sum = 0
    for i in range(k):
        di = points[i, :numbers[i]]-centers[i] #5000 x 784
        k2 = np.square(linalg.norm(di, 2))
        sum += k2
    return sum

def calc_min_loss():
    sum = 0
    i = np.argmin(numbers)
    di = points[i, :numbers[i]]-centers[i] #5000 x 784
    k2 = np.square(linalg.norm(di, 2))
    sum = k2
    return sum

n,d = X_train.shape #60000
ks = [2,4,8,16,32,64]
centers_k = {}

for k in ks:
    print(f"starting k={k}")
    centers = np.zeros((k,d), dtype = float)
    prev_centers = np.zeros((k,d), dtype = float)
    # initialize k centers randomly
    idx = np.random.choice(n, k, replace = False)
    centers = X_train[idx]
    print(centers.shape)
    prev_centers = centers.copy()+1
    print(f"new centers: {centers}")
    centers_saved = {}
    centers_saved = [centers]
    diff = np.ones(40, dtype=float)

    count = 0

    iter = 0
    prev = time.time()
    prev_iter = prev
    prev_loss = np.infty

    run +=1
    display_means(centers)
    this_loss = 0
    loss = []
    min_loss = []
    while np.abs(prev_loss-this_loss)/k > 300 and iter<20:
        prev_centers = centers.copy()
        prev_loss=this_loss
        numbers = np.zeros(k,dtype=int)
        points = np.zeros((k,n,d))
        e_count = 0
        for e in X_train:
            e_count+=1
            min_dist = np.infty
            closest_center = -1
            for i in range(k):
                if dist(e, centers[i]) < min_dist:
                    min_dist = dist(e, centers[i])
                    closest_center = i
            points[closest_center,numbers[closest_center],] = e
            numbers[closest_center] += 1
        for t in range(k):
            if points[t] is not None:
                centers[t] = np.mean(points[t,:numbers[t]], axis=0)
            else:
                print(f"center {t} has no points!")
        #assign points to clusters
        centers_saved.append(centers)
        this_loss = calc_loss()
        this_min_loss = calc_min_loss()
        loss.append(this_loss)
        min_loss.append(this_min_loss)
        display_means(centers)
        
        diff[iter] = np.sum(np.abs(prev_centers - centers))
        print(f"iter: {iter} | loss: {this_loss} | minloss: {this_min_loss} | diff: {diff[iter]}")

        iter += 1
        noww = time.time()
        print(f"iteration time: {noww-prev_iter} seconds")
        prev_iter = noww
    #end of k
    centers_k[k] = centers
#In[]
for  i in range(6):
    print(len(centers_k[2**(i+1)]))
#In[]
train_error_k = {}
test_error_k = {}
ks = [2,4,8,16,32,64]
for k_iter in ks:
    print(k_iter)
    train_error = 0
    for e in X_train:
        min_dist = np.infty
        closest_center = -1
        for i in range(k_iter):
            dis = dist(e, centers_k[k_iter][i])
            if dis < min_dist:
                min_dist = dis
                closest_center = i
        train_error += np.mean(linalg.norm(centers_k[k_iter][closest_center] - e,2))
    train_error /= X_train.shape[0]
    train_error_k[k_iter] = train_error
    print(f'train error for k={k_iter} is {train_error}')

    test_error = 0
    for e in X_test:
        min_dist = np.infty
        closest_center = -1
        for i in range(k_iter):
            dis = dist(e, centers_k[k_iter][i])
            if dis < min_dist:
                min_dist = dis
                closest_center = i
        test_error += np.mean(linalg.norm(centers_k[k_iter][closest_center] - e,2))
    test_error /= X_test.shape[0]
    test_error_k[k_iter] = test_error
    print(f'test error for k={k_iter} is {test_error}')

tr = (train_error_k.items())
k1, error_tr = zip(*tr)
ts = (test_error_k.items())
k2, error_ts = zip(*ts)
#In[]
plt.plot(k1,error_tr,'b.',label= 'Training Error')
plt.plot(k2,error_ts,'r.',label= 'Testing Error')
plt.title('Train and Test Error for k')
plt.xlabel('Iteration')
plt.ylabel('Error ')
plt.legend()
plt.savefig('error_k'+str(error_tr[0])+'.png')
plt.show()



# In[]
plt.plot(loss[:15])
plt.title('Objective Function')
plt.xlabel('Iteration')
plt.ylabel('Objective ')
# plt.legend()
plt.savefig('minloss'+str(loss[0])+'.png')
plt.show()

# In[]
print(np.sum(centers_saved, axis=0))
print(np.sum(np.abs(centers_saved[0]-centers_saved[3])))
# print(points)
    #evaluate centers
# In[]
centers_np = np.array(centers_saved)
print(centers_saved[:2])
x = range(0,k,1)
plt.plot(x,centers_np[0], '.')
plt.plot(x,centers_np[1], '.')

plt.show()



