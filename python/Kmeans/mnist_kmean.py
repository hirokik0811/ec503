'''
Created on Apr 24, 2019

@author: kwibu
'''
import os, csv, time, logging
import scipy.io as sio
from scipy.sparse import csr_matrix 
import numpy as np
import cupy as cp
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

if os.path.isdir('./results') == False:
    os.mkdir('./results')

logging.basicConfig(filename='./results/mnist_kmean.log', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

DATASET = r'C:/Users/kwibu/eclipse-workspace/ec503_project/mnist.mat'

print('Loading ' + DATASET + '...')
f = sio.loadmat(DATASET)
print('Finished loading...')

X_train = np.transpose(np.asarray(csr_matrix.todense(f['Xtr'])))
X_test = np.transpose(np.asarray(csr_matrix.todense(f['Xte'])))
y_train = np.transpose(np.asarray(f['ytr']))
y_test = np.transpose(np.asarray(f['yte']))
#X_test = X_test[0:1, :]
#y_test = y_test[0:1, :]

print('Shape of training data ([#samples, #dimension]): [%s]' % ','.join(map(str, X_train.shape)))
print('Shape of test data ([#samples, #dimension]): [%s]' % ','.join(map(str, X_test.shape)))
print('Shape of training label ([#samples, #dimension]): [%s]' % ','.join(map(str, y_train.shape)))
print('Shape of test label ([#samples, #dimension]): [%s]' % ','.join(map(str, y_test.shape)))

k_star = np.sqrt(X_train.shape[0]).astype(int)
k_list = [int(k_star/2**lg) for lg in range(-2, int(np.log(k_star)))] + [1]
k_list.reverse()
print('List of k to be tested: [%s]' % ','.join(map(str, k_list)))
logging.info('List of k to be tested: [%s]' % ','.join(map(str, k_list)))
times = []
accuracies = []
for k in k_list:
    print("k: %d" % k)
    
    # Preprocess the data with KMeans
    print("fitting...")
    kmeans = KMeans(n_clusters=k).fit(X_train)
    print("done.")
    clst_label = []
    clst_data = []
    for i in np.unique(kmeans.labels_):
        idx = np.where(kmeans.labels_ == i)[0]
        clst_label.append(y_train[idx])
        clst_data.append(X_train[idx])
        
    # Predict with Nearest Neighbor
    temp_time = []
    correct = 0
    for i, te in enumerate(X_test):
        print("test # %d" % i)
        start_time = time.clock()
        target_clst = cp.asnumpy(cp.argmin(cp.linalg.norm(cp.array(kmeans.cluster_centers_-X_test[i]), axis=1)))
        nearest = cp.asnumpy(cp.argmin(cp.linalg.norm(cp.array(clst_data[target_clst]-X_test[i]), axis=1)))
        pred = clst_label[target_clst][nearest]
        end_time = time.clock()
        temp_time.append(end_time-start_time)
        correct += (y_test[i, 0] == pred)
    times.append(np.mean(temp_time))
    times[-1] = times[-1]/times[0]
    accuracies.append(correct*100/X_test.shape[0])
    print('Accuracy: %.3f %%' % (correct*100/X_test.shape[0]))
    print('Estimation Time: %.10f sec' % np.mean(temp_time))
    
accuracies = [np.round(a, decimals=1) for a in accuracies]
times = [np.round(t, decimals=10) for t in times]
print('Accuracies: [%s %%]' % ','.join(map(str, accuracies)))
print('Estimation Time / Estimation Time without Clustering: [%s %%]' % ','.join(map(str, times)))
logging.info('Accuracies: [%s %%]' % ','.join(map(str, accuracies)))
logging.info('Estimation Time / Estimation Time without Clustering: [%s %%]' % ','.join(map(str, times)))

plt.plot(k_list, times)
plt.xlabel('k')
plt.ylabel('Estimation Time / Estimation Time without Clustering)')
plt.show()