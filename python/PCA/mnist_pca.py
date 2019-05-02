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

logging.basicConfig(filename='./results/mnist_pca.log', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

DATASET = r'C:/Users/kwibu/eclipse-workspace/ec503_project/mnist.mat'

print('Loading ' + DATASET + '...')
f = sio.loadmat(DATASET)
print('Finished loading...')

X_train = np.transpose(np.asarray(csr_matrix.todense(f['Xtr'])))
X_test = np.transpose(np.asarray(csr_matrix.todense(f['Xte'])))[:1000]
y_train = np.transpose(np.asarray(f['ytr']))
y_test = np.transpose(np.asarray(f['yte']))[:1000]
#X_test = X_test[0:1, :]
#y_test = y_test[0:1, :]

print('Shape of training data ([#samples, #dimension]): [%s]' % ','.join(map(str, X_train.shape)))
print('Shape of test data ([#samples, #dimension]): [%s]' % ','.join(map(str, X_test.shape)))
print('Shape of training label ([#samples, #dimension]): [%s]' % ','.join(map(str, y_train.shape)))
print('Shape of test label ([#samples, #dimension]): [%s]' % ','.join(map(str, y_test.shape)))

k_list = [0, 1, 4, 16, 64, 256]
K = np.cov(np.transpose(X_train))
evals, V = np.linalg.eig(K)
X_train_bar = np.matmul(np.transpose(X_train), np.ones((X_train.shape[0], 1)))/X_train.shape[0]
X_train_cent = X_train - np.matmul(np.ones((X_train.shape[0], 1)), np.transpose(X_train_bar))
X_test_bar = np.matmul(np.transpose(X_test), np.ones((X_test.shape[0], 1)))/X_test.shape[0]
X_test_cent = X_test - np.matmul(np.ones((X_test.shape[0], 1)), np.transpose(X_test_bar))

accuracies = []
times = []
for k in k_list:
    topk_idx = np.argsort(evals)[::-1][:k]
    Vk = V[:, topk_idx]
    X_train_red = np.matmul(X_train_cent, Vk)
    X_test_red = np.matmul(X_test_cent, Vk)
    if k == 0:
        X_train_red = X_train
        X_test_red = X_test
    correct = 0
    start_time = time.clock()
    for i in range(X_test.shape[0]):
        print(i)
        nearest = cp.asnumpy(cp.argmin(cp.linalg.norm(cp.array(X_train_red-X_test_red[i]), axis=1)))
        pred = y_train[nearest]
        correct += (pred == y_test[i])
    end_time = time.clock()
    print('Estimation Time: %.10f sec' % (end_time-start_time))
    times.append(end_time-start_time)
    times[-1] /= times[0]
    accuracies.append(correct*100/X_test.shape[0])
    
accuracies = [np.round(a, decimals=2) for a in accuracies]
times = [np.round(t, decimals=10) for t in times]
print('Accuracies: [%s %%]' % ','.join(map(str, accuracies)))
print('Estimation Time / Estimation Time without Clustering: [%s %%]' % ','.join(map(str, times)))
logging.info('Accuracies: [%s %%]' % ','.join(map(str, accuracies)))
logging.info('Estimation Time / Estimation Time without PCA: [%s ]' % ','.join(map(str, times)))

plt.plot(k_list, times)
plt.xlabel('k')
plt.ylabel('Estimation Time / Estimation Time without Clustering)')
plt.show()