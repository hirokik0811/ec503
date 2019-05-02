'''
Created on Apr 29, 2019

@author: kwibu
'''
import csv
import numpy as np

N_SAMPLES = 5760
N_DIM = 512
    
def kmean_clustered_data(data_file, cluster_file):
    """
    Argument:
    data_file: string, name of a data file
    cluster_file: string, name of a cluster data file, whose each line starts with the cluster number and contains indices belong to the cluster.
    
    Return:
    clst_data: list of numpy arrays, each array has the data belongs to the cluster.
    clst_ids: list of list of strings, each array has the ids belong to the clustsr
    """
    ids = []
    data = np.zeros((N_SAMPLES, N_DIM))
    with open(data_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            ids.append(row[0])
            data[i, :] = np.array([e for e in row[1:]]).astype(float)
    
    clst_data = []
    clst_ids = []
    with open(cluster_file, 'r') as f:
        for clst, lin in enumerate(f):
            lin = lin.replace('\n', '')
            indices = np.array([e for e in lin.split(', ')[:-1]][1:], dtype=int)
            clst_data.append(data[indices])
            clst_ids.append([id for i, id in enumerate(ids) if i in indices])
    return [clst_data, clst_ids]
    
def kmean_centroids(clst_data):
    """
    Argument:
    clst_data: list of numpy arrays, each array has the data belongs to the cluster.
        
    Return:
    cs: numpy array, array of centroids of clusters
    """
    cs = []
    for clst in clst_data:
        cs.append(np.mean(clst, axis=0))
    return np.array(cs)

def kmean_lookup(target, centroids):
    """
    Argument:
    target: 1x512 numpy array, the target data to be classified.
    centroids: k x 512 numpy array, centroids for each cluster
    
    Return:
    clst: int, the cluster target belongs to.
    """
    clst = np.argmin(np.linalg.norm(centroids - target, axis=1))
    return clst
    
clst_data1, clst_ids1 = kmean_clustered_data('Client_Character_6000.csv', r'./datafiles/Client_Character_6000_kmean_1_cluster.txt')
clst_data75, clst_ids75 = kmean_clustered_data('Client_Character_6000.csv', r'./datafiles/Client_Character_6000_kmean_75_cluster.txt')

centroids1 = kmean_centroids(clst_data1)
centroids75 = kmean_centroids(clst_data75)

ids = []
data = np.zeros((N_SAMPLES, N_DIM))
with open('Client_Character_6000.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(reader):
        ids.append(row[0])
        data[i, :] = np.array([e for e in row[1:]]).astype(float)
        
clst1 = kmean_lookup(data[100], centroids1)
clst75 = kmean_lookup(data[100], centroids75)
nearest1 = np.argmin(np.linalg.norm(clst_data1[clst1]-data[100], axis=1))
pred1 = clst_ids1[clst1][nearest1]
nearest75 = np.argmin(np.linalg.norm(clst_data75[clst75]-data[100], axis=1))
pred75 = clst_ids75[clst75][nearest75]
print(pred1)
print(pred75)
