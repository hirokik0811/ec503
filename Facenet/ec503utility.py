import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Pass in a single row array, numbers only, no names
# For example, [1,2,3,4]
# It can be numpy array or regular Python array
def pca_singleVector(singleVector):
    if singleVector.shape[1] != 512:
        return

    reshaped = np.reshape(singleVector, (8, 64))
    pca = PCA(n_components = 8)
    reduced = pca.fit_transform(reshaped)
    return reduced.flatten()
