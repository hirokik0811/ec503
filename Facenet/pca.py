import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import csv

dataset = pd.read_csv('Client_Character_6000.csv', header = None)

numRows = dataset.shape[0]
numElemsInEachRow = dataset.shape[1]
numElemsInEachRowWithoutName = dataset.shape[1] - 1

pca = PCA(n_components = 64)

#test = dataset.loc[0][1:]
#test = test.values.reshape(-1, 1)
#print(test)

names = []
datasetWithoutNames = np.empty((0, numElemsInEachRowWithoutName))

for i in range(0, numRows):
    names.append(dataset.loc[i][0])
    datasetWithoutNames = np.append(datasetWithoutNames, [dataset.loc[i][1:]], axis = 0)
    #print(currName)

reducedDatasetWithoutNames = pca.fit_transform(datasetWithoutNames)

reducedDataset = {}
for i in range(0, numRows):
    reducedDataset[names[i]] = reducedDatasetWithoutNames[i]

with open('PCA_client_character.csv', 'w', newline='') as csvfile:
    outputwriter = csv.writer(csvfile, delimiter=',')

    for i in range(0, numRows):
        currRowFromNumpyArrayToRegularArray = []
        for dataNumber in reducedDataset[names[i]]:
            currRowFromNumpyArrayToRegularArray.append(dataNumber)
        rowToWrite = [names[i]] + currRowFromNumpyArrayToRegularArray
        outputwriter.writerow(rowToWrite)



'''
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2, random_state = 0)

sc = sklearn.preprocessing.StandardScaler()   
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)

pca = sklearn.decomposition.PCA(n_components = 10)
X_train = pca.fit_transform(X_train) 
X_test = pca.transform(X_test) 

classifier = sklearn.linear_model.LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_guess = classifier.predict(X_test)
'''
