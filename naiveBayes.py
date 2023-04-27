import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np

# rows: samples 680
# cols = genes 150
# add another col for labels
# labels can be derived: NC: control, SA: severe, notSA, VSA
# .index.values

# 1st col usesless, remove the first col

#dataSet = pd.read_csv("Data/BAL/normdataBAL0715.txt", sep = '\t', nrows= 1000, usecols = range(2, 156)).T
dataSet = pd.read_csv("filtered_data_BAL.csv", usecols = range(1,155)).T
dataSet["Labels"] = 0


dataI = dataSet.index.values
for i in range(len(dataSet)):
    if dataI[i].find("NC") != -1:
        dataSet["Labels"][i] = 0
    elif dataI[i].find("_SA_") != -1:
        dataSet["Labels"][i] = 1
    elif dataI[i].find("_notSA_") != -1:
        dataSet["Labels"][i] = 2
    elif dataI[i].find("_VSA_") != -1:
        dataSet["Labels"][i] = 3
#print(dataSet)

# shuffle the data  --> shuffle rows 
#shuffledDataSet = dataSet.sample(frac = 1)
X = dataSet.iloc[:,:-1].to_numpy() 
y = dataSet.iloc[:,-1:].to_numpy()

XTrain, XTest, yTrain, yTest = train_test_split(
    X, y, test_size = 0.20, random_state =125, shuffle = True)

guassianModel = GaussianNB()
guassianModel.fit(XTrain, yTrain)


yPred = guassianModel.predict(XTest)

accuray = accuracy_score(yPred, yTest)
f1 = f1_score(yPred, yTest, average="weighted")

print("Accuracy:", accuray)
print("F1 Score:", f1)

