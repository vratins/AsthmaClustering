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
import seaborn as sns 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# --------------------------------------------------------------------------------------------------------------------------------------------------
dataSet = pd.read_csv("filtered_data_BAL.csv", usecols = range(1,155)).T
dataSet = dataSet.drop(dataSet.index[[101, 102]]) 
dataSet["Labels"] = 0
# rows: gene names, using 152 rows
# cols: samples, 2307 cols

dataI = dataSet.index.values
for i in range(len(dataSet)):
    if dataI[i].find("NC") != -1:
        dataSet["Labels"][i] = 0
    elif dataI[i].find("_SA_") != -1:
        dataSet["Labels"][i] = 1
    elif dataI[i].find("_notSA_") != -1:
        dataSet["Labels"][i] = 2

X = dataSet.iloc[:,:-1]     # 152 rows x 2306 cols  
y = dataSet.iloc[:,-1:]     # 152 rows x 1 col

# --------------------------------------------------------------------------------------------------------------------------------------------------
# Split data for test and train

XTrain, XTest, yTrain, yTest = train_test_split(
    X, y, test_size = 0.20, random_state = 125, shuffle = True)

# XTest:    31 rows x 2306 cols
# XTrain:   121 rows x 2306 cols
# yTest:    31 rows x 1 col
# yTrain:   121 rows x 1 col

# --------------------------------------------------------------------------------------------------------------------------------------------------
# Gaussian Naive Bayes Model
guassianModel = GaussianNB()
guassianModel.fit(XTrain, yTrain)
yPred = guassianModel.predict(XTest)    # 31 predictions

# --------------------------------------------------------------------------------------------------------------------------------------------------
# Naive Bayes Classifier Accuracy 
accuray = accuracy_score(yPred, yTest)
f1 = f1_score(yPred, yTest, average="weighted")
print("Accuracy:", accuray)
print("F1 Score:", f1)
print("Naive Bayes score: ",guassianModel.score(XTest, yTest))

# --------------------------------------------------------------------------------------------------------------------------------------------------
# Plotting
matrix = confusion_matrix(yTest, yPred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
disp = disp.plot(cmap=plt.cm.Blues,values_format='g')
plt.show()


