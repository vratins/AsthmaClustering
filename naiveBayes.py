import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report)
from sklearn.model_selection import train_test_split, KFold
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

X = dataSet.iloc[:,:-1].to_numpy()    # 152 rows x 2306 cols  
y = dataSet.iloc[:,-1:].to_numpy()     # 152 rows x 1 col

# --------------------------------------------------------------------------------------------------------------------------------------------------
# Split data for test and train

# XTrain, XTest, yTrain, yTest = train_test_split(
#     X, y, test_size = 0.20, random_state = 125, shuffle = True)

# # XTest:    31 rows x 2306 cols
# # XTrain:   121 rows x 2306 cols
# # yTest:    31 rows x 1 col
# # yTrain:   121 rows x 1 col

# # --------------------------------------------------------------------------------------------------------------------------------------------------
# # Gaussian Naive Bayes Model
# guassianModel = GaussianNB()
# guassianModel.fit(XTrain, yTrain)
# yPred = guassianModel.predict(XTest)    # 31 predictions

# # --------------------------------------------------------------------------------------------------------------------------------------------------
# # Naive Bayes Classifier Accuracy 
# accuray = accuracy_score(yPred, yTest)
# f1 = f1_score(yPred, yTest, average="weighted")
# print("Accuracy:", accuray)
# print("F1 Score:", f1)
# print("Naive Bayes score: ",guassianModel.score(XTest, yTest))

# create a Gaussian Naive Bayes model
gaussianModel = GaussianNB()

# split the data into 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = []
f1_scores = []

# loop over each fold
for train_index, test_index in kf.split(X):
    # get the training and testing data for this fold
    XTrain, XTest = X[train_index], X[test_index]
    yTrain, yTest = y[train_index], y[test_index]

    # train the model on the training data
    gaussianModel.fit(XTrain, yTrain)

    # make predictions on the test data
    yPred = gaussianModel.predict(XTest)

    # calculate the accuracy and F1 score for this fold
    accuracy = accuracy_score(yPred, yTest)
    f1 = f1_score(yPred, yTest, average="weighted")

    # append the scores to the lists
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)

# calculate the average scores over all folds
avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
avg_f1 = sum(f1_scores) / len(f1_scores)

print("Average Accuracy:", avg_accuracy)
print("Average F1 Score:", avg_f1)

# --------------------------------------------------------------------------------------------------------------------------------------------------
# Plotting
matrix = confusion_matrix(yTest, yPred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
disp = disp.plot(cmap=plt.cm.Blues,values_format='g')
plt.show()


