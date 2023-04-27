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

# rows: samples 680
# cols = genes 150
# add another col for labels
# labels can be derived: NC: control, SA: severe, notSA, VSA
# .index.values
dataSet = pd.read_csv("Data/BAL/normdataBAL0715.txt", sep = '\t', nrows= 1000, usecols = range(2, 156)).T
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
print(dataSet)


















# control = dataSet[dataSet.diagnosis == 'C']
# severe = dataSet[dataSet.diagnosis == 'SA']
# notSevere = dataSet[dataSet.diagnosis == 'notSA']
# verySevere = dataSet[dataSet.diagnosis == 'vSA']




# dataSet.diagnosis = [0 if i == "C" in dataSet.diagnosis]
# dataSet.diagnosis = []

# dаtаSet.diаgnоsis = [1 if i== "M" else 0 fоr i in dаtаset.diаgnоsis]

#plt.title("Control vs Benign Tumor")
# plt.xlabel("Radius Mean")
# plt.ylabel("Texture Mean")
# plt.scatter(control.radius_mean, M.texture_mean, color = "red", label = "Malignant", alpha = 0.3)
# plt.scatter(B.radius_mean, B.texture_mean, color = "lime", label = "Benign", alpha = 0.3)
# plt.legend()
# plt.show()





# XTrain, XTest, yTrain, yTest = train_test_split(
#     X, y, test_size=0.33, random_state=125
# )

# guassianModel = GaussianNB()
# guassianModel.fit(XTrain, yTrain)


# yPred = guassianModel.predict(XTest)

# accuray = accuracy_score(yPred, yTest)
# f1 = f1_score(yPred, yTest, average="weighted")

# print("Accuracy:", accuray)
# print("F1 Score:", f1)

