import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

dataSet = pd.read_csv("Data/BAL/normdataBAL0715.txt", sep = '\t', nrows= 1000, usecols = range(2, 155))
# print(dataSet)
# dataSet.info()


XTrain, XTest, yTrain, yTest = train_test_split(
    X, y, test_size=0.33, random_state=125
)

guassianModel = GaussianNB()
guassianModel.fit(XTrain, yTrain)


yPred = guassianModel.predict(XTest)

accuray = accuracy_score(yPred, yTest)
f1 = f1_score(yPred, yTest, average="weighted")

print("Accuracy:", accuray)
print("F1 Score:", f1)

