from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd 

xtrain = pd.read_csv("xtrain.csv")
ytrain = pd.read_csv("ytrain.csv").values.ravel()
xtest = pd.read_csv("xtest.csv")
ytest = pd.read_csv("ytest.csv").values.ravel()
xtrain = xtrain.drop(columns='Roadway_Severity_Score')
xtest = xtest.drop(columns ='Roadway_Severity_Score')

clf = RandomForestClassifier(n_estimators=200,class_weight='balanced_subsample', random_state=42)
clf.fit(xtrain, ytrain)

pred = clf.predict(xtest)
accuracy = accuracy_score(ytest, pred)
print("Accuracy:", accuracy)

print(classification_report(ytest, pred))
matrix = confusion_matrix(ytest, pred) 
acc = accuracy_score(ytest, pred)
prec = precision_score(ytest, pred, average='weighted')
recall = recall_score(ytest, pred, average='weighted')
f1 = f1_score(ytest, pred, average='weighted')

print("Confusion Matrix:")
print(matrix)
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", recall)
print("F1:", f1)



