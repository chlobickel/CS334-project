from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.preprocessing import StandardScaler

xtrain = pd.read_csv("xtrain.csv")
ytrain = pd.read_csv("ytrain.csv").values.ravel()
xtest = pd.read_csv("xtest.csv")
ytest = pd.read_csv("ytest.csv").values.ravel()

# scaling for knn only
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

param_grid = {
    'n_neighbors': [2, 3, 5, 7, 10, 12, 15, 20]
}
clf = KNeighborsClassifier()
search = GridSearchCV(estimator=clf, param_grid=param_grid)
search.fit(xtrain, ytrain)

bestKNN = search.best_estimator_
pred = bestKNN.predict(xtest) 
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
