from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, cohen_kappa_score

xtrain = pd.read_csv("xtrain.csv")
ytrain = pd.read_csv("ytrain.csv").values.ravel()
xtest = pd.read_csv("xtest.csv")
ytest = pd.read_csv("ytest.csv").values.ravel()

params = {
    'C': [0.001, 0.01, 0.1, 1, 5, 10, 100],
    'penalty': ['l1', 'l2'],
    'max_iter': [3000],
}

clf = LogisticRegression()
lrgrid = GridSearchCV(estimator=clf, param_grid=params)
lrgrid.fit(xtrain, ytrain)

print("Best Parameters:", lrgrid.best_params_)

bestlr = lrgrid.best_estimator_
ypred = bestlr.predict(xtest)

print("Accuracy:", accuracy_score(ytest, ypred))
print("Confusion matrix:", confusion_matrix(ytest, ypred))
print("Classification report:", classification_report(ytest, ypred))
