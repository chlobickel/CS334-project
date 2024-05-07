import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression

xtrain = pd.read_csv("xtrain.csv")
ytrain = pd.read_csv("ytrain.csv")
xtest = pd.read_csv("xtest.csv")
ytest = pd.read_csv("ytest.csv")
ytrain = ytrain.values.ravel()
ytest = ytest.values.ravel()

classifiers = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=15),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample', random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7, weights='distance'),
    'Logistic Regression': LogisticRegression(C=0.1, penalty='l2')
}

plt.figure(figsize=(15, 10))

for name, clf in classifiers.items():
    clf.fit(xtrain, ytrain)

    if hasattr(clf, "decision_function"):
        y_score = clf.decision_function(xtest)
    else:
        y_score = clf.predict_proba(xtest)

    ytest_bin = label_binarize(ytest, classes=np.unique(ytrain))
    n_classes = ytest_bin.shape[1]

    plt.figure()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(ytest_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
