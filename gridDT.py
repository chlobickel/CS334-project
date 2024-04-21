from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree


xtrain = pd.read_csv("xtrain.csv")
ytrain = pd.read_csv("ytrain.csv")
xtest = pd.read_csv("xtest.csv")
ytest = pd.read_csv("ytest.csv")
"""
param_grid = {
    'max_depth': [None, 5, 10, 15, 20, 25],
    'min_samples_leaf': [3, 5, 7, 10, 15], 
}
clf = DecisionTreeClassifier()
grid_search_dt = GridSearchCV(estimator=clf, param_grid=param_grid)
grid_search_dt.fit(xtrain, ytrain)
print("Best Parameters:", grid_search_dt.best_params_)
"""

################
"""best = grid_search_dt.best_estimator_
y_pred = best.predict(xtest)
accuracy = accuracy_score(ytest, y_pred)
print("Accuracy:", accuracy)"""

"""y_prob = clf.predict(xtest)

#metrics
report = classification_report(ytest, y_prob)
print(report)

cm = confusion_matrix(ytest, y_prob)
print(cm)"""

from sklearn.preprocessing import LabelEncoder
# had to do label encoding because the numerical ones would not plot properly for some reason
classnames = ['(O) No Injury', '(C) Possible Injury / Complaint',
               'B) Suspected Minor/Visible Injury',
                 '(A) Suspected Serious Injury', '(K) Fatal Injury']
ytrain = ytrain.values.ravel()
le = LabelEncoder()
encodedy = le.fit_transform(ytrain)
#this is the optimal params:
clf = DecisionTreeClassifier( max_depth=5, min_samples_leaf=15)
clf.fit(xtrain, encodedy)

li = list(xtrain.columns)

plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=list(xtrain.columns), class_names=classnames, fontsize=10)
plt.show()