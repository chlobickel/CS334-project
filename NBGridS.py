import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the datasets
X_train = pd.read_csv('xtrain.csv')
X_test = pd.read_csv('xtest.csv')
y_train = pd.read_csv('ytrain.csv').squeeze()  # Convert to Series if necessary
y_test = pd.read_csv('ytest.csv').squeeze()

# Define the model pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('classifier', GaussianNB())
])

# Define the parameter grid
param_grid = {
    'classifier__var_smoothing': np.logspace(0,-9, num=100)
}

# Setup the grid search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,               # Number of folds in cross-validation
    verbose=1,          # Higher number gives more output
    n_jobs=-1           # Use all available cores
)

# Perform the grid search
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.3f}".format(grid_search.best_score_))

# Evaluate on the test set
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))
