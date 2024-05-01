import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

# Load data from CSV files
x_train = pd.read_csv("xtrain.csv")
x_test = pd.read_csv("xtest.csv")
y_train = pd.read_csv("ytrain.csv")
y_test = pd.read_csv("ytest.csv")

# Load data from CSV files
x_train = pd.read_csv("xtrain.csv")
x_test = pd.read_csv("xtest.csv")
y_train = pd.read_csv("ytrain.csv")
y_test = pd.read_csv("ytest.csv")

# Define the Random Under-Sampler and Random Forest Classifier within a pipeline
rus = RandomUnderSampler(random_state=42)
rf = RandomForestClassifier(random_state=42)
pipeline = make_pipeline(rus, rf)

# Parameters grid for Random Forest
param_dist = {
    'randomforestclassifier__n_estimators': [100, 200],
    'randomforestclassifier__max_features': ['sqrt', 'log2'],
    'randomforestclassifier__max_depth': [10, 20, None],
    'randomforestclassifier__min_samples_split': [2, 5],
    'randomforestclassifier__min_samples_leaf': [1, 2]
}

# Setup the randomized search with fewer iterations and cross-validation folds
random_search = RandomizedSearchCV(
    estimator=pipeline, 
    param_distributions=param_dist, 
    n_iter=10, 
    cv=3, 
    verbose=2, 
    random_state=42, 
    n_jobs=-1
)

# Perform the random search
random_search.fit(x_train, y_train.values.ravel())  # Ensure y_train is appropriately shaped for fitting

# Best parameters and best score
print("Best Parameters:", random_search.best_params_)
print("Best Cross-validation Score:", random_search.best_score_)

# Evaluate on the test set
y_pred = random_search.predict(x_test)
print("Test Set Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))