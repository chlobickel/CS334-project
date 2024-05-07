import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Load data from CSV files
x_train = pd.read_csv("xtrain.csv")
x_test = pd.read_csv("xtest.csv")
y_train = pd.read_csv("ytrain.csv")
y_test = pd.read_csv("ytest.csv")

# Define the model
rf = RandomForestClassifier(random_state=42)

# Parameters grid
param_dist = {
    'n_estimators': [100,200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10,20 ,None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Setup the randomized search
random_search = RandomizedSearchCV(
    estimator=rf, 
    param_distributions=param_dist, 
    n_iter=10, 
    cv=3, 
    verbose=2, 
    n_jobs=-1
)

# Perform the random search
random_search.fit(x_train, y_train)

# Predict probabilities for the test set and ensure they are properly shaped
y_probs = random_search.best_estimator_.predict_proba(x_test)[:, 1]  # This ensures we are getting the probabilities for the positive class

# Save the probabilities along with the actual labels for later use in ROC plotting
roc_data = pd.DataFrame({
    'y_true': y_test,
    'y_probs': y_probs  # y_probs is now a 1-dimensional array
})
roc_data.to_csv('roc_data.csv', index=False)

# Best parameters and best score
print("Best Parameters:", random_search.best_params_)
print("Best Cross-validation Score:", random_search.best_score_)

# Evaluate on the test set
y_pred = random_search.predict(x_test)
print("Test Set Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
