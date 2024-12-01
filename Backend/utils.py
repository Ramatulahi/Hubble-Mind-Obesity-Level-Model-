# Utility functions (saving models, hyperparameter tuning)
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def save_model(model, filename):
    """
    Save the trained model to a file.
    Args:
    model (sklearn.base.BaseEstimator): The trained model.
    filename (str): The filename to save the model.
    """
    joblib.dump(model, filename)

def hyperparameter_tuning(X_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV for Random Forest.
    Args:
    X_train (pd.DataFrame): The feature matrix for training.
    y_train (pd.Series): The target variable for training.
    Returns:
    dict: Best hyperparameters found by GridSearchCV.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_
