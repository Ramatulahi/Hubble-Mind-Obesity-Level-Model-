#Model training (Logistics Regression, Random Forest)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model.
    Args:
    X_train (pd.DataFrame): The feature matrix for training.
    y_train (pd.Series): The target variable for training.
    Returns:
    LogisticRegression: The trained Logistic Regression model.
    """
    log_reg_model = LogisticRegression(max_iter=1000)
    log_reg_model.fit(X_train, y_train)
    return log_reg_model

def train_random_forest(X_train, y_train):
    """
    Train Random Forest model.
    Args:
    X_train (pd.DataFrame): The feature matrix for training.
    y_train (pd.Series): The target variable for training.
    Returns:
    RandomForestClassifier: The trained Random Forest model.
    """
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def apply_smote(X_train, y_train):
    """
    Apply SMOTE for class imbalance handling.
    Args:
    X_train (pd.DataFrame): The feature matrix for training.
    y_train (pd.Series): The target variable for training.
    Returns:
    tuple: The resampled feature matrix and target variable.
    """
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled
