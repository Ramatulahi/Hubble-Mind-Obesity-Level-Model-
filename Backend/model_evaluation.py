# Model evaluation (classification reports, confusion matrices)
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using accuracy and classification report.
    Args:
    model (sklearn.base.BaseEstimator): The trained model.
    X_test (pd.DataFrame): The feature matrix for testing.
    y_test (pd.Series): The target variable for testing.
    Returns:
    confusion_matrix: The confusion matrix of the model's predictions.
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))  # Generate classification report
    cm = confusion_matrix(y_test, y_pred)  # Generate confusion matrix
    return cm

def plot_confusion_matrix(cm, labels, filename='reports/confusion_matrices.png'):
    """
    Plot and save confusion matrix for both models.
    Args:
    cm (ndarray): The confusion matrix to plot.
    labels (list): List of class labels.
    filename (str): File path to save the plot.
    """
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"Confusion matrix plot saved to {filename}")

def plot_feature_importance(model, X, filename='reports/feature_importance.png'):
    """
    Plot and save feature importance for Random Forest model.
    Args:
    model (RandomForestClassifier): The trained Random Forest model.
    X (pd.DataFrame): The feature matrix.
    filename (str): File path to save the plot.
    """
    feature_importance = model.feature_importances_
    sns.barplot(x=feature_importance, y=X.columns)
    plt.title('Feature Importance - Random Forest')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"Feature importance plot saved to {filename}")
