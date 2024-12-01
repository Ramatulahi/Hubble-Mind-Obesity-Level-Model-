# =======================================
# Main Script: Running Everything
# =======================================
from Backend.data_preprocessing import handle_outliers
from Backend.data_preprocessing import load_and_inspect_data, handle_missing_values, encode_categorical_data, feature_engineering, normalize_data
from Backend.model_training import train_logistic_regression, train_random_forest, apply_smote
from Backend.model_evaluation import evaluate_model, plot_confusion_matrix, plot_feature_importance, save_evaluation_report
from Backend.utils import save_model

# Load and preprocess the data
data = load_and_inspect_data('data/obesity_data.csv')
data = handle_missing_values(data)
data = encode_categorical_data(data)
data = feature_engineering(data)

continuous_vars = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
data = handle_outliers(data, continuous_vars)
data = normalize_data(data, continuous_vars)

# Prepare data for model training
X = data.drop(columns=['NObeyesdad'])  # Features
y = data['NObeyesdad']  # Target

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the data
X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

# Train models
log_reg = train_logistic_regression(X_train_resampled, y_train_resampled)
rf_model = train_random_forest(X_train_resampled, y_train_resampled)

# Evaluate models
y_pred_log = log_reg.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
cm_log_reg = evaluate_model(log_reg, X_test, y_test)
cm_rf = evaluate_model(rf_model, X_test, y_test)

# Save evaluation report
save_evaluation_report(y_test, y_pred_log, y_pred_rf)

# Save confusion matrices and feature importance plots
labels = ['Insufficient Weight', 'Normal Weight', 'Overweight Level I', 'Overweight Level II', 'Obesity Type I', 'Obesity Type II', 'Obesity Type III']
save_confusion_matrix_plot(cm_log_reg, labels, filename='reports/confusion_matrices_log_reg.png')
save_confusion_matrix_plot(cm_rf, labels, filename='reports/confusion_matrices_rf.png')
save_feature_importance_plot(rf_model, X, filename='reports/feature_importance.png')

# Save the trained Random Forest model
save_model(rf_model, 'models/random_forest_model.pkl')

print("Project complete and saved!")
