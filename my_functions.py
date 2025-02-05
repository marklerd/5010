# Python helper function file
import os
import pandas as pd
import numpy as np
import importlib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, cross_val_predict, permutation_test_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score,roc_curve, confusion_matrix, f1_score, balanced_accuracy_score, make_scorer, auc, precision_recall_curve, average_precision_score, precision_score, recall_score
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.utils import resample, compute_class_weight
from sklearn.manifold import TSNE
from sklearn.feature_selection import RFE
from my_functions import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
import scipy.stats
from scipy.stats import ttest_ind, ranksums, mannwhitneyu
from statsmodels.stats import multitest
import statsmodels.stats.multitest as multi
from sklearn.preprocessing import StandardScaler

# ignore all warnings
import warnings
from sklearn.exceptions import ConvergenceWarning

# Down sampling function
def downsample_data(X, y, random_state=0):
    """
    Downsamples the majority class to match the size of the minority class.

    Parameters:
    - X: Feature set (DataFrame or array-like)
    - y: Target set (array-like)
    - random_state: Seed for reproducibility

    Returns:
    - X_downsampled: Downsampled feature set
    - y_downsampled: Downsampled target set
    """
    # Combine X and y into a single dataframe for easier manipulation
    df = pd.DataFrame(X)
    df['target'] = y
    
    # Separate the majority and minority classes
    majority_class = df[df['target'] == 0]
    minority_class = df[df['target'] == 1]
    
    # Downsample the majority class to the size of the minority class
    majority_class_downsampled = resample(majority_class, 
                                          replace=False,     # sample without replacement
                                          n_samples=len(minority_class),  # match the number of minority class samples
                                          random_state=random_state)    # for reproducibility
    
    # Combine the downsampled majority class with the minority class
    df_downsampled = pd.concat([majority_class_downsampled, minority_class])
    
    # Separate features and target again
    X_downsampled = df_downsampled.drop('target', axis=1)
    y_downsampled = df_downsampled['target']
    
    return X_downsampled, y_downsampled

# Plot PCA functino
def plot_pca(X_train, y_train, random_state=0, figsize=(8, 6)):
    """
    Applies PCA on the training data and plots the results.

    Parameters:
    - X_train: Training feature set (array-like or DataFrame)
    - y_train: Training target set (array-like)
    - random_state: Seed for reproducibility (default is 0)
    - figsize: Size of the plot (default is (8, 6))

    Returns:
    - X_pca: PCA transformed feature set
    """
    # Apply PCA
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_train)

    # Plot the PCA results
    plt.figure(figsize=figsize)
    plt.scatter(X_pca[y_train == 0, 0], X_pca[y_train == 0, 1], 
                color='blue', label='RNA_seq', alpha=0.5)
    plt.scatter(X_pca[y_train == 1, 0], X_pca[y_train == 1, 1], 
                color='red', label='Microarray', alpha=0.5)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('PCA of Training Data Pre Normalisation')
    plt.legend()
    plt.grid()
    plt.show()
    
    return X_pca

# Train ml model with gridsearchCV with AUR_ROC score
def train_model_with_gridsearch_roc(model, param_grid, X_train, y_train, X_test, y_test, scoring='roc_auc', 
                                 cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)):
    """
    Trains a model using GridSearchCV with specified hyperparameters, then evaluates it on both
    the training and test data. Ensures reproducibility by setting random_state.

    Parameters:
    - model: The machine learning model (e.g., RandomForestClassifier(), SVC()).
    - param_grid: Dictionary of hyperparameters for the model.
    - X_train: Training feature set.
    - y_train: Training target set.
    - X_test: Test feature set.
    - y_test: Test target set.
    - scoring: Metric for evaluating the model in GridSearchCV (default is 'roc_auc', can also use 'f1_weighted').
    - cv: Cross-validation splitting strategy (default is StratifiedKFold with n_splits=5, random_state=0).
    
    Returns:
    - best_model: The model with the best parameters found through GridSearchCV.
    - best_params: The best hyperparameters.
    - accuracy_train: Accuracy score on the training set.
    - accuracy_test: Accuracy score on the test set.
    - auc_roc_train: AUC-ROC score on the training set (if applicable).
    - auc_roc_test: AUC-ROC score on the test set (if applicable).
    - report_test: Classification report on the test set.
    - conf_matrix: Confusion matrix for the test set.
    """
    
    # Ensure the model has a random state set, if applicable
    if hasattr(model, 'random_state'):
        model.random_state = 0

    # Initialize GridSearchCV with the specified model and hyperparameter grid
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               cv=cv, scoring=scoring, n_jobs=-1, verbose=1)

    # Fit the model on the training data
    grid_search.fit(X_train, y_train)

    # Get the best model and its hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    ### Evaluate on training set
    y_train_pred = best_model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {accuracy_train}")
    
    # Calculate AUC-ROC for the train set if binary classification and roc_auc is selected
    if scoring == 'roc_auc' and len(set(y_train)) == 2:  # Binary classification
        y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]  # Probabilities for the positive class
        auc_roc_train = roc_auc_score(y_train, y_train_pred_proba)
        print(f"Train AUC-ROC: {auc_roc_train}")
    else:
        auc_roc_test = None

    ### Evaluate on test set
    y_test_pred = best_model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)

    # Calculate classification report
    report_test = classification_report(y_test, y_test_pred)

    # Calculate AUC-ROC for the test set if binary classification and roc_auc is selected
    if scoring == 'roc_auc' and len(set(y_train)) == 2:  # Binary classification
        y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
        auc_roc_test = roc_auc_score(y_test, y_test_pred_proba)
        print(f"Test AUC-ROC: {auc_roc_test}")
    else:
        auc_roc_test = None

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Plotting Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'], 
                yticklabels=['True Negative', 'True Positive'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    # Print the results
    print(f"Best Hyperparameters: {best_params}")
    print(f"Test Accuracy: {accuracy_test}")
    print("Test Classification Report:")
    print(report_test)

    # Return the best model, its parameters, accuracy on train and test sets, AUC-ROC scores (if applicable), 
    # report, and confusion matrix
    return best_model, best_params, accuracy_train, accuracy_test, auc_roc_train, auc_roc_test, report_test, conf_matrix

# train ML model with gridsearchCV using f1_weighted score
def train_model_with_gridsearch_f1(model, param_grid, X_train, y_train, X_test, y_test, scoring='f1_weighted', 
                                 cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)):
    """
    Trains a model using GridSearchCV with specified hyperparameters, then evaluates it on both
    the training and test data. Ensures reproducibility by setting random_state.

    Parameters:
    - model: The machine learning model (e.g., RandomForestClassifier(), SVC()).
    - param_grid: Dictionary of hyperparameters for the model.
    - X_train: Training feature set.
    - y_train: Training target set.
    - X_test: Test feature set.
    - y_test: Test target set.
    - scoring: Metric for evaluating the model in GridSearchCV (default is 'f1_weighted').
    - cv: Cross-validation splitting strategy (default is StratifiedKFold with n_splits=5, random_state=0).
    
    Returns:
    - best_model: The model with the best parameters found through GridSearchCV.
    - best_params: The best hyperparameters.
    - accuracy_train: Accuracy score on the training set.
    - accuracy_test: Accuracy score on the test set.
    - f1_train: F1-Weighted score on the training set.
    - f1_test: F1-Weighted score on the test set.
    - report_test: Classification report on the test set.
    - conf_matrix: Confusion matrix for the test set.
    """
    
    # Ensure the model has a random state set, if applicable
    if hasattr(model, 'random_state'):
        model.random_state = 0

    # Initialize GridSearchCV with the specified model and hyperparameter grid
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               cv=cv, scoring=scoring, n_jobs=-1, verbose=1)

    # Fit the model on the training data
    grid_search.fit(X_train, y_train)

    # Get the best model and its hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    ### Evaluate on training set
    y_train_pred = best_model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    f1_train = f1_score(y_train, y_train_pred, average='weighted')
    print(f"Training Accuracy: {accuracy_train}")
    print(f"Training F1-Weighted Score: {f1_train}")

    ### Evaluate on test set
    y_test_pred = best_model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred, average='weighted')
    print(f"Test Accuracy: {accuracy_test}")
    print(f"Test F1-Weighted Score: {f1_test}")

    # Calculate classification report
    report_test = classification_report(y_test, y_test_pred)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Plotting Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'], 
                yticklabels=['True Negative', 'True Positive'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    # Print the results
    print(f"Best Hyperparameters: {best_params}")
    print(f"Test Classification Report:")
    print(report_test)

    # Return the best model, its parameters, accuracy on train and test sets, F1-Weighted scores, 
    # report, and confusion matrix
    return best_model, best_params, accuracy_train, accuracy_test, f1_train, f1_test, report_test, conf_matrix

# Train ML model with PCA components, GridsearchCV with AUC_ROC score
def train_model_with_gridsearch_PCA_roc(model, param_grid, X_train, y_train, X_test, y_test, scoring='roc_auc', cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0), n_components_pca=None):
    """
    Trains a model using GridSearchCV with specified hyperparameters and optional PCA, 
    then evaluates it on both the training and test data.

    Parameters:
    - model: The machine learning model (e.g., RandomForestClassifier(), SVC()).
    - param_grid: Dictionary of hyperparameters for the model.
    - X_train: Training feature set.
    - y_train: Training target set.
    - X_test: Test feature set.
    - y_test: Test target set.
    - scoring: Metric for evaluating the model in GridSearchCV (default is 'roc_auc').
    - cv: Cross-validation splitting strategy (default is StratifiedKFold with n_splits=5, random_state=0).
    - n_components_pca: Number of components for PCA (None means no PCA is applied).
    
    Returns:
    - best_model: The model with the best parameters found through GridSearchCV.
    - best_params: The best hyperparameters.
    - accuracy_train: Accuracy score on the training set.
    - accuracy_test: Accuracy score on the test set.
    - auc_roc_train: AUC-ROC score on the training set (if applicable).
    - auc_roc_test: AUC-ROC score on the test set (if applicable).
    - report_test: Classification report on the test set.
    """
    
    # Ensure the model has a random state set, if applicable
    if hasattr(model, 'random_state'):
        model.random_state = 0

    # Build a pipeline with optional PCA
    steps = []
    if n_components_pca is not None:
        steps.append(('pca', PCA(n_components=n_components_pca)))  # Add PCA as the first step
    
    steps.append(('model', model))  # Add the model as the second step
    pipeline = Pipeline(steps)  # Create the pipeline
    
    # Initialize GridSearchCV with the pipeline and hyperparameter grid
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, 
                               cv=cv, scoring=scoring, n_jobs=-1, verbose=1)

    # Fit the model on the training data
    grid_search.fit(X_train, y_train)

    # Get the best model and its hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    ### Evaluate on training set
    y_train_pred = best_model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {accuracy_train}")

    # Calculate AUC-ROC for the training set if binary classification and roc_auc is selected
    if scoring == 'roc_auc' and len(set(y_train)) == 2:  # Binary classification
        y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]
        auc_roc_train = roc_auc_score(y_train, y_train_pred_proba)
        print(f"Training AUC-ROC: {auc_roc_train}")
    else:
        auc_roc_train = None

    ### Evaluate on test set
    y_test_pred = best_model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)

    # Calculate classification report
    report_test = classification_report(y_test, y_test_pred)

    # Calculate AUC-ROC for the test set if binary classification and roc_auc is selected
    if scoring == 'roc_auc' and len(set(y_train)) == 2:  # Binary classification
        y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
        auc_roc_test = roc_auc_score(y_test, y_test_pred_proba)
        print(f"Test AUC-ROC: {auc_roc_test}")
    else:
        auc_roc_test = None

    # Print the results
    print(f"Best Hyperparameters: {best_params}")
    print(f"Test Accuracy: {accuracy_test}")
    print("Test Classification Report:")
    print(report_test)

    # Return the best model, its parameters, accuracy on train and test sets, AUC-ROC scores (if applicable), and report
    return best_model, best_params, accuracy_train, accuracy_test, auc_roc_train, auc_roc_test, report_test

# ML model training using PCA component, GridsearchCV with F1_weighted score
def train_model_with_gridsearch_PCA_f1(model, param_grid, X_train, y_train, X_test, y_test, scoring='f1_weighted', cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0), n_components_pca=None):
    """
    Trains a model using GridSearchCV with specified hyperparameters and optional PCA, 
    then evaluates it on both the training and test data.

    Parameters:
    - model: The machine learning model (e.g., RandomForestClassifier(), SVC()).
    - param_grid: Dictionary of hyperparameters for the model.
    - X_train: Training feature set.
    - y_train: Training target set.
    - X_test: Test feature set.
    - y_test: Test target set.
    - scoring: Metric for evaluating the model in GridSearchCV (default is 'f1_weighted').
    - cv: Cross-validation splitting strategy (default is StratifiedKFold with n_splits=5, random_state=0).
    - n_components_pca: Number of components for PCA (None means no PCA is applied).
    
    Returns:
    - best_model: The model with the best parameters found through GridSearchCV.
    - best_params: The best hyperparameters.
    - accuracy_train: Accuracy score on the training set.
    - accuracy_test: Accuracy score on the test set.
    - f1_train: Weighted F1 score on the training set.
    - f1_test: Weighted F1 score on the test set.
    - report_test: Classification report on the test set.
    """
    
    # Ensure the model has a random state set, if applicable
    if hasattr(model, 'random_state'):
        model.random_state = 0

    # Build a pipeline with optional PCA
    steps = []
    if n_components_pca is not None:
        steps.append(('pca', PCA(n_components=n_components_pca)))  # Add PCA as the first step
    
    steps.append(('model', model))  # Add the model as the second step
    pipeline = Pipeline(steps)  # Create the pipeline
    
    # Initialize GridSearchCV with the pipeline and hyperparameter grid
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, 
                               cv=cv, scoring=scoring, n_jobs=-1, verbose=1)

    # Fit the model on the training data
    grid_search.fit(X_train, y_train)

    # Get the best model and its hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    ### Evaluate on training set
    y_train_pred = best_model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    f1_train = f1_score(y_train, y_train_pred, average='weighted')
    print(f"Training Accuracy: {accuracy_train}")
    print(f"Training F1-Weighted: {f1_train}")

    ### Evaluate on test set
    y_test_pred = best_model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred, average='weighted')
    report_test = classification_report(y_test, y_test_pred)

    print(f"Test Accuracy: {accuracy_test}")
    print(f"Test F1-Weighted: {f1_test}")
    print("Test Classification Report:")
    print(report_test)

    # Return the best model, its parameters, accuracy on train and test sets, F1 scores, and report
    return best_model, best_params, accuracy_train, accuracy_test, f1_train, f1_test, report_test

# train ML model using LDA components, GridsearchCV with AUC-ROC score
def train_model_with_gridsearch_LDA(model, param_grid, X_train, y_train, X_test, y_test, scoring='roc_auc', cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0), n_components_lda=None):
    """
    Trains a model using GridSearchCV with specified hyperparameters and LDA for dimensionality reduction,
    then evaluates it on both the training and test data.

    Parameters:
    - model: The machine learning model (e.g., RandomForestClassifier(), SVC()).
    - param_grid: Dictionary of hyperparameters for the model.
    - X_train: Training feature set.
    - y_train: Training target set.
    - X_test: Test feature set.
    - y_test: Test target set.
    - scoring: Metric for evaluating the model in GridSearchCV (default is 'roc_auc').
    - cv: Cross-validation splitting strategy (default is StratifiedKFold with n_splits=5, random_state=0).
    - n_components_lda: Number of components to reduce to using LDA.
    
    Returns:
    - Same as the original function.
    """
    
    # Set LDA with specified number of components
    lda = LDA(n_components=n_components_lda)

    # Create a pipeline with LDA and the classifier model
    pipeline = Pipeline(steps=[
        ('lda', lda),
        ('model', model)
    ])

    # Initialize GridSearchCV with the pipeline
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, 
                               cv=cv, scoring=scoring, n_jobs=-1, verbose=1)

    # Fit the model on the training data with LDA
    grid_search.fit(X_train, y_train)

    # Get the best model and its hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Evaluate on training set
    y_train_pred = best_model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {accuracy_train}")

    # Calculate AUC-ROC for training set if binary classification and roc_auc is selected
    if scoring == 'roc_auc' and len(set(y_train)) == 2:  # Binary classification
        y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]
        auc_roc_train = roc_auc_score(y_train, y_train_pred_proba)
        print(f"Training AUC-ROC: {auc_roc_train}")
    else:
        auc_roc_train = None

    # Evaluate on test set
    y_test_pred = best_model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)

    # Calculate classification report
    report_test = classification_report(y_test, y_test_pred)

    # Calculate AUC-ROC for test set if binary classification and roc_auc is selected
    if scoring == 'roc_auc' and len(set(y_train)) == 2:  # Binary classification
        y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
        auc_roc_test = roc_auc_score(y_test, y_test_pred_proba)
        print(f"Test AUC-ROC: {auc_roc_test}")
    else:
        auc_roc_test = None

    # Print the results
    print(f"Best Hyperparameters: {best_params}")
    print(f"Test Accuracy: {accuracy_test}")
    print("Test Classification Report:")
    print(report_test)

    # Return the best model, its parameters, accuracy on train and test sets, AUC-ROC scores (if applicable), and report
    return best_model, best_params, accuracy_train, accuracy_test, auc_roc_train, auc_roc_test, report_test

# Permutation importance score and feature ranking
def compute_permutation_importance(model, X_train, y_train, threshold=0, scoring='f1_weighted', n_repeats=5, random_state=0, n_jobs=-1, output_csv="important_features.csv"):
    """
    Computes permutation importance for the given model and training data, then extracts the important features 
    based on the importance mean threshold.

    Parameters:
    - model: Trained model to evaluate permutation importance.
    - X_train: Training feature set (DataFrame).
    - y_train: Training target set.
    - threshold: Threshold for feature importance mean to filter important features (default is 0).
    - scoring: Metric to evaluate permutation importance (default is 'f1_weighted').
    - n_repeats: Number of times to permute a feature (default is 5).
    - random_state: Random state for reproducibility.
    - n_jobs: Number of jobs to run in parallel (default is -1 for using all cores).
    - output_csv: Path to save the important features (default is "important_features.csv").

    Returns:
    - important_features_list: List of selected important features.
    - importance_df: DataFrame of all features with their importance means and standard deviations.
    """

    # Compute permutation importance on the training data
    perm_importance = permutation_importance(
        model, X_train, y_train, scoring=scoring, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs
    )

    # Create a DataFrame to display feature importance
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance Mean': perm_importance.importances_mean,
        'Importance Std': perm_importance.importances_std
    })

    # Sort features by importance mean
    importance_df = importance_df.sort_values(by='Importance Mean', ascending=False)

    # Display the feature importance
    print(importance_df)

    # Filter features with importance mean greater than the threshold
    important_features = importance_df[importance_df['Importance Mean'] > threshold]
    print(f"Features with importance mean > {threshold}: \n{important_features}")

    # Extract selected features
    selected_features = important_features['Feature'].values
    print(f"Selected features: {selected_features}")

    # Save the important features to a CSV file
    important_features.to_csv(output_csv, index=False)

    # Return the list of important features and the DataFrame
    important_features_list = important_features['Feature'].tolist()
    return important_features_list, importance_df

# Define your parameter grid for XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

rf_param_grid = {
    'bootstrap': [True],  # Keep bootstrap fixed to True
    'max_depth': [3, 4],  # Keep the depth options
    'min_samples_leaf': [4, 5],  # Reduce range for leaf sizes
    'min_samples_split': [10, 20, 30],  # Reduce options for split points
    'n_estimators': [100, 150],  # Narrow down the number of trees
    'max_features': ['sqrt', 'log2'],  # Reduce feature selection methods
    'max_samples': [0.5, 0.6, 0.7]  # Narrow down sample fractions
}

svm_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 0.001, 0.01, 0.1, 1, 10],
    'degree': [2, 3, 4, 5]
}


lr_param_grid = {
    'penalty': ['l1', 'l2'],  # L1 for Lasso, L2 for Ridge
    'C': [0.01, 0.1, 1.0, 10.0],  # Regularization strengths
    'solver': ['saga'],  # 'saga' supports both L1 and L2
    'max_iter': [1000, 10000],  # Iterations for convergence
    'tol': [1e-4, 1e-3],  # Tolerance for stopping criteria
}


ridge_param_grid = {
    'C': [0.1, 1.0, 10.0],  # Fewer C values
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],  # Fewer solvers
    'max_iter': [1000, 10000],  # Single max_iter value
    'tol': [1e-4, 1e-3],
}


lasso_param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'solver': ['liblinear', 'saga'],
    'max_iter': [1000, 10000],
    'tol': [1e-4, 1e-3],
}


knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],    # Number of neighbors to use
    'weights': ['uniform', 'distance'], # Weight function
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], # Algorithm used to compute nearest neighbors
    'leaf_size': [10, 30, 50],          # Leaf size affects speed
    'p': [1, 2]                         # Power parameter for Minkowski metric (1=Manhattan, 2=Euclidean)
}


mlp_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],  # Different layer architectures
    'activation': ['relu', 'tanh', 'logistic'],                 # Activation functions
    'solver': ['adam', 'lbfgs', 'sgd'],                         # Optimization algorithms
    'alpha': [0.0001, 0.001, 0.01, 0.1],                        # Regularization term
    'learning_rate': ['constant', 'adaptive'],                  # Learning rate schedule
}

def filter_highly_correlated_features(X_train, threshold=0.8):
    """
    Filters out highly correlated features from the input DataFrame and plots heatmaps of the original 
    and filtered correlation matrices.

    Parameters:
        X_train (pd.DataFrame): The input DataFrame containing features.
        threshold (float): The correlation threshold above which features will be considered redundant.

    Returns:
        pd.DataFrame: A DataFrame with highly correlated features removed.
    """
    
    # Step 1: Calculate the correlation matrix
    correlation_matrix = X_train.corr()

    # Step 2: Identify highly correlated features
    def get_redundant_pairs(corr_matrix, threshold):
        # Get the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with a correlation greater than the threshold
        redundant_pairs = [(column, index) for column in upper.columns for index in upper.index 
                           if abs(upper[column][index]) > threshold]

        return redundant_pairs

    redundant_pairs = get_redundant_pairs(correlation_matrix, threshold)

    # Create a set of features to drop
    features_to_drop = set([index for column, index in redundant_pairs])

    # Step 3: Filter out highly correlated features
    X_train_filtered_corr = X_train.drop(columns=features_to_drop)

    # Create heatmap for the original correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={"shrink": .8})
    plt.title("Correlation Heatmap - Before Filtering")
    plt.show()

    # Create heatmap for the filtered correlation matrix
    filtered_correlation_matrix = X_train_filtered_corr.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(filtered_correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={"shrink": .8})
    plt.title("Correlation Heatmap - After Filtering")
    plt.show()
    
    return X_train_filtered_corr


def recursive_feature_addition_with_gridsearch_plot(model, param_grid, feature_importance_list, X_train, y_train, X_test, y_test):
    # Initialize lists to store results
    selected_features = []
    f1_train_scores = []  # For training data (cross-validated)
    f1_test_scores = []   # For testing data (cross-validated)
    
    # Variables to track the best F1 score and corresponding feature set
    best_f1_score = -np.inf  # Initialize to negative infinity
    best_num_features = 0
    best_feature_set = []
    best_params = None

    # Set the random seed for reproducibility
    np.random.seed(0)
    
    # Start Recursive Feature Addition
    for feature in feature_importance_list[:500]:
        selected_features.append(feature)
        
        # Perform hyperparameter tuning with GridSearchCV (using all available cores)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)  # Multicore processing and set random state
        grid_search.fit(X_train[selected_features], y_train)
        
        # Get the best model from the grid search
        best_model = grid_search.best_estimator_
        
        # Perform 5-fold cross-validation with the selected features (on training set)
        f1_cv_scores_train = cross_val_score(best_model, X_train[selected_features], y_train, cv=5, scoring='f1_weighted', n_jobs=-1)  # Multicore processing and set random state
        
        # Calculate the average F1 score across the 10 folds for training data
        f1_train_avg = np.mean(f1_cv_scores_train)
        f1_train_scores.append(f1_train_avg)
        
        # Perform cross-validated predictions on the entire test dataset (using all cores)
        y_pred_cv = cross_val_predict(best_model, X_test[selected_features], y_test, cv=5, n_jobs=-1)  # Multicore processing and set random state
        
        # Calculate F1 score on cross-validated predictions (test evaluation)
        f1_test_avg = f1_score(y_test, y_pred_cv, average='weighted')
        f1_test_scores.append(f1_test_avg)
        
        # Update the best F1 score and feature set if the current one is better
        if f1_test_avg > best_f1_score:
            best_f1_score = f1_test_avg
            best_num_features = len(selected_features)
            best_feature_set = list(selected_features)  # Copy of the current best feature set
            best_params = grid_search.best_params_

    # Plot F1 score vs. number of features for both training and testing sets
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(f1_train_scores) + 1), f1_train_scores, marker='o', linestyle='-', color='b', label='Train F1-weighted Score', zorder=1)
    plt.plot(range(1, len(f1_test_scores) + 1), f1_test_scores, marker='o', linestyle='-', color='g', label='Test F1-weighted Score', zorder=2)
    
    # Annotate the point with the highest test F1 score
    plt.scatter(best_num_features, best_f1_score, color='red', label=f'Best Test F1: {best_f1_score:.4f}', zorder=3)
    plt.text(best_num_features, best_f1_score, f'  Best Score ({best_num_features} features)', 
             verticalalignment='bottom', horizontalalignment='right', zorder=3)
    
    plt.xlabel('Number of Features')
    plt.ylabel('F1-weighted Score')
    plt.title('F1-weighted Score vs. Number of Features (Train and Test)')
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"Best Test F1 Score: {best_f1_score:.4f}")
    print(f"Number of Features: {best_num_features}")
    print(f"Selected Features: {best_feature_set}")

    return best_params, f1_train_scores, f1_test_scores, best_feature_set


from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import permutation_test_score, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

def permutation_test_with_model(
    model, X_train, y_train, param_grid=None, scoring='balanced_accuracy', 
    cv=5, n_permutations=1000, random_state=0
):
    """
    Function to evaluate model performance with a permutation test, using Balanced Accuracy
    and optional grid search optimization.
    
    Parameters:
    - model: The model to evaluate (SVM, Logistic Regression, etc.).
    - X_train: Training features.
    - y_train: Training labels.
    - param_grid: Parameter grid for GridSearchCV (default is None).
    - scoring: The scoring metric to use ('balanced_accuracy').
    - cv: Number of cross-validation folds (default is 5).
    - n_permutations: Number of label permutations to perform (default is 1000).
    - random_state: Random seed for reproducibility (default is 0).
    
    Returns:
    - Original score, permuted scores, and p-value from the permutation test.
    """
    # Check scoring metric
    if scoring == 'balanced_accuracy':
        scorer = make_scorer(balanced_accuracy_score)
    else:
        raise ValueError("Invalid scoring metric. Only 'balanced_accuracy' is supported in this function.")
    
    # Optimize model using GridSearchCV if param_grid is provided
    if param_grid is not None:
        grid_search = GridSearchCV(model, param_grid, scoring=scorer, cv=cv, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best parameters from GridSearchCV: {grid_search.best_params_}")
    else:
        best_model = model
    
    # Perform permutation test
    score, perm_scores, p_value = permutation_test_score(
        best_model, X_train, y_train, scoring=scorer, cv=cv, 
        n_permutations=n_permutations, random_state=random_state, n_jobs=-1
    )

    # Print results
    print(f"Model score (original): {score:.4f}")
    print(f"Mean permutation score: {np.mean(perm_scores):.4f}")
    print(f"Empirical p-value: {p_value:.4f}")
    
    # Plot permutation scores
    plt.figure(figsize=(10, 6))
    plt.hist(perm_scores, bins=20, edgecolor='k', alpha=0.7)
    
    # Vertical lines for mean score on permuted data, max score on permuted data, and mean score on original data
    plt.axvline(np.mean(perm_scores), color='blue', linestyle='--', label=f'Mean Permutation Score: {np.mean(perm_scores):.4f}')
    plt.axvline(np.max(perm_scores), color='orange', linestyle='--', label=f'Max Permutation Score: {np.max(perm_scores):.4f}')
    plt.axvline(score, color='red', linestyle='--', label=f'Original Score: {score:.4f}')
    
    plt.title(f'Permutation Test (p-value = {p_value:.4f})')
    plt.xlabel('Balanced accuracy')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid()
    plt.show()
    
    return score, perm_scores, p_value

def train_model_with_gridsearch_metrics(model, 
                                        param_grid, 
                                        model_name, 
                                        lr_params, 
                                        metrics_df, 
                                        X_train, 
                                        y_train, 
                                        X_test, 
                                        y_test,
                                        dataset,
                                        scoring,
                                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)):
    """
    Trains a model using GridSearchCV with specified hyperparameters, then evaluates it on both
    the training and test data. Ensures reproducibility by setting random_state.

    Parameters:
    - model: The machine learning model (e.g., RandomForestClassifier(), SVC()).
    - param_grid: Dictionary of hyperparameters for the model.
    - model_name: Name of the model (for plotting purposes).
    - X_train: Training feature set.
    - y_train: Training target set.
    - X_test: Test feature set.
    - y_test: Test target set.
    - scoring: Metric for evaluating the model in GridSearchCV (default is 'f1_weighted').
    - cv: Cross-validation splitting strategy (default is StratifiedKFold with n_splits=10, random_state=0).
    
    Returns:
    - metrics: List containing the following metrics:
        - accuracy_train
        - accuracy_test
        - f1_train
        - f1_test
        - balanced_accuracy
        - roc_auc
        - precision
        - recall
        - specificity
        - false_positive_rate
    - best_model: The model with the best hyperparameters found through GridSearchCV.
    - metrics_df: Updated dataframe containing the metrics for all models.
    """
    
    # Ensure the model has a random state set, if applicable
    if hasattr(model, 'random_state'):
        model.random_state = 0
        
    # If model_name = "Dummy Classifier" --> skip gridsearch
    if model_name == "Dummy Classifier":
        best_model = DummyClassifier(strategy="most_frequent", random_state=0).fit(X_train, y_train)
    elif model_name == "Logistic Regression" and dataset != "dge":
        # Step 1: Create a new SVC model using the best parameters
        best_model = LogisticRegression(**lr_params, random_state=0,penalty='l2').fit(X_train, y_train)
    else:
        # Initialize GridSearchCV with the specified model and hyperparameter grid
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                                   cv=cv, scoring=scoring, n_jobs=-1, verbose=1)

        # Fit the model on the training data
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_

    ### Evaluate on training set
    y_train_pred = best_model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    f1_train = f1_score(y_train, y_train_pred, average='weighted')

    ### Evaluate on test set with 5-fold cross-validation (average metrics)
    cv_folds = 5  # Set the number of cross-validation folds
    accuracy_test = np.mean(cross_val_score(best_model, X_test, y_test, cv=cv_folds, scoring='accuracy'))
    f1_test = np.mean(cross_val_score(best_model, X_test, y_test, cv=cv_folds, scoring='f1_weighted'))
    balanced_accuracy = np.mean(cross_val_score(best_model, X_test, y_test, cv=cv_folds, scoring='balanced_accuracy'))
    precision = np.mean(cross_val_score(best_model, X_test, y_test, cv=cv_folds, scoring='precision_weighted'))
    recall = np.mean(cross_val_score(best_model, X_test, y_test, cv=cv_folds, scoring='recall_weighted'))
    
    # For specificity and false positive rate, calculate confusion matrix-based metrics
    specificity_list = []
    fpr_list = []
    
    # Get cross-validation predictions
    y_test_pred_cv = cross_val_predict(best_model, X_test, y_test, cv=cv_folds)
    
    for fold in range(cv_folds):
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_cv).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        specificity_list.append(specificity)
        fpr_list.append(fpr)

    # Compute average specificity and false positive rate
    specificity = np.mean(specificity_list)
    fpr = np.mean(fpr_list)
    
    if scoring == 'roc_auc':
        roc_auc,prc_auc = plot_auc_roc_curve(y_test, best_model, X_test, model_name, cv=cv)
        
    else:
        roc_auc = np.mean(cross_val_score(best_model, X_test, y_test, cv=cv_folds, scoring='roc_auc'))
        prc_auc = plot_precision_recall_curve(best_model, X_test, y_test, model_name, cv=cv)
    
    # Print the results
    print(f"Model: {model_name}")
    print(f"Training Accuracy: {accuracy_train:.4f}")
    print(f"Test Accuracy (CV average): {accuracy_test:.4f}")
    print(f"Training F1-Weighted Score: {f1_train:.4f}")
    print(f"Test F1-Weighted Score (CV average): {f1_test:.4f}")
    print(f"Balanced Accuracy (CV average): {balanced_accuracy:.4f}")
    print(f"ROC-AUC Score (CV average): {roc_auc:.4f}")
    print(f"Precision (CV average): {precision:.4f}")
    print(f"Recall (CV average): {recall:.4f}")
    print(f"Specificity (CV average): {specificity:.4f}")
    print(f"False Positive Rate (CV average): {fpr:.4f}")

    # Compile metrics into a list
    metrics = [
        accuracy_train,
        f1_train,
        accuracy_test,
        f1_test,
        balanced_accuracy,
        roc_auc,
        precision,
        recall,
        prc_auc,
        specificity,
        fpr
    ]
    
    # Add the metrics for this model to the dataframe
    metrics_df[model_name] = metrics

    # Return metrics, the best model, and the updated dataframe
    return metrics, best_model, metrics_df

def plot_precision_recall_curve(model, X_test, y_test, model_name, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)):

    # Store precision, recall, and AUC-PRC for each fold
    all_precisions = []
    all_recalls = []
    auc_prc_scores = []

    for train_index, test_index in cv.split(X_test, y_test):
        X_train_cv, X_val_cv = X_test.iloc[train_index], X_test.iloc[test_index]
        y_train_cv, y_val_cv = y_test.iloc[train_index], y_test.iloc[test_index]

        # Get predicted probabilities for the validation fold
        y_val_scores = model.predict_proba(X_val_cv)[:, 1]
        
        # Calculate precision and recall for the fold
        precisions, recalls, _ = precision_recall_curve(y_val_cv, y_val_scores)
        
        # Calculate AUC-PRC for the fold
        auc_prc = auc(recalls, precisions)
        auc_prc_scores.append(auc_prc)
        
        # Interpolate precision values for a smooth mean curve
        all_precisions.append(np.interp(np.linspace(0, 1, 100), recalls[::-1], precisions[::-1]))
        all_recalls.append(np.linspace(0, 1, 100))
        
        # Plot the individual fold's precision-recall curve with AUC-PRC annotation
        plt.plot(recalls, precisions, alpha=0.5, label=f'Fold AUC-PRC = {auc_prc:.3f}')

    # Calculate mean precision and standard deviation
    all_precisions = np.array(all_precisions)
    mean_precision = np.mean(all_precisions, axis=0)
    precision_std = np.std(all_precisions, axis=0)
    
    # Mean AUC-PRC across folds
    mean_auc_prc = np.mean(auc_prc_scores)
    std_auc_prc = np.std(auc_prc_scores)

    # Plot the mean precision-recall curve with ±1 standard deviation
    plt.fill_between(np.linspace(0, 1, 100), mean_precision - precision_std, mean_precision + precision_std, 
                     color='gray', alpha=0.2, label='± 1 STD')
    plt.plot(np.linspace(0, 1, 100), mean_precision, color='blue', 
             label=f'Mean AUC-PRC = {mean_auc_prc:.3f} ± {std_auc_prc:.3f})', linewidth=2)
    
    # Plot settings
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend()
    plt.grid()
    plt.show()
    
    return mean_auc_prc
    
    
def plot_auc_roc_curve(y_test, model, X_test, model_name, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)):
    """
    Plots the AUC-ROC curve with individual cross-validation folds and a shaded area representing ±1 standard deviation.

    Parameters:
    - y_test: True labels for the test set.
    - model: The trained model to use for predictions.
    - X_test: Test feature set.
    - model_name: Name of the model (for plotting purposes).
    - cv: Cross-validation splitting strategy for obtaining predictions.
    """
    # Lists to store metrics for each fold
    tprs = []
    aucs = []
    auprc_scores = []
    mean_fpr = np.linspace(0, 1, 100)  # Standardized FPR values for plotting the mean ROC curve

    # Perform cross-validation
    i = 0
    for train_index, test_index in cv.split(X_test, y_test):
        X_train_cv, X_val_cv = X_test.iloc[train_index], X_test.iloc[test_index]
        y_train_cv, y_val_cv = y_test.iloc[train_index], y_test.iloc[test_index]
        
        # Train and predict probabilities for the validation fold
        model.fit(X_train_cv, y_train_cv)
        y_val_scores = model.predict_proba(X_val_cv)[:, 1]  # Use probabilities for the ROC curve
        
        # Calculate FPR and TPR for the current fold
        fpr, tpr, _ = roc_curve(y_val_cv, y_val_scores)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)  # Store AUC for the fold
        
        # Calculate AUC-PRC for the fold
        precision, recall, _ = precision_recall_curve(y_val_cv, y_val_scores)
        auprc = average_precision_score(y_val_cv, y_val_scores)
        auprc_scores.append(auprc)
        
        # Interpolate TPR values for consistent plotting across folds
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0  # Ensure TPR starts at 0
        
        # Plot the ROC curve for the current fold
        plt.plot(fpr, tpr, alpha=0.3, label=f'Fold {i + 1} ROC curve (AUC = {roc_auc:.2f})')
        i += 1

    # Compute the mean and standard deviation for TPR values across folds
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure TPR ends at 1
    mean_auc = auc(mean_fpr, mean_tpr)
    std_tpr = np.std(tprs, axis=0)
    
    # Mean AUC-PRC across folds
    mean_auprc = np.mean(auprc_scores)
    std_auprc = np.std(auprc_scores)
    
    # Plot the mean ROC curve
    plt.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean ROC curve (AUC = {mean_auc:.2f})', lw=2)

    # Fill the ±1 std dev shaded area
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='gray', alpha=0.2, label='± 1 std. dev.')

    # Plot the chance line (random performance)
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Chance', lw=2)

    # Formatting the plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'AUC-ROC Curve for {model_name}')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    return mean_auc,mean_auprc
    
def recursive_feature_addition_with_gridsearch_plot_roc(model, param_grid, feature_importance_list, X_train, y_train, X_test, y_test):
    # Initialize lists to store results
    selected_features = []
    auc_train_scores = []  # For training data (cross-validated)
    auc_test_scores = []   # For testing data (cross-validated)
    
    # Variables to track the best AUC-ROC score and corresponding feature set
    best_auc_score = -np.inf  # Initialize to negative infinity
    best_num_features = 0
    best_feature_set = []
    best_params = None

    # Set the random seed for reproducibility
    np.random.seed(0)
    
    # Start Recursive Feature Addition
    for feature in feature_importance_list[:500]:
        selected_features.append(feature)
        
        # Perform hyperparameter tuning with GridSearchCV (using all available cores)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)  # Multicore processing and set random state
        grid_search.fit(X_train[selected_features], y_train)
        
        # Get the best model from the grid search
        best_model = grid_search.best_estimator_
        
        # Perform 10-fold cross-validation with the selected features (on training set)
        auc_cv_scores_train = cross_val_score(best_model, X_train[selected_features], y_train, cv=5, scoring='roc_auc', n_jobs=-1)  # Multicore processing and set random state
        
        # Calculate the average AUC-ROC score across the 10 folds for training data
        auc_train_avg = np.mean(auc_cv_scores_train)
        auc_train_scores.append(auc_train_avg)
        
        # Perform cross-validated predictions on the entire test dataset (using all cores)
        y_pred_cv = cross_val_predict(best_model, X_test[selected_features], y_test, cv=5, n_jobs=-1)  # Multicore processing and set random state
        
        # Calculate AUC-ROC score on cross-validated predictions (test evaluation)
        auc_test_avg = roc_auc_score(y_test, y_pred_cv)
        auc_test_scores.append(auc_test_avg)
        
        # Update the best AUC-ROC score and feature set if the current one is better
        if auc_test_avg > best_auc_score:
            best_auc_score = auc_test_avg
            best_num_features = len(selected_features)
            best_feature_set = list(selected_features)  # Copy of the current best feature set
            best_params = grid_search.best_params_
            
        print(f"{feature} done")

    # Plot AUC-ROC score vs. number of features for both training and testing sets
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(auc_train_scores) + 1), auc_train_scores, marker='o', linestyle='-', color='b', label='Train AUC-ROC Score', zorder=1)
    plt.plot(range(1, len(auc_test_scores) + 1), auc_test_scores, marker='o', linestyle='-', color='g', label='Test AUC-ROC Score', zorder=2)
    
    # Annotate the point with the highest test AUC-ROC score
    plt.scatter(best_num_features, best_auc_score, color='red', label=f'Best Test AUC-ROC: {best_auc_score:.4f}', zorder=3)
    plt.text(best_num_features, best_auc_score, f'  Best Score ({best_num_features} features)', 
             verticalalignment='bottom', horizontalalignment='right', zorder=3)
    
    plt.xlabel('Number of Features')
    plt.ylabel('AUC-ROC Score')
    plt.title('AUC-ROC Score vs. Number of Features (Train and Test)')
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"Best Test AUC-ROC Score: {best_auc_score:.4f}")
    print(f"Number of Features: {best_num_features}")
    print(f"Selected Features: {best_feature_set}")

    return best_params, auc_train_scores, auc_test_scores, best_feature_set

def differential_expression_analysis_optimized(X_train, y_train, fold_change_threshold=1.2, p_value_threshold=0.05):
    # Pre-compute means for each gene in each condition
    control_means = X_train[y_train == 0].mean(axis=0)
    case_means = X_train[y_train == 1].mean(axis=0)
    
    # Compute raw fold change (without log transform)
    #fold_changes = (case_means + 1) / (control_means + 1)
    log2_fc = case_means - control_means
    fc = 2**log2_fc
    fc_list = [(i if i > 1 else -1/i) for i in fc]
    
    # Perform Mann-Whitney U test for each gene and store p-values in a list
    pvalues = [
        mannwhitneyu(
            X_train[gene][y_train == 0],
            X_train[gene][y_train == 1],
            alternative='two-sided'
        )[1]
        for gene in X_train.columns
    ]
    
    # Adjust p-values using FDR correction
    fdr_adjusted_pvalues = multipletests(pvalues, method='fdr_bh')[1]
    
    # Create DataFrame to store results
    results = pd.DataFrame({
        'gene': X_train.columns,
        'p_value': pvalues,
        'fdr': fdr_adjusted_pvalues,
        'fold_change': fc_list #fold_changes
    })
    
    # Filter results based on FDR threshold and fold change threshold
    filtered_results = results[(results['fdr'] < p_value_threshold) & (np.abs(results['fold_change']) > fold_change_threshold)]
    
    return results, filtered_results

# train ML model with gridsearchCV using f1_weighted score
def train_model_with_gridsearch_f1_no_test(model, param_grid, X_train, y_train, scoring='f1_weighted', 
                                 cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)):
    
    # Ensure the model has a random state set, if applicable
    if hasattr(model, 'random_state'):
        model.random_state = 0

    # Initialize GridSearchCV with the specified model and hyperparameter grid
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               cv=cv, scoring=scoring, n_jobs=-1, verbose=1)

    # Fit the model on the training data
    grid_search.fit(X_train, y_train)

    # Get the best model and its hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    ### Evaluate on training set
    y_train_pred = best_model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    f1_train = f1_score(y_train, y_train_pred, average='weighted')
    print(f"Training Accuracy: {accuracy_train}")
    print(f"Training F1-Weighted Score: {f1_train}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_train, y_train_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    

    # Print the results
    print(f"Best Hyperparameters: {best_params}")

    # Return the best model, its parameters, accuracy on train and test sets, F1-Weighted scores, 
    # report, and confusion matrix
    return best_model, best_params

def train_model_with_gridsearch_roc_no_test(model, param_grid, X_train, y_train, scoring='roc_auc', 
                                 cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)):
   
    # Ensure the model has a random state set, if applicable
    if hasattr(model, 'random_state'):
        model.random_state = 0

    # Initialize GridSearchCV with the specified model and hyperparameter grid
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               cv=cv, scoring=scoring, n_jobs=-1, verbose=1)

    # Fit the model on the training data
    grid_search.fit(X_train, y_train)

    # Get the best model and its hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    ### Evaluate on training set
    y_train_pred = best_model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {accuracy_train}")
    
    # Calculate AUC-ROC for the train set if binary classification and roc_auc is selected
    if scoring == 'roc_auc' and len(set(y_train)) == 2:  # Binary classification
        y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]  # Probabilities for the positive class
        auc_roc_train = roc_auc_score(y_train, y_train_pred_proba)
        print(f"Train AUC-ROC: {auc_roc_train}")
    else:
        auc_roc_test = None

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_train, y_train_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Return the best model, its parameters, accuracy on train and test sets, AUC-ROC scores (if applicable), 
    # report, and confusion matrix
    return best_model, best_params

def recursive_feature_addition_with_gridsearch_plot_no_test(model, param_grid, feature_importance_list, X_train, y_train):
    # Initialize lists to store results
    selected_features = []
    f1_train_scores = []
    
    # Variables to track the best F1 score and corresponding feature set
    best_f1_score = -np.inf
    best_num_features = 0
    best_feature_set = []
    best_params = None

    # Variables to hold values from the previous iteration
    prev_f1_score = None
    prev_f1_cv_scores_train = None

    np.random.seed(0)  # For reproducibility
    
    # Start Recursive Feature Addition
    for i, feature in enumerate(feature_importance_list[:500]):
        selected_features.append(feature)
        
        # Perform hyperparameter tuning with GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train[selected_features], y_train)
        
        # Get the best model from the grid search
        best_model = grid_search.best_estimator_
        
        # Perform 5-fold cross-validation with the selected features
        f1_cv_scores_train = cross_val_score(best_model, X_train[selected_features], y_train, cv=5, scoring='f1_weighted', n_jobs=-1)
        
        # Calculate the average F1 score across folds for training data
        f1_train_avg = np.mean(f1_cv_scores_train)
        f1_train_scores.append(f1_train_avg)
        
        # Skip t-test on the first iteration
        if i == 0:
            prev_f1_score = f1_train_avg
            prev_f1_cv_scores_train = f1_cv_scores_train
            best_f1_score = f1_train_avg
            best_num_features = len(selected_features)
            best_feature_set = list(selected_features)
            best_params = grid_search.best_params_
            continue

        # Perform t-test between the F1 scores of the current and previous feature sets
        _, p_value = ttest_ind(f1_cv_scores_train, prev_f1_cv_scores_train)
        
        # If the p-value is significant, update the best scores and feature set
        if p_value < 0.05:
            best_f1_score = f1_train_avg
            best_num_features = len(selected_features)
            best_feature_set = list(selected_features)
            best_params = grid_search.best_params_
        
        # Update previous values for the next iteration
        prev_f1_score = f1_train_avg
        prev_f1_cv_scores_train = f1_cv_scores_train

    # Plot F1 score vs. number of features
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(f1_train_scores) + 1), f1_train_scores, marker='o', linestyle='-', color='b', label='Train F1-weighted Score', zorder=1)
    
    # Highlight the best feature set with a red dot and dotted line
    plt.axvline(x=best_num_features, color='r', linestyle='--', zorder=0)
    plt.scatter([best_num_features], [best_f1_score], color='r', s=100, label='Best Feature Set', zorder=2)
    
    
    plt.xlabel('Number of Features')
    plt.ylabel('F1-weighted Score')
    plt.title('F1-weighted Score vs. Number of Features (Train)')
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"Best Train F1 Score: {best_f1_score:.4f}")
    print(f"Number of Features: {best_num_features}")
    print(f"Selected Features: {best_feature_set}")

    return best_params, f1_train_scores, best_feature_set

def recursive_feature_addition_with_optimized_model(model, feature_importance_list, X_train, y_train, target_f1_score=0.85):
    # Initialize lists to store results
    selected_features = []
    f1_train_scores = []
    
    # Variables to hold values from the previous iteration
    prev_f1_score = None
    prev_f1_cv_scores_train = None

    # Track the minimum number of features to achieve the target F1-score
    min_features_for_target_score = None
    feature_set_for_target_score = None
    selected_f1_score = -np.inf

    # Set random seed for reproducibility
    np.random.seed(0)

    # Start Recursive Feature Addition
    for i, feature in enumerate(feature_importance_list[:500]):
        selected_features.append(feature)
        
        # Perform 5-fold cross-validation with the selected features
        f1_cv_scores_train = cross_val_score(
            model, X_train[selected_features], y_train, cv=10, scoring='f1_weighted', n_jobs=-1
        )
        
        # Calculate the average F1 score across the folds
        f1_train_avg = np.mean(f1_cv_scores_train)
        f1_train_scores.append(f1_train_avg)
        
        # Skip t-test on the first iteration
        if i == 0:
            prev_f1_score = f1_train_avg
            prev_f1_cv_scores_train = f1_cv_scores_train
            best_f1_score = f1_train_avg
            best_num_features = len(selected_features)
            best_feature_set = list(selected_features)
            continue

        # Check if the target F1 score is reached
        if f1_train_avg >= target_f1_score and min_features_for_target_score is None:
            min_features_for_target_score = len(selected_features)
            feature_set_for_target_score = list(selected_features)
            selected_f1_score = f1_train_avg
        
        # Update previous values for the next iteration
        prev_f1_score = f1_train_avg
        prev_f1_cv_scores_train = f1_cv_scores_train

    # Plot F1 score vs. number of features
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(f1_train_scores) + 1), f1_train_scores, marker='o', linestyle='-', color='b', label='Train F1-weighted Score', zorder=1)
    
    # Highlight the minimum feature set for target score with a green dot and dotted line
    if min_features_for_target_score:
        plt.axvline(x=min_features_for_target_score, color='r', linestyle='--', label='Target F1-Score Feature Count', zorder=0)
        plt.scatter([min_features_for_target_score], [target_f1_score], color='r', s=100, label=f'Target F1-Score ({target_f1_score})', zorder=2)

    # Highlight the best feature set with a red dot and dotted line
    #plt.axvline(x=best_num_features, color='r', linestyle='--', label='Best Feature Count', zorder=0)
    #plt.scatter([best_num_features], [best_f1_score], color='r', s=100, label='Best F1-Score', zorder=2)
    plt.text(min_features_for_target_score, selected_f1_score, f'  Best Score ({min_features_for_target_score} features)', 
             verticalalignment='bottom', horizontalalignment='right', zorder=3)
    
    plt.xlabel('Number of Features')
    plt.ylabel('F1-weighted Score')
    plt.title('F1-weighted Score vs. Number of Features (Train)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Output the results
    print(f"Best Train F1 Score: {selected_f1_score:.4f}")
    print(f"Minimum Number of Features for Target F1 Score: {min_features_for_target_score}")
    print(f"Selected Features for Target F1 Score: {feature_set_for_target_score}")

    return feature_set_for_target_score

def train_model_with_gridsearch_metrics_all(model, 
                                        param_grid, 
                                        model_name, 
                                        metrics_df, 
                                        X_train, 
                                        y_train, 
                                        X_test, 
                                        y_test,
                                        dataset,
                                        scoring,
                                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)):
    # Calculate sample weights to handle class imbalance
    class_counts = y_train.value_counts()
    majority_class_count = class_counts.max()
    sample_weight_train = y_train.apply(lambda x: majority_class_count / class_counts[x])
    
    # Calculate sample weights for y_test as well
    class_counts_test = y_test.value_counts()
    majority_class_count_test = class_counts_test.max()
    sample_weight_test = y_test.apply(lambda x: majority_class_count_test / class_counts_test[x])

    # Ensure the model has a random state set, if applicable
    if hasattr(model, 'random_state'):
        model.random_state = 0
    
    # List of models that support sample_weight
    supports_sample_weight = not isinstance(model, (KNeighborsClassifier, MLPClassifier, DummyClassifier))
    
    # Skip GridSearch for Dummy Classifier
    if model_name == "Dummy Classifier":
        best_model = DummyClassifier(strategy="most_frequent", random_state=0).fit(X_train, y_train)
    else:
        # Initialize GridSearchCV with the specified model and hyperparameter grid
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                                   cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
        
        # Check if sample weights should be applied
        if supports_sample_weight:
            grid_search.fit(X_train, y_train, sample_weight=sample_weight_train)
        else:
            grid_search.fit(X_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_
    
    # For specificity and false positive rate, calculate confusion matrix-based metrics
    specificity_list = []
    sensitivity_list = []
    fpr_list = []

    ### Evaluate on training set
    y_train_pred = best_model.predict(X_train)

    ### Evaluate on test set with 5-fold cross-validation (average metrics)
    cv_folds = 5  # Set the number of cross-validation folds
    if supports_sample_weight:
        accuracy_train = accuracy_score(y_train, y_train_pred, sample_weight=sample_weight_train)
        f1_train = f1_score(y_train, y_train_pred, average='weighted', sample_weight=sample_weight_train)
        accuracy_test = np.mean(cross_val_score(best_model, X_test, y_test, cv=cv_folds, scoring='accuracy', fit_params={'sample_weight': sample_weight_test}))
        f1_test = np.mean(cross_val_score(best_model, X_test, y_test, cv=cv_folds, scoring='f1_weighted', fit_params={'sample_weight': sample_weight_test}))
        balanced_accuracy = np.mean(cross_val_score(best_model, X_test, y_test, cv=cv_folds, scoring='balanced_accuracy', fit_params={'sample_weight': sample_weight_test}))
        precision = np.mean(cross_val_score(best_model, X_test, y_test, cv=cv_folds, scoring='precision_weighted', fit_params={'sample_weight': sample_weight_test}))
        recall = np.mean(cross_val_score(best_model, X_test, y_test, cv=cv_folds, scoring='recall_weighted', fit_params={'sample_weight': sample_weight_test}))
        # Get cross-validation predictions
        y_test_pred_cv = cross_val_predict(best_model, X_test, y_test, cv=cv_folds, fit_params={'sample_weight': sample_weight_test})
    else:
        accuracy_train = accuracy_score(y_train, y_train_pred)
        f1_train = f1_score(y_train, y_train_pred, average='weighted')
        accuracy_test = np.mean(cross_val_score(best_model, X_test, y_test, cv=cv_folds, scoring='accuracy'))
        f1_test = np.mean(cross_val_score(best_model, X_test, y_test, cv=cv_folds, scoring='f1_weighted'))
        balanced_accuracy = np.mean(cross_val_score(best_model, X_test, y_test, cv=cv_folds, scoring='balanced_accuracy'))
        precision = np.mean(cross_val_score(best_model, X_test, y_test, cv=cv_folds, scoring='precision_weighted'))
        recall = np.mean(cross_val_score(best_model, X_test, y_test, cv=cv_folds, scoring='recall_weighted'))
        y_test_pred_cv = cross_val_predict(best_model, X_test, y_test, cv=cv_folds)
    
    for fold in range(cv_folds):
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_cv).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        specificity_list.append(specificity)
        sensitivity_list.append(sensitivity)
        fpr_list.append(fpr)

    # Compute average specificity and false positive rate
    specificity = np.mean(specificity_list)
    sensitivity = np.mean(sensitivity_list)
    fpr = np.mean(fpr_list)
    
    if scoring == 'roc_auc':
        roc_auc, prc_auc = plot_auc_roc_curve(y_test, best_model, X_test, model_name, cv=cv)
        roc_auc_train = np.mean(cross_val_score(best_model, X_train, y_train, cv=cv, scoring='roc_auc'))

    else:
        if supports_sample_weight:
            roc_auc = np.mean(cross_val_score(best_model, X_test, y_test, cv=cv, scoring='roc_auc', fit_params={'sample_weight': sample_weight_test}))
            roc_auc_train = np.mean(cross_val_score(best_model, X_train, y_train, cv=cv, scoring='roc_auc', fit_params={'sample_weight': sample_weight_train}))

            
            
        else:
            roc_auc_train = np.mean(cross_val_score(best_model, X_train, y_train, cv=cv, scoring='roc_auc'))

            roc_auc = np.mean(cross_val_score(best_model, X_test, y_test, cv=cv, scoring='roc_auc'))
        prc_auc = plot_precision_recall_curve(best_model, X_test, y_test, model_name, cv=cv)
        
    # Print the results
    print(f"Model: {model_name}")
    print(f"Training Accuracy: {accuracy_train:.4f}")
    print(f"Test Accuracy (CV average): {accuracy_test:.4f}")
    print(f"Training F1-Weighted Score: {f1_train:.4f}")
    print(f"Test F1-Weighted Score (CV average): {f1_test:.4f}")
    print(f"Balanced Accuracy (CV average): {balanced_accuracy:.4f}")
    print(f"Training ROC-AUC Score (CV average): {roc_auc_train:.4f}")
    print(f"ROC-AUC Score (CV average): {roc_auc:.4f}")
    print(f"Precision (CV average): {precision:.4f}")
    print(f"Recall (CV average): {recall:.4f}")
    print(f"Specificity (CV average): {specificity:.4f}")
    print(f"Sensitivity (CV average): {sensitivity:.4f}")
    print(f"False Positive Rate (CV average): {fpr:.4f}")

    # Compile metrics into a list
    metrics = [
        accuracy_train,
        f1_train,
        roc_auc_train,
        accuracy_test,
        f1_test,
        balanced_accuracy,
        roc_auc,
        precision,
        recall,
        prc_auc,
        specificity,
        sensitivity,
        fpr
    ]
    
    # Add the metrics for this model to the dataframe
    metrics_df[model_name] = metrics

    # Return metrics, the best model, and the updated dataframe
    return metrics, best_model, metrics_df

def recursive_feature_addition_with_optimized_model_roc(model, feature_importance_list, X_train, y_train, target_auc_score):
    # Initialize lists to store results
    selected_features = []
    auc_train_scores = []
    
    # Variables to hold values from the previous iteration
    prev_auc_score = None
    prev_auc_cv_scores_train = None

    # Track the minimum number of features to achieve the target AUC-ROC score
    min_features_for_target_score = None
    feature_set_for_target_score = None
    selected_auc_score = -np.inf

    # Set random seed for reproducibility
    np.random.seed(0)

    # Start Recursive Feature Addition
    for i, feature in enumerate(feature_importance_list[:500]):
        selected_features.append(feature)
        
        # Perform 5-fold cross-validation with the selected features
        auc_cv_scores_train = cross_val_score(
            model, X_train[selected_features], y_train, cv=10, scoring='roc_auc', n_jobs=-1
        )
        
        # Calculate the average AUC-ROC score across the folds
        auc_train_avg = np.mean(auc_cv_scores_train)
        auc_train_scores.append(auc_train_avg)
        
        # Skip t-test on the first iteration
        if i == 0:
            prev_auc_score = auc_train_avg
            prev_auc_cv_scores_train = auc_cv_scores_train
            best_auc_score = auc_train_avg
            best_num_features = len(selected_features)
            best_feature_set = list(selected_features)
            continue

        # Check if the target AUC-ROC score is reached
        if auc_train_avg >= target_auc_score and min_features_for_target_score is None:
            min_features_for_target_score = len(selected_features)
            feature_set_for_target_score = list(selected_features)
            selected_auc_score = auc_train_avg
        
        # Update previous values for the next iteration
        prev_auc_score = auc_train_avg
        prev_auc_cv_scores_train = auc_cv_scores_train

    # Plot AUC-ROC score vs. number of features
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(auc_train_scores) + 1), auc_train_scores, marker='o', linestyle='-', color='b', label='Train AUC-ROC Score', zorder=1)
    
    # Highlight the minimum feature set for target score with a green dot and dotted line
    if min_features_for_target_score:
        plt.axvline(x=min_features_for_target_score, color='r', linestyle='--', label='Target AUC-ROC Feature Count', zorder=0)
        plt.scatter([min_features_for_target_score], [target_auc_score], color='r', s=100, label=f'Target AUC-ROC ({target_auc_score})', zorder=2)

    # Add text for the best feature set with a red dot
    plt.text(min_features_for_target_score, selected_auc_score, f'  Best Score ({min_features_for_target_score} features)', 
             verticalalignment='bottom', horizontalalignment='right', zorder=3)
    
    plt.xlabel('Number of Features')
    plt.ylabel('AUC-ROC Score')
    plt.title('AUC-ROC Score vs. Number of Features (Train)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Output the results
    print(f"Best Train AUC-ROC Score: {selected_auc_score:.4f}")
    print(f"Minimum Number of Features for Target AUC-ROC Score: {min_features_for_target_score}")
    print(f"Selected Features for Target AUC-ROC Score: {feature_set_for_target_score}")

    return feature_set_for_target_score

def evaluate_model_performance_with_plot(model, X_test, y_test):
    """
    Evaluates the performance of a trained model on test data and plots a confusion matrix.
    
    Parameters:
    - model: Trained model.
    - X_test: Features for testing.
    - y_test: True labels for testing.
    
    Returns:
    - A dictionary containing the calculated metrics.
    """
    # Get model predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average='binary')  # Change average if multiclass
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # PRC AUC (Precision-Recall Curve AUC)
    if y_pred_proba is not None:
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
        prc_auc = auc(recall_vals, precision_vals)
    else:
        prc_auc = None
    
    # Confusion matrix and derived metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall  # Sensitivity is the same as recall
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'], 
                yticklabels=['True Negative', 'True Positive'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Create results dictionary
    results = {
        'Accuracy (Test)': accuracy_test,
        'F1 Score (Test)': f1_test,
        'Balanced Accuracy': balanced_accuracy,
        'ROC AUC': roc_auc,
        'Precision': precision,
        'Recall': recall,
        'PRC AUC': prc_auc,
        'Specificity': specificity,
        'Sensitivity': sensitivity,
        'False Positive Rate': false_positive_rate,
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn,
        'True Positives': tp
    }
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results.items(), columns=['Metric', 'Value'])
    
    return results_df