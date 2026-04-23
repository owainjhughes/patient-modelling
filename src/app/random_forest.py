import os
import joblib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

def _save(output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def _infer_target(df):
    return df.columns[-1]


def _infer_features(df, target_col):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    return [c for c in num_cols if c != target_col]


def _plot_confusion_matrix(cm, labels, title, filename, output_dir):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    _save(output_dir, filename)


def _plot_roc(fpr, tpr, roc_auc, title, filename, output_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    _save(output_dir, filename)


def _plot_feature_importance_rf(model, feature_names, output_dir):
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    plt.figure(figsize=(8, max(4, len(feature_names) // 2)))
    importances.plot(kind='barh', color='steelblue')
    plt.title('Feature Importance – Random Forest')
    plt.xlabel('Importance Score')
    plt.grid(True, axis='x')
    plt.tight_layout()
    _save(output_dir, 'rf_feature_importance.png')


def _plot_feature_importance_lr(model, feature_names, output_dir):
    coefs = pd.Series(model.coef_[0], index=feature_names).sort_values()
    plt.figure(figsize=(8, max(4, len(feature_names) // 2)))
    coefs.plot(kind='barh', color='steelblue')
    plt.title('Feature Importance – Logistic Regression Coefficients')
    plt.xlabel('Coefficient Weight')
    plt.grid(True, axis='x')
    plt.tight_layout()
    _save(output_dir, 'lr_feature_importance.png')


def run_random_forest(input_csv, target_col=None, feature_cols=None, test_size=0.2, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Random Forest pipeline — loading: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)

    # Resolve target and features
    if target_col is None:
        target_col = _infer_target(df)
        print(f"No target specified — using last column: '{target_col}'")
    if feature_cols is None:
        feature_cols = _infer_features(df, target_col)
        print(f"No features specified — using numeric columns: {feature_cols}")

    if not feature_cols:
        raise ValueError("No usable numeric feature columns found. "
                         "Pass feature_cols explicitly.")

    X = df[feature_cols].dropna()
    y = df.loc[X.index, target_col]

    print(f"Target: '{target_col}'  |  Classes: {sorted(y.unique())}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Dataset: {X.shape[0]} rows after dropping NaN rows")

    labels = sorted(y.unique())
    is_binary = len(labels) == 2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print(f"\n{'-'*40}\nTraining Random Forest (100 trees)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_sc, y_train)
    y_pred_rf = rf.predict(X_test_sc)

    print("\nRandom Forest — Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))
    print("\nRandom Forest — Classification Report:")
    print(classification_report(y_test, y_pred_rf))

    _plot_confusion_matrix(
        confusion_matrix(y_test, y_pred_rf), labels,
        'Confusion Matrix – Random Forest', 'rf_confusion_matrix.png', output_dir
    )
    _plot_feature_importance_rf(rf, feature_cols, output_dir)

    print(f"\n{'-'*40}\nTraining Logistic Regression (baseline)...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    y_pred_lr = lr.predict(X_test_sc)

    print("\nLogistic Regression — Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lr))
    print("\nLogistic Regression — Classification Report:")
    print(classification_report(y_test, y_pred_lr))

    _plot_confusion_matrix(
        confusion_matrix(y_test, y_pred_lr), labels,
        'Confusion Matrix – Logistic Regression', 'lr_confusion_matrix.png', output_dir
    )
    _plot_feature_importance_lr(lr, feature_cols, output_dir)

    if is_binary:  # ROC curve only meaningful for binary classification
        pos_label = labels[1]
        y_probs = lr.predict_proba(X_test_sc)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs, pos_label=pos_label)
        roc_auc_val = auc(fpr, tpr)
        print(f"\nLogistic Regression — AUC: {roc_auc_val:.2f}")
        _plot_roc(fpr, tpr, roc_auc_val,
                  'ROC Curve – Logistic Regression', 'lr_roc_curve.png', output_dir)

    print(f"\n{'-'*40}\n5-Fold Cross-Validation (Logistic Regression)...")
    cv_scores = cross_val_score(lr, X, y, cv=5, scoring='accuracy')
    print(f"CV Accuracy Scores: {cv_scores}")
    print(f"Mean: {cv_scores.mean():.2f}  |  Std: {cv_scores.std():.2f}")

    model_path = os.path.join(output_dir, 'random_forest_model.pkl')
    joblib.dump(rf, model_path)
    print(f"\nRandom Forest model saved to: {model_path}")

    lr_path = os.path.join(output_dir, 'logistic_regression_model.pkl')
    joblib.dump(lr, lr_path)
    print(f"Logistic Regression model saved to: {lr_path}")

    print(f"{'='*60}\n")
    return rf, lr


if __name__ == '__main__':
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else 'outputs/cleaned_data.csv'
    target = sys.argv[2] if len(sys.argv) > 2 else None
    run_random_forest(csv, target_col=target, output_dir='outputs')
