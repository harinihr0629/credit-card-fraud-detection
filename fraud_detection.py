import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("creditcard.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# ==========================================
# 2. CHECK DATASET
# ==========================================

print("\nFirst 5 rows:")
print(data.head())

print("\nFraud and Normal Transactions:")
print(data["Class"].value_counts())


# ==========================================
# 3. SEPARATE INPUT AND OUTPUT
# ==========================================

X = data.drop("Class", axis=1)
y = data["Class"]


# ==========================================
# 4. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 5. SCALE FEATURES
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 6. LOGISTIC REGRESSION
# ==========================================

print("\nTraining Logistic Regression...")

logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

logistic_model.fit(X_train_scaled, y_train)

lr_prediction = logistic_model.predict(X_test_scaled)


# ==========================================
# 7. LOGISTIC REGRESSION RESULTS
# ==========================================

print("\n================================")
print("LOGISTIC REGRESSION RESULTS")
print("================================")

print("Accuracy :", accuracy_score(y_test, lr_prediction))
print("Precision:", precision_score(y_test, lr_prediction))
print("Recall   :", recall_score(y_test, lr_prediction))
print("F1 Score :", f1_score(y_test, lr_prediction))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_prediction))

print("\nClassification Report:")
print(classification_report(y_test, lr_prediction))


# ==========================================
# 8. RANDOM FOREST
# ==========================================

print("\nTraining Random Forest...")

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

random_forest_model.fit(X_train, y_train)

rf_prediction = random_forest_model.predict(X_test)
# ==========================================
# 9. SAVE TRAINED MODELS
# ==========================================

print("\nSaving trained models...")

joblib.dump(logistic_model, "logistic_model.pkl")
joblib.dump(random_forest_model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Logistic Regression model saved!")
print("Random Forest model saved!")
print("Scaler saved!")


# ==========================================
# 9. RANDOM FOREST RESULTS
# ==========================================

print("\n================================")
print("RANDOM FOREST RESULTS")
print("================================")

print("Accuracy :", accuracy_score(y_test, rf_prediction))
print("Precision:", precision_score(y_test, rf_prediction))
print("Recall   :", recall_score(y_test, rf_prediction))
print("F1 Score :", f1_score(y_test, rf_prediction))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_prediction))

print("\nClassification Report:")
print(classification_report(y_test, rf_prediction))


# ==========================================
# 10. FINAL COMPARISON
# ==========================================

print("\n================================")
print("MODEL COMPARISON")
print("================================")

print("\nLogistic Regression")
print("Accuracy :", accuracy_score(y_test, lr_prediction))
print("Precision:", precision_score(y_test, lr_prediction))
print("Recall   :", recall_score(y_test, lr_prediction))
print("F1 Score :", f1_score(y_test, lr_prediction))

print("\nRandom Forest")
print("Accuracy :", accuracy_score(y_test, rf_prediction))
print("Precision:", precision_score(y_test, rf_prediction))
print("Recall   :", recall_score(y_test, rf_prediction))
print("F1 Score :", f1_score(y_test, rf_prediction))


print("\n================================")
print("PROGRAM COMPLETED")
print("================================")