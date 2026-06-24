import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Save business columns before encoding
monthly_charges = df["MonthlyCharges"]
tenure = df["tenure"]

# Remove customerID
df = df.drop("customerID", axis=1)

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Features and target
X = df_encoded.drop("Churn_Yes", axis=1)
y = df_encoded["Churn_Yes"]

# Train-test split
X_train, X_test, y_train, y_test, charge_train, charge_test, tenure_train, tenure_test = train_test_split(
    X,
    y,
    monthly_charges,
    tenure,
    test_size=0.2,
    random_state=42
)

# Logistic Regression
model = LogisticRegression(max_iter=3000)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", round(accuracy * 100, 2), "%")


# Export for Power BI
results = pd.DataFrame({
    "Actual_Churn": y_test.values,
    "Predicted_Churn": predictions,
    "Churn_Probability": probabilities,
    "MonthlyCharges": charge_test.values,
    "Tenure": tenure_test.values
})


results.to_csv(
    "churn_dashboard_data.csv",
    index=False
)

print("Dashboard file updated successfully!")