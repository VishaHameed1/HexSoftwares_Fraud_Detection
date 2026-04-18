import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Synthetic Data Generation (Creating our own dataset)
def generate_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'amount': np.random.uniform(10, 5000, n_samples),
        'distance_from_home': np.random.uniform(0, 100, n_samples),
        'is_online_order': np.random.randint(0, 2, n_samples),
        'is_fraud': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]) # 5% Fraud
    }
    return pd.DataFrame(data)

# 2. Training Logic
print("Generating Synthetic Financial Data...")
df = generate_data()

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Training Random Forest Model for Fraud Detection...")
model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)

# 3. Predict & Show Results
predictions = model.predict(X_test)

print("\n--- Hex Softwares Model Report ---")
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 4. Manual Testing
def test_tx(amt, dist, online):
    res = model.predict([[amt, dist, online]])
    status = "⚠️ FRAUD DETECTED" if res[0] == 1 else "✅ NORMAL TRANSACTION"
    print(f"Testing: Amt: {amt}, Dist: {dist}, Online: {online} -> {status}")

print("\n--- Testing Real-time Scenarios ---")
test_tx(50, 2, 1)    # Normal case
test_tx(4500, 95, 1) # Potential Fraud (High amount + High distance)