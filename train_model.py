from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os


diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

print("Dataset loaded.")
print(f"Dataset shape: {X.shape}")
print(f"Features: {diabetes.feature_names}")
print(f"Target range: {y.min():.1f} to {y.max():.1f}")


print("Sample preparation.")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

print("Training model.")
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=10
)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model evaluations.")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.3f}")

os.makedirs('models', exist_ok=True)

with open('models/diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model successfully trained and saved.")
