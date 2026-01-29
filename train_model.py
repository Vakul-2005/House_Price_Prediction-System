import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt
import os

# Make sure model folder exists
os.makedirs("model", exist_ok=True)

# Load data
data = pd.read_csv("data/house_data.csv")
print("Data shape:", data.shape)
print(data.head())

# Features & target
FEATURES = ['area','bedrooms','bathrooms','stories','parking']
X = data[FEATURES]
y = data['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Train model
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f"R2: {r2:.3f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Save model
joblib.dump(model, "models/house_price_model.pkl")
print("✅ Model saved successfully!")
