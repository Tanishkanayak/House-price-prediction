import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('C:/Users/KIIT/Desktop/Proj/house_prices.csv')

# Show dataset info
print(df.info())

# Drop irrelevant columns
df.drop(columns=['Index', 'Title', 'Description', 'location', 'Society', 'Ownership', 'Dimensions', 'Plot Area'], inplace=True)

# Rename 'Price (in rupees)' to 'Price'
df.rename(columns={"Price (in rupees)": "Price"}, inplace=True)

# Drop rows where 'Price' is missing
df = df.dropna(subset=['Price'])

# Convert numeric-like object columns to float
convert_cols = ['Amount(in rupees)', 'Carpet Area', 'Floor', 'Bathroom', 'Balcony', 'Car Parking', 'Super Area']
for col in convert_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values instead of dropping all
df.fillna({
    'Carpet Area': df['Carpet Area'].median(),
    'Bathroom': df['Bathroom'].median(),
    'Balcony': df['Balcony'].median(),
    'Car Parking': 0,  # Assume 0 if not mentioned
    'Super Area': df['Super Area'].median()
}, inplace=True)

# Selecting numeric features
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

# Ensure 'Price' is included
if 'Price' not in numeric_features:
    raise ValueError("Error: 'Price' column is missing or not numeric. Check column names.")

# Define features and target
X = df[numeric_features].drop(columns=['Price'], errors='ignore')
y = df['Price']

# Debugging: Check if dataset is empty
print("X Columns:", X.columns)
print("Number of rows in X:", len(X))
print("Number of rows in y:", len(y))

if X.empty or y.empty:
    raise ValueError("🚨 Error: X or y is empty. Fix missing data before proceeding.")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'R2 Score: {r2}')
