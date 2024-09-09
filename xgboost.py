import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:\\mnt\\data\\train.csv'
data = pd.read_csv(file_path)

# Assuming 'date' is the datetime column and 'oil_temp' is the target variable
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')

# Normalize the features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Convert scaled data back to a DataFrame
scaled_data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

# Feature Engineering: Create lag features
for lag in range(1, 25):  # Create lag features for the past 24 hours
    scaled_data[f'lag_{lag}'] = scaled_data['oil_temp'].shift(lag)

# Drop rows with NaN values (which appear because of lagging)
scaled_data = scaled_data.dropna()

# Define input features (X) and target variable (y)
X = scaled_data.drop(['oil_temp'], axis=1)
y = scaled_data['oil_temp']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Additional Feature Engineering: Time-based features
X_train['hour'] = X_train.index.hour
X_test['hour'] = X_test.index.hour

X_train['dayofweek'] = X_train.index.dayofweek
X_test['dayofweek'] = X_test.index.dayofweek

# Initialize the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# Plotting the predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Actual vs Predicted Oil Temperature (Test Set)')
plt.xlabel('Time steps')
plt.ylabel('Oil Temperature')
plt.legend()
plt.show()

# Prepare the last sequence from the training data
last_sequence = scaled_data[-24:]  # The last 24 rows

# Create lag features for the last sequence
for lag in range(1, 25):
    last_sequence[f'lag_{lag}'] = last_sequence['oil_temp'].shift(lag)

# Drop rows with NaN values (due to lagging)
last_sequence = last_sequence.dropna()

# Add time-based features
last_sequence['hour'] = last_sequence.index.hour
last_sequence['dayofweek'] = last_sequence.index.dayofweek

# Predict the next 24 hours
next_24_hours = model.predict(last_sequence.drop(['oil_temp'], axis=1))

# Inverse transform the predictions to get the actual values
next_24_hours_rescaled = scaler.inverse_transform(
    np.concatenate((next_24_hours.reshape(-1, 1), np.zeros((next_24_hours.shape[0], scaled_data.shape[1] - 1))), axis=1)
)[:, 0]

print("Predicted Oil Temperature for the next 24 hours:")
print(next_24_hours_rescaled)
