import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.optimizers import Adam

# Load your dataset
data = pd.read_csv('stock_data.csv')  # Replace with your dataset
features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment']]  # Adjust columns as needed
target = data['Close']  # The target variable

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)
scaled_target = scaler.fit_transform(target.values.reshape(-1, 1))

# Prepare data for LSTM/GRU
def create_dataset(features, target, time_step=30):
    X, y = [], []
    for i in range(len(features) - time_step - 1):
        X.append(features[i:(i + time_step)])
        y.append(target[i + time_step])
    return np.array(X), np.array(y)

time_step = 30
X, y = create_dataset(scaled_features, scaled_target, time_step)

# Split data into training and test sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build and train LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
lstm_history = lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Build and train GRU model
def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(units=50, return_sequences=True, input_shape=input_shape))
    model.add(GRU(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

gru_model = build_gru_model((X_train.shape[1], X_train.shape[2]))
gru_history = gru_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Predict and evaluate
lstm_predictions = lstm_model.predict(X_test)
gru_predictions = gru_model.predict(X_test)

# Inverse transform the predictions and true values
lstm_predictions = scaler.inverse_transform(lstm_predictions)
gru_predictions = scaler.inverse_transform(gru_predictions)
y_test = scaler.inverse_transform(y_test)

lstm_mse = mean_squared_error(y_test, lstm_predictions)
gru_mse = mean_squared_error(y_test, gru_predictions)

print(f'LSTM Mean Squared Error: {lstm_mse}')
print(f'GRU Mean Squared Error: {gru_mse}')

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(y_test, label='True Prices', color='blue')
plt.plot(lstm_predictions, label='LSTM Predictions', color='red')
plt.plot(gru_predictions, label='GRU Predictions', color='green')
plt.legend()
plt.title('Stock Price Predictions')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.show()