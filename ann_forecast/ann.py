import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import seaborn as sns
sns.set()
forecast_range = 8

vix_data = yf.download("^VIX", start="2000-01-01")
inputs = pd.Series(vix_data['Adj Close'].values)
outputs = pd.DataFrame()

for i in inputs.index:
    idx_position = inputs.index.get_loc(i)
    if idx_position + forecast_range < len(inputs):
        future_values = inputs.iloc[idx_position + 1 : idx_position + 1 + forecast_range].values
        if len(future_values) == forecast_range:
            outputs = pd.concat([outputs, pd.DataFrame([future_values], index=[i])])

if not inputs.index.isin(outputs.index).all():
    common_index = inputs.index.intersection(outputs.index)
    inputs = inputs.loc[common_index]
    outputs = outputs.loc[common_index]

def build_network(X_train, y_train):
    model = Sequential([
        Dense(5, input_dim=1, kernel_initializer='uniform', activation='relu'),
        Dense(5, kernel_initializer='uniform', activation='relu'),
        Dense(y_train.shape[1], kernel_initializer='uniform', activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=10, epochs=50)
    return model


X_train = np.array(inputs).reshape(-1, 1)
y_train = np.array(outputs)
network = build_network(X_train, y_train)
first_layer_weights, first_layer_biases = network.layers[0].get_weights()
print("First Layer Weights:\n", first_layer_weights)
print("First Layer Biases:\n", first_layer_biases)
y_pred = network.predict(X_train)

idx = outputs.index[-1]
plt.figure()
actual = [inputs.loc[idx]] + list(outputs.loc[idx])
predicted = [inputs.loc[idx]] + list(y_pred[outputs.index.get_loc(idx)])
plt.plot(range(len(actual)), actual, 'g-', label='Actual')
plt.plot(range(len(predicted)), predicted, 'r--', label='Predicted')
plt.title(f'Time step: {idx}')
plt.legend()
plt.show()