import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

df = pd.read_csv('kaggle_ex0.csv')

df['Date'] = pd.to_datetime(df['Date'])

df = df.drop('School DBN', axis=1)

df = df.fillna(method='ffill')

df['Date_numeric'] = (df['Date'] - pd.to_datetime('2018-01-01')).dt.days

features = ['Enrolled', 'Absent', 'Present', 'Released']
data = df[features + ['Date_numeric']]

window_size = 10

X = []
y = []

for i in range(len(data) - window_size):
    X.append(data.iloc[i:i + window_size].values)
    y.append(data.iloc[i + window_size][['Enrolled', 'Absent', 'Present', 'Released']].values)

X = np.array(X)
y = np.array(y)

print("Girdi (X) veri şekli:", X.shape)
print("Çıktı (y) veri şekli:", y.shape)

model = Sequential()
model.add(SimpleRNN(64, activation='tanh', input_shape=(window_size, X.shape[2])))
model.add(Dense(4))

model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)

X_test = X[-5:]
y_test_pred = model.predict(X_test)
y_test = y[-5:]


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, y_test_pred)
print(f'Mean Absolute Percentage Error (MAPE): {mape}%')