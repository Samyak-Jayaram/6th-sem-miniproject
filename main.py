import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import xgboost as xgb
import matplotlib.pyplot as plt


file_path = 'correct_cleaned_china_data.csv'
data = pd.read_csv(file_path)


data.dropna(inplace=True)


X = data.drop(columns=['AC'])
y = data['AC']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))


model = Sequential()
model.add(LSTM(50, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model.add(Dense(1)) 


model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=1)


y_pred_lstm = model.predict(X_test_lstm).flatten()


X_with_lstm = np.concatenate((X_test, y_pred_lstm[:, np.newaxis]), axis=1)


xgb_reg = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, random_state=42)


xgb_reg.fit(X_with_lstm, y_test)


y_pred_xgb = xgb_reg.predict(X_with_lstm)


r2_lstm_xgb = r2_score(y_test, y_pred_xgb)
print("R² score for LSTM + XGBoost:", r2_lstm_xgb)
    
print("Actual AC and Predicted AC for the first 20 rows:")
print("  Actual AC   |   Predicted AC")
print("----------------------------------")
for i in range(20):
    print(f"{y_test.iloc[i]:12.2f} | {y_pred_xgb[i]:14.2f}")

results = pd.DataFrame({
    'Actual': y_test.reset_index(drop=True),
    'LSTM_Predictions': y_pred_lstm,
    'LSTM_XGBoost_Predictions': y_pred_xgb
})


plt.figure(figsize=(14, 7))


plt.plot(results['Actual'], label='Actual', color='black')


plt.plot(results['LSTM_Predictions'], label='LSTM Predictions', color='blue', linestyle='dotted')


plt.plot(results['LSTM_XGBoost_Predictions'], label='LSTM + XGBoost Predictions', color='red', linestyle='dashed')


plt.title('Model Predictions vs Actual Values')
plt.xlabel('Samples')
plt.ylabel('Adoption Cost')
plt.legend()


plt.show()


plt.figure(figsize=(14, 7))


plt.subplot(1, 2, 1)
plt.scatter(results['Actual'], results['LSTM_Predictions'], alpha=0.5, color='blue')
plt.plot([results['Actual'].min(), results['Actual'].max()],
         [results['Actual'].min(), results['Actual'].max()],
         'k--', lw=2)
plt.title('LSTM Predictions vs Actual Values')
plt.xlabel('Actual Adoption Cost')
plt.ylabel('LSTM Predicted Adoption Cost')


plt.subplot(1, 2, 2)
plt.scatter(results['Actual'], results['LSTM_XGBoost_Predictions'], alpha=0.5, color='red')
plt.plot([results['Actual'].min(), results['Actual'].max()],
         [results['Actual'].min(), results['Actual'].max()],
         'k--', lw=2)
plt.title('LSTM + XGBoost Predictions vs Actual Values')
plt.xlabel('Actual Adoption Cost')
plt.ylabel('LSTM + XGBoost Predicted Adoption Cost')


plt.tight_layout()
plt.show()
