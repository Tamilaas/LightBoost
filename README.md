import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

dataset = pd.read_csv('C:/Users/Lenova/Desktop/human_development_indicators_for_kazakhstan_1.csv')
dataset = dataset.dropna(subset=['Parameter_value'])
dataset = dataset[dataset['Parameter_value'].apply(lambda x: str(x).replace('.', '').isdigit())]  
X = dataset[['Parameter_code']]
y = dataset['Parameter_value'].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae:.2f}')

plt.scatter(X_test, y_test, color='blue', label='Реальные значения')
plt.scatter(X_test, predictions, color='red', label='Предсказанные значения')
plt.xlabel('Parameter_code')
plt.ylabel('Parameter_value')
plt.legend()
plt.show()


