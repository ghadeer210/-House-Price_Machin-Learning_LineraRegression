# Evaluation Model in R2_Score && MSE && MAE

from lib import *
from init_Data import run_init_Data
from train_model import run_train_model

path = r'C:\Users\Ghadeer\Desktop\House Prices ML&Git\DataSet\train.csv'
X_train, X_test, y_train, y_test, X, y = run_init_Data(path, 0.2, 1)
regressor = run_train_model(path, X_train, y_train)

y_pred = regressor.predict(X_test)
print(f'R2_Score: {r2_score(y_test,y_pred) * 100:.2f}%')
print(f'MSE: {mean_squared_error(y_test,y_pred)}')    # MSE = np.mean((pred - y_test)**2)
print(f'MAE: {mean_absolute_error(y_test,y_pred)}') # MAE = np.mean(np.absolute(pred - y_test))