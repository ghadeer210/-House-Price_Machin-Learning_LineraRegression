## Training the Simple Linear Regression model on the DataSet initalized

from lib import *
from init_Data import run_init_Data
path = r'C:\Users\Ghadeer\Desktop\House Prices ML&Git\DataSet\train.csv'
X_train, X_test, y_train, y_test, X, y = run_init_Data(path, 0.2, 1)
regressor = LinearRegression()
regressor = regressor.fit(X_train, y_train)

# function to train model 
def run_train_model(path:str, X_train:np.array, y_train:np.array):
    regressor = LinearRegression()
    regressor = regressor.fit(X_train, y_train)
    return regressor