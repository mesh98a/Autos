import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error



def display_metrics(y_test,y_test_pred,y_train,y_train_pred):
    mse_train = mean_squared_error(y_train,y_train_pred)
    mse = mean_squared_error(y_test,y_test_pred)

    rmse_train = np.sqrt(mse_train)
    rmse = np.sqrt(mse)

    r2_train = r2_score(y_train, y_train_pred)
    r2 = r2_score(y_test, y_test_pred)

    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
    mape = mean_absolute_percentage_error(y_test, y_test_pred)

    print(f"Train MSE: {round(mse_train,3)} RMSE: {round(rmse_train,3)}, R²: {round(r2_train,3)}, MAPE: {round(mape_train,3)}")
    print("-" * 100)
    print(f"Test MSE:{round(mse,3)} RMSE: {round(rmse,3)}, R²: {round(r2,3)}, MAPE: {round(mape,3)}")
    print("-" * 100)