# Import accuracy score
from sklearn.metrics import accuracy_score, confusion_matrix

# # Function to predict and evaluate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X_test_scaled, y_test):
    
    #Predicting y
    y_pred = model.predict(X_test_scaled)
    
    
    #Ecluating mae, mse, rmse, and r squared
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }