from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import pickle


# Function to train the model
def train_RFmodel(X, y):
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # Train the logistic regression model
    
    model = LinearRegression()
    lrmodel = model.fit(X_train_scaled, y_train)
    
    # Save the trained model
    with open('models/lrmodel.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    with open("models/columns.pkl", "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    return lrmodel, X_test_scaled, y_test
