import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import streamlit as st
import plotly.express as px
import pickle

url = "https://raw.githubusercontent.com/JohannaViktor/streamlit_practical/refs/heads/main/global_development_data.csv"
data = pd.read_csv(url)


def TrainRandomForest(df):
    # Select features and target
    features = ['GDP per capita', 'headcount_ratio_upper_mid_income_povline', 'year']
    target = 'Life Expectancy (IHME)'

    # Use only the required features    
    X = df[features]
    y = df[target]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    model = RandomForestRegressor(random_state=42)

    # Define hyperparameters for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model from GridSearchCV
    best_model = grid_search.best_estimator_

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Save the trained model to a file
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(best_model, model_file)

    print("Model saved as 'model.pkl'")


def model_predict(model, gdp_per_capita, headcount_ratio, year, features, true_value=None):
    # Create a DataFrame from the provided feature values
    input_data = pd.DataFrame({
        'GDP per capita': [gdp_per_capita],
        'headcount_ratio_upper_mid_income_povline': [headcount_ratio],
        'year': [year]
    })
    
    # Predict life expectancy using the model
    predicted_value = model.predict(input_data[features])[0]
    
    # If true value is provided, calculate MSE
    mse = None
    if true_value is not None:
        mse = mean_squared_error([true_value], [predicted_value])
    
    # Get the feature importance
    feature_importances = model.feature_importances_

    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=True)

    # Return prediction, MSE, and feature importance
    return predicted_value, mse, importance_df