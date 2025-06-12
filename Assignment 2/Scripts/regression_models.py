from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("D:/VSCode ProjectsRepos/BINF-5007-Materials/Assignment 2/Data/heart_disease_uci(1).csv")
heart_disease = data.copy()

def clean_data(df):
    """
    Cleans the DataFrame by removing rows with NaN values.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to clean.
    
    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    df_cleaned = df.copy()  # Create a copy of the DataFrame to avoid modifying the original


    missing_pct = df_cleaned.isnull().mean() * 100
    percent_missing = 50  # Set the threshold for missing data percentage
    cols_to_remove = []
    for col in df_cleaned.columns:   # loops though each column, exlcuding the target column and evaluates if the column should be removed based on the percentation
        if missing_pct[col] >= percent_missing:
            cols_to_remove.append(col)

    print(f"Columns with more than {percent_missing}% missing values: {cols_to_remove}")

    if cols_to_remove:# Only remove columns if any were identified
        df_cleaned = df_cleaned.drop(columns=cols_to_remove)
        print(f"Columns removed due to exceeding {percent_missing}% missing threshold: {cols_to_remove}")

    df_cleaned = df_cleaned.dropna()  # Remove rows with NaN values
    return df_cleaned

# print("Cleaning data...")
# heart_disease_cleaned = clean_data(heart_disease)
# print(heart_disease_cleaned.isnull().sum())

def elastic_net_regression(df, target_col, alpha=0.1, l1_ratio=0.5):
    """
    Performs Elastic Net regression on the provided data.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing features and target.
    target_col (str): The name of the target column.
    alpha (float): Regularization strength.
    l1_ratio (float): Mix ratio of L1 and L2 regularization.
    
    Returns:
    tuple: Model, predictions, MSE, R-squared score.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    r2 = r2_score(y_test, predictions)
    print(f"R-squared: {r2}")


    #Checking for the best alpha and l1_ratio using ElasticNetCV    
    elastic_net_cv = ElasticNetCV(alphas=[0.1, 0.5, 1.0, 5.0, 10.0], l1_ratio=[0.1, 0.5, 0.7, 0.9], cv=5)
    elastic_net_cv.fit(X_train, y_train) 
    best_alpha = elastic_net_cv.alpha_
    best_l1_ratio = elastic_net_cv.l1_ratio_  
    print(f"Best alpha: {best_alpha}, Best l1_ratio: {best_l1_ratio}") 

    return model, predictions, mse, r2, best_alpha, best_l1_ratio

print("Performing Elastic Net regression...")
heart_disease_cleaned = clean_data(heart_disease)   
model, predictions, mse, r2, best_alpha, best_l1_ratio = elastic_net_regression(heart_disease_cleaned, target_col='chol')