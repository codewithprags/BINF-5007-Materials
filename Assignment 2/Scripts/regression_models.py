from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
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
    df_cleaned = df_cleaned.dropna()  # Remove rows with NaN values

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
   

    return df_cleaned

# print("Cleaning data...")
heart_disease_cleaned = clean_data(heart_disease)
print(heart_disease_cleaned.isnull().sum())