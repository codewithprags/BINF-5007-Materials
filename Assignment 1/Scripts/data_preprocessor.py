# import all necessary libraries here
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from scipy import stats
import matplotlib.pyplot as plt



#data exploration
messy_data = pd.read_csv('D:/VSCode ProjectsRepos/BINF-5007-Materials/Assignment 1/Data/messy_data.csv')
data_explor = messy_data.copy()
#print(data_explor)
#print(data_explor.head())

#data_explor.info()
#print(data_explor.isnull().sum())



def remove_cols_percent_missing(data, percent_missing=50):
    """
    Remove columns exceeding specified missing value percentage threshold.
    Always preserves the 'target' column.
    
    Parameters:
        data (pd.DataFrame): Input DataFrame
        percent_missing (float): Threshold percentage (0-100) for column removal
        
    Returns:
        pd.DataFrame: DataFrame with high-missing columns removed
    """
    messy_data_missing = data.copy()
    cols_to_remove = []

    # Calculate missing percentages for all columns
    missing_pct = messy_data_missing.isnull().mean() * 100

    for col in messy_data_missing.columns:
        if missing_pct[col] >= percent_missing:
            cols_to_remove.append(col)

    print(f"Columns to be removed ({len(cols_to_remove)}): {cols_to_remove}")
    
    # Only remove columns if any were identified
    if cols_to_remove:
        messy_data_missing = messy_data_missing.drop(columns=cols_to_remove)
    else:
        print(f"No columns exceeded {percent_missing}% missing threshold")
    
    return messy_data_missing


# 1. Impute Missing Values
def impute_missing_values(data, strategy='mean'):
    """
    Fill missing values in the dataset.
    :param data: pandas DataFrame
    :param strategy: str, imputation method ('mean', 'median', 'mode')
    :return: pandas DataFrame
    """
    #TODO: Fill missing values based on the specified strategy
    
    messy_data_impute = data.copy()
    
    for col in messy_data_impute:
        if messy_data_impute[col].isnull().any():
            try:
                if strategy == "mean":
                 messy_data_impute[col].fillna(messy_data_impute[col].mean(), inplace=True)
                elif strategy == "median":
                    messy_data_impute[col].fillna(messy_data_impute[col].median(), inplace=True)
                elif strategy == "mode":
                       messy_data_impute[col].fillna(messy_data_impute[col].mode()[0], inplace=True)  
            except:
                messy_data_impute[col].fillna(messy_data_impute[col].mode()[0], inplace=True)  
    
    return messy_data_impute

# print("this is the test df for imput")
# test_df = impute_missing_values(messy_data,"mean")
# test_df.info()
# print(test_df)
# print(test_df.isnull().sum())


print("_____________________Checking Data for Duplicates_____________________________")
print(data_explor.duplicated().sum())
# 2. Remove Duplicates
def remove_duplicates(data):
    """
    Remove duplicate rows from the dataset.
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    # TODO: Remove duplicate rows

    messy_data_noduplicate = data.copy().drop_duplicates()

    return messy_data_noduplicate

def remove_outliers(data, show_plot= True):
    """Remove numeric outliers (|Z-score| > 3) with safety checks"""
    messy_data_nooutlier = data.copy()
    num_cols = messy_data_nooutlier.select_dtypes(include=[np.number]).columns
    
    if show_plot and not num_cols.empty:
        plt.figure(figsize=(10, 4*len(num_cols)))
        
        for i, col in enumerate(num_cols, 1):
            plt.subplot(len(num_cols), 1, i)
            
            # Safe Z-score calculation with NaN handling
            col_data = messy_data_nooutlier[col].dropna()
            if len(col_data) > 1 :  # Check for variance
                z_score = np.abs(stats.zscore(col_data))
                outliers = np.zeros(len(messy_data_nooutlier), dtype=bool)
                outliers[col_data.index] = z_score > 3
            else:
                outliers = np.zeros(len(messy_data_nooutlier), dtype=bool)

            plt.scatter(messy_data_nooutlier.index, messy_data_nooutlier[col], 
                       c=outliers, cmap='cool', alpha=0.6)
            plt.title(f"Outliers in {col} (Red = Outlier)")
            plt.xlabel("Row Index")
            plt.ylabel(col)
        
        plt.tight_layout()

    if not num_cols.empty:
        # Safe Z-score calculation for all numeric columns
        z_scores = np.abs(stats.zscore(messy_data_nooutlier[num_cols], nan_policy='omit'))
        outlier_mask = np.zeros(len(messy_data_nooutlier), dtype=bool)
        
        if z_scores.ndim == 1:  # Single column case
            valid_mask = ~np.isnan(z_scores)
            outlier_mask[valid_mask] = z_scores[valid_mask] > 3
        else:  # Multiple columns case
            valid_mask = ~np.isnan(z_scores).any(axis=1)
            outlier_mask[valid_mask] = (z_scores[valid_mask] > 3).any(axis=1)
        
        messy_data_nooutlier = messy_data_nooutlier[~outlier_mask]
    
    return messy_data_nooutlier
# 3. Normalize Numerical Data
def normalize_data(data,method='minmax'):
    """Apply normalization to numerical features.
    :param data: pandas DataFrame
    :param method: str, normalization method ('minmax' (default) or 'standard')
    """
    # TODO: Normalize numerical data using Min-Max or Standard scaling

    normal_data = data.copy()

    object_cols = normal_data.select_dtypes(include=['object', 'category'])

    if not object_cols.empty:
        normal_data = pd.get_dummies(
            normal_data,
            columns=object_cols,
            prefix_sep='_',
            drop_first=True,
            dtype='int8'
        )
    

    numeric_cols = normal_data.select_dtypes(include = ["number"]).columns

    if not numeric_cols.empty:
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError("Method must be either 'minmax' or 'standard'")
        
        # Apply scaling to numeric columns
        normal_data[numeric_cols] = scaler.fit_transform(normal_data[numeric_cols])
    
    return normal_data


# 4. Remove Redundant Features   
def remove_redundant_features(data, threshold=0.9):
    """Remove redundant or duplicate columns.
    :param data: pandas DataFrame
    :param threshold: float, correlation threshold
    :return: pandas DataFrame
    """

    messy_noredundant = data.copy()

    corr_matrix = messy_noredundant.select_dtypes(include=['number']).corr().abs()

    cols_drop = set()

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            # If correlation exceeds threshold and column hasn't been marked for removal
            if corr_matrix.iloc[i, j] > threshold and corr_matrix.columns[j] not in cols_drop:
                # Add the column name to drop list
                cols_drop.add(corr_matrix.columns[i])
    
    # Drop the redundant columns
    messy_noredundant = messy_noredundant.drop(columns=cols_drop)
    
    return messy_noredundant
    # TODO: Remove redundant features based on the correlation threshold (HINT: you can use the corr() method)
    pass

#test_df = remove_redundant_features(messy_data,threshold=0.9)
#print(test_df)
# # ---------------------------------------------------


def simple_model(input_data, split_data=True, scale_data=False, print_report=False):
    """
    A simple logistic regression model for target classification.
    Parameters:
    input_data (pd.DataFrame): The input data containing features and the target variable 'target' (assume 'target' is the first column).
    split_data (bool): Whether to split the data into training and testing sets. Default is True.
    scale_data (bool): Whether to scale the features using StandardScaler. Default is False.
    print_report (bool): Whether to print the classification report. Default is False.
    Returns:
    None
    The function performs the following steps:
    1. Removes columns with missing data.
    2. Splits the input data into features and target.
    3. Encodes categorical features using one-hot encoding.
    4. Splits the data into training and testing sets (if split_data is True).
    5. Scales the features using StandardScaler (if scale_data is True).
    6. Instantiates and fits a logistic regression model.
    7. Makes predictions on the test set.
    8. Evaluates the model using accuracy score and classification report.
    9. Prints the accuracy and classification report (if print_report is True).
    """

    # if there's any missing data, remove the columns
    input_data.dropna(inplace=True)

    # split the data into features and target
    target = input_data.copy()[input_data.columns[0]]
    features = input_data.copy()[input_data.columns[1:]]

    # if the column is not numeric, encode it (one-hot)
    for col in features.columns:
        if features[col].dtype == 'object':
            features = pd.concat([features, pd.get_dummies(features[col], prefix=col)], axis=1)
            features.drop(col, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

    if scale_data:
        # scale the data
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)
        
    # instantiate and fit the model
    log_reg = LogisticRegression(random_state=42, max_iter=100, solver='liblinear', penalty='l2', C=1.0)
    log_reg.fit(X_train, y_train)

    # make predictions and evaluate the model
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    
    # if specified, print the classification report
    if print_report:
        print('Classification Report:')
        print(report)
        print('Read more about the classification report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html and https://www.nb-data.com/p/breaking-down-the-classification')
    
    return None