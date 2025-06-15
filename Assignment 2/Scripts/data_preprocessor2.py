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
    
    Parameters:
        data (pd.DataFrame): Input DataFrame
        percent_missing (float): Threshold percentage (0-100) 
        
    Returns:
        pd.DataFrame: DataFrame with high-missing columns removed
    """

    print("_____________________Removing Columns with High percentage of Missing Values_____________________________")
    messy_data_missing = data.copy()
    cols_to_remove = []

    
    missing_pct = messy_data_missing.isnull().mean() * 100   # Calculate missing percentages for all columns

    for col in messy_data_missing.columns:   # loops though each column, exlcuding the target column and evaluates if the column should be removed based on the percentation
        if col == "target":
            continue
        elif missing_pct[col] >= percent_missing:
            cols_to_remove.append(col)

    print(f"Columns to be removed ({len(cols_to_remove)}): {cols_to_remove}")   
    
  
    if cols_to_remove:# Only remove columns if any were identified
        messy_data_missing = messy_data_missing.drop(columns=cols_to_remove)
    else:
        print(f"No columns exceeded {percent_missing}% missing threshold")
    
    return messy_data_missing
    



def impute_missing_values(data, strategy='mean', target_col='chol'):
    
    """
    Fill missing values in the dataset.
    :param data: pandas DataFrame
    :param strategy: str, imputation method ('mean', 'median', 'mode')
    :return: pandas DataFrame
    """
    #TODO: Fill missing values based on the specified strategy
    print("_____________________Imputing Missing Values_____________________________")
    messy_data_impute = data.copy()
    
    for col in messy_data_impute:  # exlcude the target column
        if col == target_col:  #if the col is the target column then skip it
            continue
    
        else :   #loops though each column and row to find null values , it will fill in the null with the either the mean. median or mode
            if messy_data_impute[col].isnull().any(): 
                if pd.api.types.is_numeric_dtype(messy_data_impute[col]):
                    if strategy == "mean":
                        messy_data_impute[col].fillna(messy_data_impute[col].mean(), inplace=True)   #replace any missing in the column with mean of the col
                    elif strategy == "median":
                        messy_data_impute[col].fillna(messy_data_impute[col].median(), inplace=True)   #replace any missing in the column with median of the col
                    elif strategy == "mode":
                       messy_data_impute[col].fillna(messy_data_impute[col].mode()[0], inplace=True)  #replace any missing in the column with mode of the col
                else:
                    messy_data_impute[col].fillna(messy_data_impute[col].mode()[0], inplace=True)  #if the col has objects,replace any missing in the column with mode of the col
    
    return messy_data_impute

# print("this is the test df for imput")
# test_df = impute_missing_values(messy_data,"mean")
# test_df.info()
# print(test_df)
# print(test_df.isnull().sum())



# 2. Remove Duplicates
def remove_duplicates(data):
    
    print(data_explor.duplicated().sum())
    """
    Remove duplicate rows from the dataset.
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    # TODO: Remove duplicate rows
    print("_____________________Checking Data for Duplicates_____________________________")
    messy_data_noduplicate = data.copy().drop_duplicates()  #drops druplicate rows

    return messy_data_noduplicate



def remove_outliers(data, show_plot=False, target_col=None):
    """
    Remove outliers from the DataFrame using z-score method for numeric columns except the target column.
    """
    from scipy import stats
    import numpy as np
    import pandas as pd
    
    df = data.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Initialize a boolean Series for outliers
    outliers = pd.Series(False, index=df.index)
    
    for col in numeric_cols:
        col_data = df[col]
        z_score = np.abs(stats.zscore(col_data, nan_policy='omit'))
        # Mark as outlier if z-score > 3
        outliers = outliers | (z_score > 3)
    
    # Remove outliers
    df_no_outliers = df.loc[~outliers]
    
    if show_plot:
        import matplotlib.pyplot as plt
        for col in numeric_cols:
            plt.figure()
            df[col].plot(kind='box', title=f'Boxplot of {col} (with outliers)')
            plt.show()
            df_no_outliers[col].plot(kind='box', title=f'Boxplot of {col} (without outliers)')
            plt.show()
    
    return df_no_outliers

# 3. Normalize Numerical Data
def normalize_data(data,method='minmax', target_col='target'):
    """
    Apply normalization to numerical features.
    :param data: pandas DataFrame
    :param method: str, normalization method ('minmax' (default) or 'standard')
    """
    # TODO: Normalize numerical data using Min-Max or Standard scaling
    print("_____________________Encoding and Normalizing the Data_____________________________")
    normal_data = data.copy()

    object_cols = normal_data.select_dtypes(include=['object', 'category']).columns  #stores all object/character columns
    numeric_cols = normal_data.select_dtypes(include = ["number"]).columns   #stores all numeric cols

    # if not object_cols.empty:   #one hot encoding for the catergorial data columns, stores each category in its own col and adds to the dataset
    #     normal_data = pd.get_dummies(
    #         normal_data,
    #         columns=object_cols,
    #         prefix_sep='_',
    #         drop_first=True,
    #         dtype='int8'
    #     )

   

    if target_col in numeric_cols:  #ensures we are leaving out target from the numeric cols, bc altering this will alter our model
        numeric_cols = numeric_cols.drop(target_col)


    if not numeric_cols.empty:  #if the num col is not empty then chose the normalization scaler based on the selection by user
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError("Method must be either 'minmax' or 'standard'") #incase of invald user input
        
        # Apply scaling to numeric columns
        normal_data[numeric_cols] = scaler.fit_transform(normal_data[numeric_cols]) #fits the data based on the scaler method chosen
    
    return normal_data


# 4. Remove Redundant Features   
def remove_redundant_features(data, threshold=0.9):
    
    """Remove redundant or duplicate columns.
    :param data: pandas DataFrame
    :param threshold: float, correlation threshold
    :return: pandas DataFrame
    """
    print("_____________________Removing Redunant Features_____________________________")
    messy_noredundant = data.copy()

    corr_matrix = messy_noredundant.select_dtypes(include=['number']).corr().abs() #creates a correlation matric for only the numeric cols in the data set
    print(corr_matrix)

    cols_drop = set() #store the cols to drop based on the high correlation factors

    for i in range(len(corr_matrix.columns)): #for loop to iterate through each col in the correcation matrix and evaluate if the values are matching the threshold
        for j in range(i):
            # If correlation exceeds threshold and column hasn't been marked for removal
            if corr_matrix.iloc[i, j] > threshold and corr_matrix.columns[j] not in cols_drop:
                # Add the column name to drop list
                cols_drop.add(corr_matrix.columns[i])
    
    # Drop the redundant columns
    print("Redundant Columns to Drop :", cols_drop)
    messy_noredundant = messy_noredundant.drop(columns=cols_drop)
    
    return messy_noredundant
    # TODO: Remove redundant features based on the correlation threshold (HINT: you can use the corr() method)


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