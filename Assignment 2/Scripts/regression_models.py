from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LogisticRegression
#from sklearn.preprocessing import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import data_preprocessor2 as dp

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

    df_cleaned = df.loc[:, df.isnull().mean()<0.50]
    df_cleaned = df_cleaned.dropna() 
    
     # Remove rows with NaN values
    unique_indices = df.drop_duplicates().index
    df_cleaned = df.iloc[unique_indices]

    return df_cleaned

# print("Cleaning data...")
# heart_disease_cleaned = clean_data(heart_disease)
# print(heart_disease_cleaned.isnull().sum())

def elastic_net_regression(df, target_col="chol", alpha=0.1, l1_ratio=0.5):
    """
    Performs Elastic Net regression on the provided data.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing features and target.
    target_col (str): The name of the target column.
    alpha (float): Regularization strength.
    l1_ratio (float): Mix ratio of L1 and L2 regularization.
    
    Returns:
    tuple: Model, predictions, mean squared error, R-squared, best alpha, best l1_ratio
    """
    df['chol'] = df['chol'].fillna(df['chol'].mean())
    #drop missing values in the target column
    df = df.dropna(subset=[target_col])  # Ensure the target column has no NaN values
    X = df.drop(columns=[target_col]) 

    y = df[target_col]

    #one-hot encoding categorical variables
    X = pd.get_dummies(X, drop_first=True)  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    r2 = r2_score(y_test, predictions)
    print(f"R-squared: {r2}")

    alphas = [0.1, 0.5, 1.0, 5.0, 10.0]  # List of alphas to try
    l1_ratios = [0.1, 0.5, 0.7, 0.9]  # List of l1_ratios to try
     # Compute and plot heatmaps for R2 and MSE for each l1_ratio and alpha combination
     
    df_results = []
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            df_results.append({'alpha': alpha, 'l1_ratio': l1_ratio, 'MSE': mse, 'R2': r2})

    results_df = pd.DataFrame(df_results)
    mse_pivot = results_df.pivot(index='l1_ratio', columns='alpha', values='MSE')
    r2_pivot = results_df.pivot(index='l1_ratio', columns='alpha', values='R2')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(mse_pivot, annot=True, fmt='.2f', cmap='viridis', cbar_kws={'label': 'MSE'})
    plt.title('MSE Heatmap')
    plt.xlabel('alpha')
    plt.ylabel('l1_ratio')

    plt.subplot(1, 2, 2)
    sns.heatmap(r2_pivot, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'label': 'R2'})
    plt.title('R2 Heatmap')
    plt.xlabel('alpha')
    plt.ylabel('l1_ratio')

    plt.tight_layout()
    plt.show()

    #Checking for the best alpha and l1_ratio using ElasticNetCV  
    elastic_net_cv = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=5)
    elastic_net_cv.fit(X_train, y_train) 
    best_alpha = elastic_net_cv.alpha_
    best_l1_ratio = elastic_net_cv.l1_ratio_  
    print(f"Best alpha: {best_alpha}, Best l1_ratio: {best_l1_ratio}") 

    #imporving the model with the best alpha and l1_ratio
    y_pred_cv = elastic_net_cv.predict(X_test)
    mse_cv = mean_squared_error(y_test, y_pred_cv)
    r2_cv = r2_score(y_test, y_pred_cv)
    print(f"Mean Squared Error with CV: {mse_cv}")
    print(f"R-squared with CV: {r2_cv}")

   
  
    return model, predictions, mse, r2, best_alpha, best_l1_ratio
#model, predictions, mse, r2, best_alpha, best_l1_ratio

# print("Performing Elastic Net regression...")
# heart_disease_cleaned = clean_data(heart_disease)   
# elastic_net_regression(heart_disease_cleaned, target_col='chol')


def logistic_regression(df, target_col="num"):
    """
    Performs Logistic Regression on the provided data.
    To predict the presence of heart disease as a binary classification problem. 
    column name = num
    Use accuracy, F1 score, AUROC, and AUPRC as evaluation metrics.
    Use LogisticRegression. Experiment with varying parameters (penalty and solver) 
    and observe their effects on model coefficients and performance.
    
    Tune the hyperparameter n_neighbors (e.g., {1, 5, 10}) and compare its impact on evaluation metrics.
    
    Plot AUROC and AUPRC curves for the model’s best confi guration.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing features and target.
    target_col (str): The name of the target column.
    
    Returns:
    tuple: Model, data, X_test, y_test, metrics, y_pred, y_proba
    """
    data = df.copy()
    # Ensure the target column is binary (0 or 1), assigns 1 to all non-zero values and 0 to zero values
    data.loc[:,target_col] = (data[target_col] != 0).astype(int)

    # Check if the target column is binary
    # If not, raise an error
    if data[target_col].nunique() != 2:
        raise ValueError(f"Target column '{target_col}' must be binary (0 or 1). Found {data[target_col].nunique()} unique values.")
    
    

    object_cols = data.select_dtypes(include=['object', 'category']).columns  #stores all object/character columns
    numeric_cols = [col for col in df.select_dtypes(include=["number"]).columns if col != target_col]   #stores all numeric cols

    # if not object_cols.empty:   #one hot encoding for the catergorial data columns, stores each category in its own col and adds to the dataset
    data = pd.get_dummies(
                        data,
                        columns=object_cols,
                        prefix_sep='_',
                        drop_first=True,
                        dtype='int8'
                        )
    
    # Standardize numeric columns
    # This scales the numeric columns to have mean=0 and variance=1
    data[numeric_cols] = StandardScaler().fit_transform(data[numeric_cols])

    data = data.dropna()                     #ensure there are no NaN values in the dataset

    X = data.drop(columns=[target_col])      # Features are all columns except the target column
    y = data[target_col]                     # Target variable is the specified target column    


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(
        penalty='l1',  # Using l1 penalty
        solver='saga',
        max_iter=1000,  #how many iterations to run the solver
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'average_precision': average_precision_score(y_test, y_proba)
    }

    # Plot AUROC and AUPRC curves
    fpr, tpr, _ = roc_curve(y_test, y_proba)    
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(12, 5)) 
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label='AUROC')   
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Logistic Regression - AUROC Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label='AUPRC')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Logistic Regression - AUPRC Curve')
    plt.legend()

    plt.show()

    return data, model, X_test, y_test, metrics, y_pred , y_proba



# # Clean the data
# heart_disease_cleaned = clean_data(heart_disease)
# heart_disease_cleaned = dp.impute_missing_values(heart_disease_cleaned,strategy='mean',target_col='chol')
# heart_disease_cleaned = dp.remove_outliers(heart_disease_cleaned, target_col='chol')
# heart_disease_cleaned = dp.remove_redundant_features(heart_disease_cleaned)

# #test_logistic_data = logistic_regression(heart_disease_cleaned, target_col='num')

# # Print the metrics
# #print(test_logistic_data[4])

# # plt.scatter(test_logistic_data['age'], test_logistic_data['num'], alpha=0.5)
# # plt.xlabel('Age')
# # plt.ylabel('Heart Disease (num)')
# # plt.title('Age vs Heart Disease')
# # plt.show()


def knn_logistic_regression(df, target_col="num", knn=5):
    """
    Performs K-Nearest Neighbors (KNN) classification on the provided data.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing features and target.
    target_col (str): The name of the target column.
    n_neighbors (int): Number of neighbors to use for KNN.
    
    Returns:
    tuple: Model, data, predictions, metrics, X_test, y_test, y_proba
    """
    

    data = df.copy()
    data.loc[:,target_col] = (data[target_col] != 0).astype(int)  # Ensure the target column is binary (0 or 1), assigns 1 to all non-zero values and 0 to zero values
    
    if data[target_col].nunique() != 2:  # Check if the target column is binary
        raise ValueError(f"Target column '{target_col}' must be binary (0 or 1). Found {data[target_col].nunique()} unique values.")
    
    object_cols = data.select_dtypes(include=['object', 'category']).columns
    numeric_cols = [col for col in df.select_dtypes(include=["number"]).columns if col != target_col]

    # One-hot encoding for categorical variables
    data = pd.get_dummies(
                        data,
                        columns=object_cols,
                        prefix_sep='_',
                        drop_first=True,
                        dtype='int8'
                        )
    
    data[numeric_cols] = StandardScaler().fit_transform(data[numeric_cols])

    data = data.dropna()  # Ensure there are no NaN values in the dataset

    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = KNeighborsClassifier(n_neighbors=knn) #input for the n_neighbors parameter as specified by the user
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'average_precision': average_precision_score(y_test, y_proba)
    }

    # AROC and AUPRC curves
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    k_values = [1, 3, 5, 7, 9, 11, 15]
    results = []

    # Loop through different k values and calculate metrics,storing them in a list, and selecting the best k based on ROC AUC score
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            'k': k,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'average_precision': average_precision_score(y_test, y_proba)
        }
        results.append(metrics)
    

    # Find the best k based on ROC AUC score, and print/store the results to be used in the plotting 
    best_k = max(results, key=lambda x: x['roc_auc'])['k']
    print(f"Best k: {best_k}")
    print("KNN Classification Metrics:")
    for metric, value in metrics.items():
        if metric != 'k':
            print(f"{metric}: {value}")


    
    plt.figure(figsize=(12, 5)) 
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label='AUROC')   
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('KNN Logistic Regression - AUROC Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label='AUPRC')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('KNN Logistic Regression - AUPRC Curve')
    plt.legend()
    plt.show()

    return model,data, y_pred, metrics, X_test, y_test, y_proba




# test_knn_data = knn_logistic_regression(heart_disease_cleaned, target_col='num', knn=)
# # Print the accuracy    
# print(test_knn_data[3])