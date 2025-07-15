"""Part 3: Build and Evaluate Models
1. Kaplan-Meier Analysis
● Generate Kaplan-Meier survival curves for at least two distinct groups (e.g., treatment type, age group, or tumor stage), ensuring each group has its own plot.
● For each plot, conduct a log-rank test to compare survival diff erences between the groups.
2. Cox Proportional Hazards Regression
● Perform a Cox regression analysis, including at least three covariates.
● Validate the proportional hazards assumption.
3. Random Survival Forests (RSF)
● Build a Random Survival Forest model to predict survival.
● Perform variable importance analysis to identify the most predictive factors.
● Compare the model’s concordance index (C-index) with that of Cox regression."""

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split
import openpyxl as px
import matplotlib.pyplot as plt
from lifelines.datasets import load_rossi



#Load/Copy the dataset

data = pd.read_excel("d:/VSCode ProjectsRepos/BINF-5007-Materials/Assignment 4/DATA/RADCURE_Clinical_v04_20241219.xlsx")
data.to_csv("d:/VSCode ProjectsRepos/BINF-5007-Materials/Assignment 4/DATA/RADCURE_Clinical_v04_20241219.csv", index=False)

print(data.head())

#get columns
columns = data.columns.tolist()
print("Columns in the dataset:", columns)
#print data summary
print(data.describe())
#print number of rows 
print("Number of rows:", data.shape[0])

#find sum of nan values in each column
nan_counts = data.isnull().sum()
#print("Number of NaN values in each column:")
#print(nan_counts)

#find unique values in path column
uniques = data['Tx Modality'].unique()
print("Unique values in 'Tx Modality' column:")
print(uniques)

def clean_data(data):
    """    Cleans the dataset by removing rows with NaN values in critical columns, binary encode key categorical variables"""

    # Drop rows with any NaN values
    data_subset = data.dropna(subset=["Length FU","Status","Last FU","Fx","Dose","RT Start","Chemo","Tx Modality","Path","Age","Sex"])

    #encode Chemo and Status columns as binary and rename columns to make them more readable
    data_subset['Chemo'] = data_subset['Chemo'].map({'Yes': 1, 'none': 2})
    data_subset['Status'] = data_subset['Status'].map({'Alive': 1, 'Dead': 2})
    data_subset['Tx Modality'] = data_subset['Tx Modality'].map({
        'RT alone': 1,
        'ChemoRT': 2,
        'RT + EGFRI': 3,
        'ChemoRT ': 2,
        'Postop RT alone': 4
    })

    data_subset.rename(columns={
        "Length FU": "FollowUp_Length",
        "Status": "Survival_Status",
        "Last FU": "Last_FollowUp",
        "Fx": "Fractions",
        "Dose": "Radiation_Dose",
        "RT Start": "Radiation_Start",
        "Chemo": "Chemotherapy",
        "Tx Modality": "Treatment_Type",
        "Path": "Pathology",
        "Age": "Patient_Age",
        "Sex": "Patient_Sex",
        "Stage": "Tumor_Stage"
    }, inplace=True)

    #print("Number of NaN values in each column of subset:")
    #print(data_subset.isnull().sum())

#ensure that the 'FollowUp_Length' and Survival_Status column is numeric
    data_subset['FollowUp_Length'] = pd.to_numeric(data_subset['FollowUp_Length'], errors='coerce')
    data_subset['Survival_Status'] = pd.to_numeric(data_subset['Survival_Status'], errors='coerce')

    return data_subset


data_cleaned = clean_data(data)
#save cleaned data to a new excel file
data_cleaned.to_csv("d:/VSCode ProjectsRepos/BINF-5007-Materials/Assignment 4/DATA/RADCURE_Cleaned.csv", index=False)
#print(data_cleaned.head())
#print(data_cleaned["Length FU"].isnull().sum())

#check if followup length and survival status are numeric
#print("FollowUp_Length is numeric:", pd.api.types.is_numeric_dtype(data_cleaned['FollowUp_Length']))
#print("Survival_Status is numeric:", pd.api.types.is_numeric_dtype(data_cleaned['Survival_Status']))
#print("first 15 rows of treatment type:", data_cleaned['Treatment_Type'].head(15))



def data_subset_cox(data):
    """Returns a subset of the data that contains only the columns selected for Cox regression analysis."""
    # Include all columns needed for Cox regression including those for encoding
    cox_data = data[["FollowUp_Length", "Survival_Status", "Treatment_Type", "Patient_Age", 
                      "Chemotherapy"]].copy() 

   
    
    
    # One-hot encode categorical variables
    cox_data = pd.get_dummies(cox_data, columns=["Treatment_Type", "Chemotherapy"],  
                             drop_first=True)

    # Keep essential columns
    essential_cols = ['FollowUp_Length', 'Survival_Status', 'Patient_Age']
    dummy_cols = [col for col in cox_data.columns if any(prefix in col for prefix in 
                 ['Treatment_Type_', 'Chemotherapy_'])]
    
    final_cols = essential_cols + dummy_cols
    cox_data = cox_data[final_cols]

    # Make sure there are no NaN values
    cox_data = cox_data.dropna()    


   # print("Cox regression data shape:", cox_data.shape)
    #print("Variables included:")
    #print([col for col in cox_data.columns if col not in ['FollowUp_Length', 'Survival_Status']])
    
    return cox_data

data_cox = data_subset_cox(data_cleaned)


def data_subset_rf(data):
    """Prepare the data for Random Survival Forests by selecting relevant columns and encoding categorical variables."""
    # Include all columns needed for Random Survival Forests
    rf_data = data.copy()

    #remove cols that have more than 99% NaN values
    rf_data = rf_data.loc[:, rf_data.isnull().mean() < 0.99]
    # remove rows with NaN values in all columns that are essential for RF analysis
    rf_data = rf_data.dropna(subset=['patient_id', 'Age', 'Sex', 'Smoking Status', 'T', 'N', 'M ', 'Stage', 'Path', 'HPV', 'Tx Modality', 'Chemo', 'RT Start', 'Dose', 'Fx', 'Last FU', 'Status', 'Length FU'])

    # Encode categorical variables
    rf_data['Smoking Status'] = rf_data['Smoking Status'].map({'Never': 0, 'Former': 1, 'Current': 2})

    rf_data['Sex'] = rf_data['Sex'].map({'Male': 0, 'Female': 1})

    rf_data['Path'] = rf_data['Path'].map({'Positive': 1, 'Negative': 0})
    rf_data['HPV'] = rf_data['HPV'].map({'Positive': 1, 'Negative': 0})
    rf_data['Tx Modality'] = rf_data['Tx Modality'].map({
        'RT alone': 1,
        'ChemoRT': 2,
        'RT + EGFRI': 3,
        'ChemoRT ': 2,
        'Postop RT alone': 4
    })
    rf_data['Chemo'] = rf_data['Chemo'].map({'Yes': 1, 'none': 2})
    rf_data['Status'] = rf_data['Status'].map({'Alive': 1, 'Dead': 2})

    #subset the data to only include columns that are essential for RF analysis and numeric for use with Sklearn, remove patient_id, followup length, and survival status
    rf_data_subset = rf_data[['Age', 'Sex', 'Smoking Status', 'Stage',
                                'Path', 'HPV', 'Tx Modality', 'Chemo', 'Dose']]

    return rf_data, rf_data_subset

