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
from lifelines import RandomSurvivalForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import concordance_index
import matplotlib.pyplot as plt
from lifelines.datasets import load_rossi

#Load/Copy the dataset

data = pd.read_excel("/Assignment 4/DATA/RADCURE_Clinical_v04_20241219.xlsx")

data.head()