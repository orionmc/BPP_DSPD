#import section
import warnings
warnings.filterwarnings('ignore')
import sys
#from IPython.display import display



import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imbPipeline

# Set format
pd.options.display.float_format = "{:.4f}".format


def analyse_data(df, target_col):

    print("000000000000000000TEST")

    # Check for null/missing values
    print("\n\nNumer of null values in each column:")
    print(df.isnull().sum())
    
    # Check data types
    print("\n\nData types:")
    print(df.dtypes)

    # Check for duplicate rows and drop them
    duplicate_rows = df[df.duplicated()]
    print("\n\nNumber of duplicate rows: ", duplicate_rows.shape)
    df = df.drop_duplicates()

    #Check for unique values in each column
    print("\n\nNumber of unique values in each column:\n")
    for col in df.columns:
        num_dist_val = len(df[col].unique())
        print(f"{col}: {num_dist_val} distinct values")

    # Remove Unneccessary value
    df = df[df['gender'].isin(['Male', 'Female'])]
    
    #Describe
    print(df.describe())

    #Age distribution histogram    
    plt.hist(df['age'], bins=30, edgecolor='black', alpha=0.7,)
    plt.title('Age Distribution Histogram')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.show()

    # Gender distribution
    sns.countplot(x='gender', data=df, palette="Set2")
    plt.title('Gender Distribution')
    plt.show()

    # BMI distribution
    sns.histplot(df['bmi'], kde=True, bins=50)
    plt.title('BMI Distribution')
    plt.show()

    # Smoking history
    sns.countplot(x='smoking_history', data=df, palette="Set2")
    plt.title('Smoking History')
    plt.show()

    # # Pair plot for feature relationships
    # print("\n\nPair plot for feature relationships:")
    # sns.pairplot(df, hue='diabetes', diag_kind='kde', palette="Set2")
    # plt.show()



    # Split the data into features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

# Simplify the smoking history categories
def smoking_categories(smoking_status):
    if smoking_status in ['never', 'No Info']:
        return 'non-smoker'
    elif smoking_status == 'current':
        return 'current'
    elif smoking_status in ['ever', 'former', 'not current']:
        return 'past_smoker'
# converting categorical data into a numerical
def one_hot_encoding(df, column_name):
    
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    # Drop the original column and append the new dummy columns to the dataframe
    df = pd.concat([df.drop(column_name, axis=1), dummies], axis=1)

    return df

def cor_matrix(df):
    # Compute the correlation matrix
    correlation_matrix = df.corr()
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=True, linewidths=0.5, fmt=".4f")
    plt.title("Correlation Matrix Heatmap")
    plt.show()

    # Create a heatmap of the correlations with the target column
    corr = df.corr()
    target_corr = corr['diabetes'].drop('diabetes')

    # Sort in descending order
    target_corr_sorted = target_corr.sort_values(ascending=False)

    sns.set(font_scale=0.7)
    sns.set_style("whitegrid")
    sns.set_palette("Set2")
    sns.heatmap(target_corr_sorted.to_frame(), cmap="coolwarm", annot=True, fmt=".4f",)
    plt.title('Correlation with Diabetes')
    plt.show()

    return