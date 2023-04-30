# -*- coding: utf-8 -*-

#Import the dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df = pd.read_csv('titanic.csv')

# Check the number of rows and columns
print('Number of rows:', df.shape[0])
print('Number of columns:', df.shape[1])

# Check the data types of each column
print(df.dtypes)

# Evaluate whether each column should be selected or not
should_select = {
    'PassengerId': False,
    'Survived': True,
    'Pclass': True,
    'Name': False,
    'Sex': True,
    'Age': True,
    'SibSp': True,
    'Parch': True,
    'Ticket': False,
    'Fare': True,
    'Cabin': False,
    'Embarked': True
}

# Define the pre-processing/feature engineering techniques for each column
preprocessing = {
    'PassengerId': [],
    'Survived': [],
    'Pclass': [],
    'Name': [],
    'Sex': ['one_hot_encoding'],
    'Age': ['imputation', 'binning'],
    'SibSp': [],
    'Parch': [],
    'Ticket': [],
    'Fare': ['scaling'],
    'Cabin': [],
    'Embarked': ['one_hot_encoding']
}

# Perform some exploratory data analysis
sns.countplot(x='Survived', data=df)
plt.show()

sns.countplot(x='Pclass', data=df)
plt.show()

sns.countplot(x='Sex', data=df)
plt.show()

sns.histplot(x='Age', data=df, bins=20)
plt.show()

sns.countplot(x='SibSp', data=df)
plt.show()

sns.countplot(x='Parch', data=df)
plt.show()

sns.histplot(x='Fare', data=df, bins=20)
plt.show()

sns.countplot(x='Embarked', data=df)
plt.show()

# Summarize the findings
summary_table = pd.DataFrame({
    'Column Name': list(should_select.keys()),
    'Should it be selected?': list(should_select.values()),
    'Pre-processing/Feature Engineering Techniques': list(preprocessing.values())
})

print(summary_table)

# Load the Titanic dataset
titanic = pd.read_csv('titanic.csv')

# Analyze each column to determine whether it should be selected for the final prediction
for col in titanic.columns:
    # Analyze PassengerId column
    if col == 'PassengerId':
        titanic = titanic.drop(col, axis=1)
        print("PassengerId column dropped")
    # Analyze Survived column
    elif col == 'Survived':
        print("Survived column selected")
    # Analyze Pclass column
    elif col == 'Pclass':
        print("Pclass column selected")
    # Analyze Name column
    elif col == 'Name':
        titanic = titanic.drop(col, axis=1)
        print("Name column dropped")
    # Analyze Sex column
    elif col == 'Sex':
        print("Sex column selected")
    # Analyze Age column
    elif col == 'Age':
        print("Age column selected")
        # Check for missing values
        if titanic[col].isnull().values.any():
            print("Missing values found in Age column")
            # Handle missing values
            # For example, replace missing values with mean age
            # titanic[col] = titanic[col].fillna(titanic[col].mean())
    # Analyze SibSp column
    elif col == 'SibSp':
        print("SibSp column selected")
    # Analyze Parch column
    elif col == 'Parch':
        print("Parch column selected")
    # Analyze Ticket column
    elif col == 'Ticket':
        titanic = titanic.drop(col, axis=1)
        print("Ticket column dropped")
    # Analyze Fare column
    elif col == 'Fare':
        print("Fare column selected")
        # Check for missing values
        if titanic[col].isnull().values.any():
            print("Missing values found in Fare column")
            # Handle missing values
            # For example, replace missing values with median fare
            # titanic[col] = titanic[col].fillna(titanic[col].median())
    # Analyze Cabin column
    elif col == 'Cabin':
        titanic = titanic.drop(col, axis=1)
        print("Cabin column dropped")