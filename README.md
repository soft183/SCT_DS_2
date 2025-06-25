# Titanic Dataset - Data Cleaning and Exploratory Data Analysis (EDA)
# Internship Task 02 - SkillCraft Technology

# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Display settings
pd.set_option('display.max_columns', None)
sns.set(style="darkgrid")

# Step 1: Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Step 2: Preview the data
print("Initial Dataset Preview:\n", df.head())
print("\nDataset Info:\n")
df.info()

# Step 3: Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Step 4: Data Cleaning
# Fill missing 'Age' with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Fill 'Embarked' with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' column due to many missing values
df.drop(columns='Cabin', inplace=True)

# Verify after cleaning
print("\nMissing Values After Cleaning:\n", df.isnull().sum())

# Step 5: Basic Statistics
print("\nStatistical Summary:\n", df.describe())

# Step 6: Exploratory Data Analysis (EDA)

# Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.xticks([0, 1], ['Not Survived', 'Survived'])
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival by Gender')
plt.xticks([0, 1], ['Not Survived', 'Survived'])
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.xticks([0, 1, 2], ['1st Class', '2nd Class', '3rd Class'])
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Step 7: Key Observations
print("\nKey Observations:")
print("- More females survived than males.")
print("- Passengers in 1st class had higher survival rates.")
print("- Younger passengers had slightly better chances of survival.")
print("- Strong correlation between Pclass, Fare, and Survival.")

# End of Task 02
print("\n✅ Task 02 Completed: Data Cleaning & EDA on Titanic Dataset.")
