import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Loading the dataset from the CSV file

data_file = '/Users/jigneshsisodiya/Downloads/Dataset.csv'
df_patients = pd.read_csv(data_file)

# Displaying the first few rows to inspect the data

print("First 5 rows of the dataset:")
print(df_patients.head())

# Checking the shape (number of rows and columns)

print(f"Dataset Shape: {df_patients.shape}")

# Step 2: Identifying and Handle Missing Values

# Checking missing values before replacement

print("Missing values in each column (before replacement):")
print(df_patients.isnull().sum())

# Replace '?' with NaN so missing values are properly recognized

df_patients.replace('?', pd.NA, inplace=True)

# Verifying missing values after replacement

print("Missing values in each column (after replacing '?'):")
print(df_patients.isnull().sum())

# Impute Missing Values Separately for Numeric and Categorical Columns

# For numeric columns: fill missing values with the mean.

numeric_cols = df_patients.select_dtypes(include=['float64', 'int64']).columns
df_patients[numeric_cols] = df_patients[numeric_cols].fillna(df_patients[numeric_cols].mean())

# For categorical columns: fill missing values with the mode (most frequent value).

for col in df_patients.select_dtypes(include=['object']).columns:
    df_patients[col] = df_patients[col].fillna(df_patients[col].mode()[0])


# Verify that missing values have been handled

print("Missing values after imputation:")
print(df_patients.isnull().sum())


# Step 3: Outlier Detection and Removal using IQR

# Only numeric columns for outlier detection

numeric_cols = df_patients.select_dtypes(include=['float64', 'int64']).columns
print("Numeric columns for outlier detection:", numeric_cols)

# Calculating the first (Q1) and third (Q3) quartiles
Q1 = df_patients[numeric_cols].quantile(0.25)
Q3 = df_patients[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

# Removing rows that have any outliers (using 1.5 * IQR rule)
df_patients = df_patients[~((df_patients[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                            (df_patients[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Checking the new shape after outlier removal
print(f"Shape after removing outliers: {df_patients.shape}")

# Step 4: Normalise the Features

# We want to normalise numerical columns (e.g., Age, Blood_Pressure, etc.)
# but we don't want to normalize columns that are identifiers or the target variable (e.g., ICU,SEX..).

from sklearn.preprocessing import StandardScaler

# The list of numeric columns

numeric_cols = df_patients.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Removing Index column or identifier

if 'index' in numeric_cols:
    numeric_cols.remove('index')

# Removing the target variable

if 'ICU' in numeric_cols:
    numeric_cols.remove('ICU')

# Removing columns that are categorical 

if 'SEX' in numeric_cols:
    numeric_cols.remove('SEX')
if 'CLASIFFICATION_FINAL' in numeric_cols:
    numeric_cols.remove('CLASIFFICATION_FINAL')

print("Numeric columns to normalize:", numeric_cols)

# A StandardScaler object

scaler = StandardScaler()

# Applying the scaler to the selected numeric columns

df_patients[numeric_cols] = scaler.fit_transform(df_patients[numeric_cols])

# Printing a few rows to check that normalization worked

print("Data after normalization:")
print(df_patients.head())

# Final Check: Print the final shape of the dataset

print(f"Final dataset shape after normalization: {df_patients.shape}")

# Saving the cleaned and normalized dataset to a CSV file for later use

df_patients.to_csv('cleaned_patients.csv', index=False)

# Step 5: Verifying Data Types and Removing Irrelevant Columns

# Checking the data types

print("Data types before adjustment:")
print(df_patients.dtypes)


# Removing irrelevant columns (eg. index)

columns_to_remove = ['index']  # Add any other irrelevant columns to this list if needed.
df_patients.drop(columns=columns_to_remove, inplace=True)

# Verify the data types and the list of columns after removal

print("Data types after adjustment:")
print(df_patients.dtypes)
print("Remaining columns:")
print(df_patients.columns)

# Plot 1: Distribution of the target variable 'ICU'

plt.figure(figsize=(8, 6))
sns.countplot(data=df_patients, x='ICU')
plt.title('Distribution of ICU Cases')
plt.xlabel('ICU (Indicator)')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# Plot 2: Relationship between AGE and ICU

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_patients, x='AGE', y='ICU')
plt.title('ICU Cases vs. Age')
plt.xlabel('Age')
plt.ylabel('ICU (Indicator)')
plt.grid(True)
plt.show()
df_patients.describe()
