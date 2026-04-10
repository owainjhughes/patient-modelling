import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

def removeConstantAndIdColumns(df, id_threshold=0.95):
    cols_before = set(df.columns)
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df.drop(columns=constant_cols, inplace=True)

    # Columns where almost every value is unique are likely row IDs
    id_cols = [
        c for c in df.columns
        if df[c].nunique() / len(df) > id_threshold
        and df[c].dtype in ['int64', 'object']
    ]
    df.drop(columns=id_cols, inplace=True)

    removed = cols_before - set(df.columns)
    if removed:
        print(f"Removed constant/ID columns: {removed}")
    print(f"Columns retained ({len(df.columns)}): {list(df.columns)}")


def detectAndParseDateColumns(df):
    for col in df.select_dtypes(include='object').columns:
        sample = df[col].dropna().head(100)
        try:
            parsed = pd.to_datetime(sample, infer_datetime_format=True, errors='raise')
            if len(parsed) / len(sample) >= 0.8:  # require 80% parseable before converting
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                print(f"Parsed '{col}' as datetime.")
        except Exception:
            pass


def handleMissingValues(df, missing_threshold=0.5):
    print(f"\nMissing values per column (before):\n{df.isnull().sum()}")
    print(f"Shape before handling missing values: {df.shape}")

    high_null = [c for c in df.columns if df[c].isnull().mean() > missing_threshold]
    if high_null:
        df.drop(columns=high_null, inplace=True)
        print(f"Dropped high-null columns (>{missing_threshold*100:.0f}% missing): {high_null}")

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            median_ts = pd.to_datetime(df[col].dropna().astype(np.int64).median(), unit='ns')
            df[col].fillna(median_ts, inplace=True)
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)
        else:
            mode = df[col].mode()
            if not mode.empty:
                df[col].fillna(mode[0], inplace=True)

    print(f"\nMissing values per column (after):\n{df.isnull().sum()}")
    print(f"Shape after handling missing values: {df.shape}")


def outlierDetection(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if numeric_cols.empty:
        print("No numeric columns found for outlier detection.")
        return df

    print(f"\nNumeric columns for outlier detection: {list(numeric_cols)}")
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    mask = ~((df[numeric_cols] < (Q1 - 1.5 * IQR)) |
             (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    df_clean = df[mask].copy()
    print(f"Shape after removing outliers: {df_clean.shape} "
          f"(removed {len(df) - len(df_clean)} rows)")
    return df_clean


def normalizeNumericColumns(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # Skip low-cardinality integer columns — they are almost certainly labels/targets (hopefully)
    cols_to_scale = [
        c for c in numeric_cols
        if not (df[c].dtype == 'int64' and df[c].nunique() <= 20)
    ]
    if not cols_to_scale:
        print("No continuous numeric columns to normalise.")
        return

    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    print(f"\nNormalised columns: {cols_to_scale}")
    skipped = [c for c in numeric_cols if c not in cols_to_scale]
    if skipped:
        print(f"Skipped (low-cardinality integer): {skipped}")


def cleanse_and_visualise(input_csv, output_csv=None):
    print(f"\n{'='*60}")
    print(f"Loading: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")

    print(f"\n{'-'*40}\nStep 1: Remove constant / ID columns")
    removeConstantAndIdColumns(df)

    print(f"\n{'-'*40}\nStep 2: Detect and parse date columns")
    detectAndParseDateColumns(df)

    print(f"\n{'-'*40}\nStep 3: Handle missing values")
    handleMissingValues(df)

    print(f"\n{'-'*40}\nStep 4: Outlier detection and removal")
    df = outlierDetection(df)

    print(f"\n{'-'*40}\nStep 5: Normalise numeric columns")
    normalizeNumericColumns(df)

    if output_csv is None:
        stem = os.path.splitext(os.path.basename(input_csv))[0]
        output_csv = os.path.join(os.path.dirname(input_csv), f"{stem}_cleaned.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nCleaned data saved to: {output_csv}")
    print(f"Final shape: {df.shape}")
    print(f"{'='*60}\n")

    return df

if __name__ == '__main__':
    df_cleaned = cleanse_and_visualise(
        input_csv='raw_data.csv',
        output_csv='outputs/cleaned_data.csv',
    )