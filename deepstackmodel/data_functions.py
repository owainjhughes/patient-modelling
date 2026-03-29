import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

def removeConstantAndIdColumns(df, id_threshold=0.95):
    cols_before = set(df.columns)
    # Drop constant columns
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df.drop(columns=constant_cols, inplace=True)

    # Drop likely ID columns (almost all values are unique)
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
    """Attempt to parse object columns that look like dates into datetime."""
    for col in df.select_dtypes(include='object').columns:
        sample = df[col].dropna().head(100)
        try:
            parsed = pd.to_datetime(sample, infer_datetime_format=True, errors='raise')
            # Only convert if at least 80% of the sample parsed successfully
            if len(parsed) / len(sample) >= 0.8:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                print(f"Parsed '{col}' as datetime.")
        except Exception:
            pass


def handleMissingValues(df, missing_threshold=0.5):
    print(f"\nMissing values per column (before):\n{df.isnull().sum()}")
    print(f"Shape before handling missing values: {df.shape}")

    # Drop high-null columns
    high_null = [c for c in df.columns if df[c].isnull().mean() > missing_threshold]
    if high_null:
        df.drop(columns=high_null, inplace=True)
        print(f"Dropped high-null columns (>{missing_threshold*100:.0f}% missing): {high_null}")

    # Fill remaining nulls by type
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
    if numeric_cols.empty:
        print("No numeric columns to normalise.")
        return

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(f"\nNormalised columns: {list(numeric_cols)}")
    print(df[numeric_cols].describe().round(3))


def visualise(df, output_dir=None):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    def _save_or_show(name):
        if output_dir:
            path = os.path.join(output_dir, name)
            plt.savefig(path, bbox_inches='tight')
            print(f"Saved: {path}")
            plt.close()
        else:
            plt.show()

    # Distribution of numeric columns
    if numeric_cols:
        n = len(numeric_cols)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.array(axes).flatten()
        for i, col in enumerate(numeric_cols):
            axes[i].hist(df[col].dropna(), bins=30, edgecolor='black')
            axes[i].set_title(col)
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle('Numeric Column Distributions', fontsize=14)
        plt.tight_layout()
        _save_or_show('distributions.png')

    # Correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(max(8, len(numeric_cols)), max(6, len(numeric_cols) - 2)))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        _save_or_show('correlation_heatmap.png')

    # Value counts for categorical columns (top 10 each)
    if categorical_cols:
        n = len(categorical_cols)
        cols = min(2, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows))
        axes = np.array(axes).flatten()
        for i, col in enumerate(categorical_cols):
            top = df[col].value_counts().head(10)
            axes[i].bar(top.index.astype(str), top.values, edgecolor='black')
            axes[i].set_title(col)
            axes[i].set_xlabel('Category')
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle('Categorical Column Value Counts', fontsize=14)
        plt.tight_layout()
        _save_or_show('categorical_counts.png')

    if df.isnull().sum().sum() > 0:
        plt.figure(figsize=(max(10, len(df.columns) // 2), 5))
        sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
        plt.title('Missing Value Map')
        plt.tight_layout()
        _save_or_show('missing_values.png')


# Main pipeline

def cleanse_and_visualise(input_csv, output_csv=None, visualise_output_dir=None):
    """Load, cleanse, and optionally visualise any CSV file.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV file.
    output_csv : str, optional
        Path to save the cleaned CSV. Defaults to '<input_stem>_cleaned.csv'
        in the same directory as the input file.
    visualise_output_dir : str or None, optional
        Directory in which to save visualisation images. If None, plots are
        displayed interactively.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame.
    """
    # Load
    print(f"\n{'='*60}")
    print(f"Loading: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")

    # Visualise raw data (missing values map)
    #visualise(df, output_dir=visualise_output_dir)

    # Clean
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

    # Visualise cleaned data
    #print(f"\n{'-'*40}\nGenerating post-cleaning visualisations")
    visualise(df, output_dir=visualise_output_dir)

    # Save
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
        visualise_output_dir=os.path.dirname(os.path.realpath(__file__))+'/outputs',
    )