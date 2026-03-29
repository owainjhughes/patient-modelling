import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def _save_or_show(output_dir, filename):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved: {path}")
        plt.close()
    else:
        plt.show()


def _numeric_cols(df):
    cols = df.select_dtypes(include=['float64', 'int64']).columns
    return [c for c in cols if df[c].nunique() > 1]


def _categorical_cols(df):
    cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    return [c for c in cols if 1 < df[c].nunique() <= 30]


def _sample(df, n=5000):
    return df if len(df) <= n else df.sample(n, random_state=42)


def _subplots_grid(n, ncols=3):
    """Return (fig, axes_flat) for n subplots arranged in a grid."""
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten()
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    return fig, axes_flat


def numericDistributions(df, num_cols, output_dir=None):
    data = _sample(df)
    fig, axes = _subplots_grid(len(num_cols))
    for ax, col in zip(axes, num_cols):
        col_data = data[col].dropna()
        ax.hist(col_data, bins=40, density=True, alpha=0.7, color='steelblue', edgecolor='none')
        ax.axvline(col_data.mean(), color='b', linestyle='--', linewidth=1, label=f'Mean {col_data.mean():.2f}')
        ax.axvline(col_data.median(), color='g', linestyle='--', linewidth=1, label=f'Med {col_data.median():.2f}')
        ax.set_title(col)
        ax.legend(fontsize=7)
        ax.grid(True, axis='y', alpha=0.3)
    plt.suptitle('Numeric Distributions', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save_or_show(output_dir, 'numeric_distributions.png')


def categoricalCounts(df, cat_cols, output_dir=None):
    """One figure with a count-bar panel per categorical column."""
    data = _sample(df)
    fig, axes = _subplots_grid(len(cat_cols))
    total = len(data)
    for ax, col in zip(axes, cat_cols):
        order = data[col].value_counts().index[:15]  # cap at 15 categories per panel
        counts = data[col].value_counts()[order]
        ax.bar(range(len(order)), counts.values, color='steelblue', edgecolor='none')
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([str(v) for v in order], rotation=45, ha='right', fontsize=7)
        for i, v in enumerate(counts.values):
            ax.text(i, v, f'{v/total*100:.1f}%', ha='center', va='bottom', fontsize=6)
        ax.set_title(col)
        ax.grid(True, axis='y', alpha=0.3)
    plt.suptitle('Categorical Distributions', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save_or_show(output_dir, 'categorical_distributions.png')


def boxPlotGrid(df, cat_col, num_cols, output_dir=None):
    """One figure with box-plot panels for each numeric col split by cat_col."""
    cols = num_cols[:6]  # cap at 6 numeric cols per figure
    data = _sample(df)
    fig, axes = _subplots_grid(len(cols))
    for ax, num in zip(axes, cols):
        order = sorted(data[cat_col].dropna().unique())
        groups = [data.loc[data[cat_col] == v, num].dropna() for v in order]
        ax.boxplot(groups, labels=[str(v) for v in order], patch_artist=True)
        ax.set_title(f'{num} by {cat_col}')
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.grid(True, axis='y', alpha=0.3)
    plt.suptitle(f'Box Plots — split by {cat_col}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save_or_show(output_dir, f'boxplots_by_{cat_col}.png')


def correlationMatrix(df, cols, title='Correlation Matrix', output_dir=None):
    corr = df[cols].corr()
    size = max(8, len(cols))
    plt.figure(figsize=(size, size - 2))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, center=0)
    plt.title(title)
    plt.tight_layout()
    _save_or_show(output_dir, f'{title.lower().replace(" ", "_")}.png')


def missingValueMap(df, output_dir=None):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("No missing values — skipping missing value map.")
        return
    plt.figure(figsize=(max(6, len(missing)), 4))
    missing.plot(kind='bar', color='steelblue', edgecolor='none')
    plt.title('Missing Value Counts per Column')
    plt.ylabel('Missing count')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    _save_or_show(output_dir, 'missing_values.png')


def visualise_csv(input_csv, output_dir=None):
    print(f"\n{'='*60}")
    print(f"Visualising: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"Shape: {df.shape}  |  Columns: {list(df.columns)}")

    num_cols = _numeric_cols(df)
    cat_cols = _categorical_cols(df)
    print(f"Numeric columns  ({len(num_cols)}): {num_cols}")
    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

    # 1 plot — missing value map
    missingValueMap(df, output_dir=output_dir)

    # 1 plot — all numeric histograms in one figure
    if num_cols:
        print("  Numeric distributions")
        numericDistributions(df, num_cols, output_dir=output_dir)

    # 1 plot — all categorical counts in one figure
    if cat_cols:
        print("  Categorical distributions")
        categoricalCounts(df, cat_cols, output_dir=output_dir)

    # 1 plot per categorical column — box plots for all numeric cols
    for cat in cat_cols:
        print(f"  Box plots by {cat}")
        boxPlotGrid(df, cat, num_cols, output_dir=output_dir)

    # 1 plot — correlation heatmap
    if len(num_cols) > 1:
        print("  Correlation matrix")
        correlationMatrix(df, num_cols, title='Numeric Correlation Matrix', output_dir=output_dir)

    print(f"{'='*60}\n")
    return df


if __name__ == '__main__':
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'outputs/cleaned_data.csv'
    visualise_csv(csv_path, output_dir='outputs')