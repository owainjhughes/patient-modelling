import os

import cleanse
import visualise
import random_forest
import kmeans

INPUT_CSV    = 'raw_data.csv'
OUTPUT_DIR   = 'outputs'
CLEANED_CSV  = os.path.join(OUTPUT_DIR, 'cleaned_data.csv')
TARGET_COL   = None   # None → last column
FEATURE_COLS = None   # None → all numeric non-target columns
KMEANS_K     = None   # None → auto-select via silhouette score


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*60 + "\nSTEP 1 — Data Cleansing\n" + "="*60)
    cleanse.cleanse_and_visualise(INPUT_CSV, output_csv=CLEANED_CSV)

    print("\n" + "="*60 + "\nSTEP 2 — Visualisation\n" + "="*60)
    visualise.visualise_csv(CLEANED_CSV, output_dir=OUTPUT_DIR)

    print("\n" + "="*60 + "\nSTEP 3 — Random Forest & Logistic Regression\n" + "="*60)
    random_forest.run_random_forest(
        CLEANED_CSV,
        target_col=TARGET_COL,
        feature_cols=FEATURE_COLS,
        output_dir=OUTPUT_DIR,
    )

    print("\n" + "="*60 + "\nSTEP 4 — KMeans Clustering\n" + "="*60)
    kmeans.run_kmeans(
        CLEANED_CSV,
        feature_cols=FEATURE_COLS,
        k=KMEANS_K,
        output_dir=OUTPUT_DIR,
    )

    print("\nPipeline complete. All outputs written to:", os.path.abspath(OUTPUT_DIR))


if __name__ == '__main__':
    main()
