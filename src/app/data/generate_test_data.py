# Claude generated test data
import numpy as np
import pandas as pd


rng = np.random.default_rng(42)

# --- 1. Small clean dataset (~50 rows, numeric + categorical + target) ---
n = 50
df1 = pd.DataFrame({
    'age':       rng.integers(18, 80, n),
    'income':    rng.integers(20000, 120000, n),
    'score':     rng.uniform(0, 100, n).round(1),
    'region':    rng.choice(['North', 'South', 'East', 'West'], n),
    'purchased': rng.integers(0, 2, n),
})
df1.to_csv('test_small.csv', index=False)
print("test_small.csv:", df1.shape)

# --- 2. Messy dataset (~200 rows, missing values, outliers, date column) ---
n = 200
age    = rng.integers(18, 75, n).astype(float)
salary = rng.integers(25000, 90000, n).astype(float)
salary[[10, 50, 130]] = [500000, -1000, 999999]       # outliers
age[rng.choice(n, size=10, replace=False)]   = np.nan  # missing
salary[rng.choice(n, size=10, replace=False)] = np.nan  # missing

df2 = pd.DataFrame({
    'emp_id':    range(1001, 1001 + n),
    'age':       age,
    'salary':    salary,
    'dept':      rng.choice(['HR', 'Eng', 'Sales', 'Finance'], n),
    'years_exp': rng.integers(0, 40, n),
    'rating':    rng.choice([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0], n),
    'join_date': pd.date_range('2010-01-01', periods=n, freq='ME').strftime('%Y-%m-%d'),
    'left':      rng.integers(0, 2, n),
})
df2.to_csv('test_messy.csv', index=False)
print("test_messy.csv:", df2.shape)

# --- 3. Customer dataset (~300 rows, good for clustering) ---
n = 300
df3 = pd.DataFrame({
    'customer_id':     range(1, n + 1),
    'age':             rng.integers(18, 70, n),
    'annual_income':   rng.integers(15000, 150000, n),
    'spending_score':  rng.integers(1, 100, n),
    'num_purchases':   rng.integers(1, 200, n),
    'avg_order_value': rng.uniform(5, 500, n).round(2),
    'days_since_last': rng.integers(1, 365, n),
    'loyalty_tier':    rng.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], n),
    'churned':         rng.integers(0, 2, n),
})
df3.to_csv('test_customers.csv', index=False)
print("test_customers.csv:", df3.shape)
