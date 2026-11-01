"""Generates a deliberately messy sales dataset for demonstrating DataPilot AI.

Run once:  python sample_data\\generate_sample_data.py
The fixed random seed means you always get the same file.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # fixed seed -> reproducible data
N = 500

FIRST = ["Aarav", "Priya", "Rohan", "Sneha", "Vikram", "Anjali",
         "Karan", "Neha", "Rahul", "Pooja", "Amit", "Kavya"]
LAST = ["Sharma", "Patel", "Singh", "Gupta", "Mehta", "Joshi",
        "Reddy", "Nair", "Kapoor", "Verma", "Iyer", "Desai"]

PRODUCTS = {
    "Electronics": ["Laptop", "Headphones", "Monitor", "Keyboard"],
    "Furniture": ["Desk", "Chair", "Bookshelf"],
    "Office Supplies": ["Notebook", "Pen Set", "Stapler", "Printer Paper"],
    "Clothing": ["T-Shirt", "Jacket", "Sneakers"],
}
BASE_PRICE = {"Electronics": 300, "Furniture": 200,
              "Office Supplies": 20, "Clothing": 50}

# ---- Build clean base data ------------------------------------------------
categories = rng.choice(list(PRODUCTS), N, p=[0.25, 0.20, 0.35, 0.20])
products = [rng.choice(PRODUCTS[c]) for c in categories]
prices = np.array([round(BASE_PRICE[c] * rng.uniform(0.7, 1.3), 2) for c in categories])

quantity = rng.integers(1, 10, N)
quantity[rng.choice(N, 6, replace=False)] = rng.integers(150, 300, 6)  # OUTLIERS

discount = rng.choice([0, 5, 10, 15, 20], N, p=[0.4, 0.2, 0.2, 0.1, 0.1])
sales = np.round(quantity * prices * (1 - discount / 100), 2)

ages = rng.integers(18, 70, N).astype(float)
ages[rng.choice(N, 3, replace=False)] = [150, 130, 999]  # OUTLIERS (impossible ages)

order_dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, N), unit="D")

df = pd.DataFrame({
    "Order ID": [f"ORD-{1000 + i}" for i in range(N)],
    "Order Date": order_dates.strftime("%Y-%m-%d"),      # dates stored as text
    "Customer Name": [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(N)],
    "Customer Email": [f"customer{i}@example.com" for i in range(N)],  # sensitive column
    "Region": rng.choice(["North", "South", "East", "West"], N, p=[0.3, 0.3, 0.2, 0.2]),
    "Category": categories,
    "Product": products,
    "Quantity": quantity,
    "Unit Price ": prices,                               # note the trailing space
    "Discount": [f"{d}%" for d in discount],             # numbers stored as text
    "Sales": sales,
    "Payment Method": rng.choice(["Credit Card", "Debit Card", "UPI", "Cash", "PayPal"], N),
    "Customer Age": ages,
    "Rating": rng.choice([1, 2, 3, 4, 5], N, p=[0.05, 0.10, 0.20, 0.35, 0.30]).astype(float),
})


# ---- Make it dirty ----------------------------------------------------------
def mess_up(value):
    """Randomly change case or add spaces: 'North' -> 'north', 'NORTH', 'North '."""
    r = rng.random()
    if r < 0.08:
        return value.lower()
    if r < 0.14:
        return value.upper()
    if r < 0.20:
        return value + " "
    return value


df["Region"] = df["Region"].map(mess_up)
df["Payment Method"] = df["Payment Method"].map(mess_up)


def add_missing(column, fraction):
    """Blank out a random fraction of a column's values."""
    mask = rng.random(len(df)) < fraction
    df.loc[mask, column] = np.nan


add_missing("Customer Age", 0.08)
add_missing("Rating", 0.10)
add_missing("Payment Method", 0.05)
add_missing("Discount", 0.03)

# Add 20 exact duplicate rows, then shuffle everything
duplicates = df.sample(20, random_state=1)
df = pd.concat([df, duplicates]).sample(frac=1, random_state=2).reset_index(drop=True)

output_path = Path(__file__).parent / "sample_sales.csv"
df.to_csv(output_path, index=False)
print(f"Created {output_path.name} with {df.shape[0]} rows and {df.shape[1]} columns")