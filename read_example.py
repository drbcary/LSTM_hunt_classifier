import pandas as pd
import os

filepath = os.path.join("example_files", "CricketHuntTimes_PV01.xlsx")
print(f"Reading: {filepath}")

# Try reading all sheets
xls = pd.ExcelFile(filepath)
print(f"Sheet names: {xls.sheet_names}")

for sheet in xls.sheet_names:
    print(f"\n=== Sheet: {sheet} ===")
    df = pd.read_excel(filepath, sheet_name=sheet)
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    print(f"Dtypes:\n{df.dtypes}")
    print(f"Head (20 rows):\n{df.head(20).to_string()}")
    print(f"Tail (5 rows):\n{df.tail(5).to_string()}")
