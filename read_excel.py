import pandas as pd
import sys

try:
    df = pd.read_excel('example_files/CricketHuntTimes_PV01.xlsx')
    print(df)
except Exception as e:
    print(f"Error reading file: {e}")
