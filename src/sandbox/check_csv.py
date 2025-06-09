"""
    poetry run python -m src.sandbox.check_csv
"""

import pandas as pd

CSV_NAME = "./result/test_outputs.csv"
# CSV_NAME = "./data/processed_data.csv"

if __name__ == "__main__":
    df = pd.read_csv(CSV_NAME)

    
    print(f"columns : {df.columns}")
    print(f"rows: {len(df)}")

    print("========= head ============")
    df = df.head(2)
    print(df)

    for col in df.columns:
        print(f"========= col = {col} =======")
        print(df[col])