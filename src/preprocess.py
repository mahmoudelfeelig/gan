import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
import joblib

RAW = Path("Dataset.csv")
PROCESSED = Path("train_processed.parquet")  # Save in root, no data/ folder

def split_ip(df, col, prefix):
    octets = df[col].str.split('.', expand=True).astype(int)
    octets.columns = [f"{prefix}_{i+1}" for i in range(4)]
    return octets

def build_pipeline(df):
    cat_cols = ["proto", "conn_state", "history", "label"]
    num_cols = [c for c in df.columns if c not in cat_cols]
    ct = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
         ("num", StandardScaler(), num_cols)]
    )
    return Pipeline([("ct", ct)])

def main():
    df = pd.read_csv(RAW)
    for side in ["orig", "resp"]:
        df = pd.concat(
            [df, split_ip(df, f"id.{side}_h", f"{side}_ip")], axis=1
        )
    df = df.drop(columns=["id.orig_h", "id.resp_h"])

    pipe = build_pipeline(df)
    X = pipe.fit_transform(df)

    joblib.dump(pipe, "preprocess.joblib")
    pd.DataFrame(X.toarray()).to_parquet(PROCESSED)
    print("✅ Preprocessing complete. Output saved to:", PROCESSED)

if __name__ == "__main__":
    main()
