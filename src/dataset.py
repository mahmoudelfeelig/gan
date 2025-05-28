import torch
from torch.utils.data import Dataset
import pandas as pd
from joblib import load
from pathlib import Path

class TrafficDataset(Dataset):
    def __init__(self, parquet_path: Path, transform_path: Path):
        self.X = pd.read_parquet(parquet_path).values.astype("float32")
        self.transform = load(transform_path)  # not used here but handy
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx])

def get_loader(batch_size=512, shuffle=True):
    ds = TrafficDataset(
        Path("train_processed.parquet"), Path("preprocess.joblib")
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
