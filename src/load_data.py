#!/usr/bin/env python3
import os
from rich import print
import pandas as pd

import torch
import torch.nn
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from joblib import dump, load


class CustomDataset(Dataset):
    def __init__(self, data, target):
        super(CustomDataset, self).__init__()
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def load_data(data_dir, data_name, split_ratio=0.7, batch_size=16):
    data_file = os.path.join(data_dir, data_name)
    yield_file = os.path.join(data_dir, "yield_data.csv") 
    
    try:
        data = pd.read_csv(data_file)
        yield_data = pd.read_csv(yield_file)
    except Exception as e:
        print(e, "check data path", end="")

    # split dataset
    train_dataset, test_subset, train_yield, test_yield = train_test_split(
        data.values, 
        yield_data.values, 
        test_size=(1 - split_ratio),
        random_state=421
    )

    train_subset, val_subset, train_yield, val_yield = train_test_split(
        train_dataset,
        train_yield,
        test_size=0.2,
        random_state=422,
    )

    # standaridize
    standardize_scaler = StandardScaler() 
    standardize_scaler.fit(train_subset)

    # save standar 
    dump(standardize_scaler, "standardize_scaler.bin")    

    train_subset = standardize_scaler.transform(train_subset)
    val_subset = standardize_scaler.transform(val_subset)
    test_subset = standardize_scaler.transform(test_subset)

    train_subset = CustomDataset(train_subset, train_yield)
    val_subset = CustomDataset(val_subset, val_yield)
    test_subset = CustomDataset(test_subset, test_yield)
    

    train_dataloader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
    )

    val_dataloader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    test_dataloader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    data_dir = os.path.abspath("../notebooks")
    # test = load_data(data_dir, "p4m_data.csv")
    # train_dataloader, val_dataloader, test_dataloader = load_data(data_dir, "planet_data.csv")
    train_dataloader, val_dataloader, test_dataloader = load_data(data_dir, "p4m_data.csv")

    for data, target in train_dataloader:
        print("train data")
        print(f"train data dimension: {data.shape}, target data dimension: {target.shape}")
        break

    for data, target in val_dataloader:
        print("val data")
        print(f"val data dimension: {data.shape}, target data dimension: {target.shape}")
        break

    for data, target in test_dataloader:
        print("test data")
        print(f"test data dimension: {data.shape}, target data dimension: {target.shape}")
        break
   
    print(
        len(train_dataloader.dataset),
        len(val_dataloader.dataset),
        len(test_dataloader.dataset),
    )


     
