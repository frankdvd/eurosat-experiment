import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
from sklearn.model_selection import train_test_split


class CustomImageDataset(Dataset):
    def __init__(self, df):
        self.df = torch.tensor(df.values).to(torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df[idx][:-1], self.df[idx][-1].to(torch.int64)



def get_eurosat_features_dataloader(path):

    data = pd.read_csv(path)
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_dl = torch.utils.data.DataLoader(
        CustomImageDataset(train_data),
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_dl = torch.utils.data.DataLoader(
        CustomImageDataset(test_data), batch_size=100, num_workers=2, pin_memory=True
    )

    return train_dl, test_dl




