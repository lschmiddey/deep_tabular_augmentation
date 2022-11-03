import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np


def create_datasets(train_data: np.array, train_target: np.array,
                    test_data: np.array, test_target: np.array):
    """Converts NumPy arrays into PyTorch datasets."""

    trn_ds = TensorDataset(
        torch.tensor(train_data).float(),
        torch.tensor(train_target).long())
    tst_ds = TensorDataset(
        torch.tensor(test_data).float(),
        torch.tensor(test_target).long())
    return trn_ds, tst_ds


def create_datasets_with_embs(train_data: np.array, train_target: np.array,
                              test_data: np.array, test_target: np.array,
                              train_embs: np.array, test_embs: np.array):
    """Converts NumPy arrays into PyTorch datasets."""

    trn_ds = TensorDataset(
        torch.tensor(train_data).float(),
        torch.tensor(train_embs).float(),
        torch.tensor(train_target).long())
    tst_ds = TensorDataset(
        torch.tensor(test_data).float(),
        torch.tensor(test_embs).float(),
        torch.tensor(test_target).long())
    return trn_ds, tst_ds


def create_datasets_no_target_var(train_data: np.array, test_data: np.array):
    """Converts NumPy arrays into PyTorch datasets. No need of any dependend Variable"""

    train_target = np.zeros((len(train_data),), dtype=int)
    test_target = np.zeros((len(test_data),), dtype=int)

    trn_ds = TensorDataset(
        torch.tensor(train_data).float(),
        torch.tensor(train_target).long())
    tst_ds = TensorDataset(
        torch.tensor(test_data).float(),
        torch.tensor(test_target).long())
    return trn_ds, tst_ds  
    

def create_loaders(data: TensorDataset, bs=128, jobs=0, device='cpu'):
    """Wraps the datasets returned by create_datasets function with data loaders."""

    trn_ds, tst_ds = data
    trn_dl = DataLoader(trn_ds, batch_size=bs, shuffle=True, num_workers=jobs, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    tst_dl = DataLoader(tst_ds, batch_size=bs, shuffle=False, num_workers=jobs, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

    return trn_dl, tst_dl

class DataBunch():
    def __init__(self, train_dl, test_dl):
        self.train_dl, self.test_dl = train_dl, test_dl

    @property
    def train_ds(self): return self.train_dl.dataset

    @property
    def test_ds(self): return self.test_dl.dataset