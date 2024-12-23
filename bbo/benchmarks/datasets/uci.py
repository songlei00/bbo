import torch
from torch.utils.data import Dataset
from uci_datasets import Dataset as uci_Dataset, all_datasets


class UCIDataset(Dataset):
    """
    https://github.com/treforevans/uci_datasets/tree/master
    """
    def __init__(
        self,
        dataset: str,
        train: bool = True,
        split: int = 0
    ):
        assert dataset in all_datasets
        data = uci_Dataset(dataset, print_stats=False)
        train_X, train_Y, test_X, test_Y = data.get_split(split=split)
        self.data = torch.from_numpy(train_X if train else test_X)
        self.targets = torch.from_numpy(train_Y if train else test_Y)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    
    def __len__(self):
        return len(self.data)