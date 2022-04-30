import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

URL = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
MD5 = "c8fa014336c82ac7804f0398fcb19387"
SUBDIR = '2750'



def random_split(dataset, ratio=0.9, random_state=None):
    if random_state is not None:
        state = torch.random.get_rng_state()
        torch.random.manual_seed(random_state)
    n = int(len(dataset) * ratio)
    split = torch.utils.data.random_split(dataset, [n, len(dataset) - n])
    if random_state is not None:
        torch.random.set_rng_state(state)
    return split


class EuroSAT(ImageFolder):
    def __init__(self, root='data', transform=None, target_transform=None):
        self.download(root)
        root = os.path.join(root, SUBDIR)
        super().__init__(root, transform=transform, target_transform=target_transform)

    @staticmethod
    def download(root):
        if not check_integrity(os.path.join(root, "EuroSAT.zip")):
            download_and_extract_archive(URL, root, md5=MD5)
            
        # # Keep first 2000 samples.
        # # Put it outside the downloading, so it can be used for existing dataset.
        # data_path = './data/2750' #root is the folder holding the README.md
        # labels = ['Forest', 'River', 'Highway', 'AnnualCrop', 'SeaLake', 'HerbaceousVegetation', 'Industrial', 'Residential', 'PermanentCrop', 'Pasture']
        # print("Maually balancing classes")
        # for i in range(2001, 3001):
        #     for label in labels:
        #         file_path = data_path + '/' + label + '/' + label + '_' + str(i) + '.jpg'
        #         if os.path.exists(file_path):
        #             os.remove(file_path)
        # print("Balanced")


# Apparently torchvision doesn't have any loader for this so I made one
# Advantage compared to without loader: get "for free" transforms, DataLoader
# (workers), etc
class ImageFiles(Dataset):
    """
    Generic data loader where all paths must be given
    """

    def __init__(self, paths: [str], loader=default_loader, transform=None):
        self.paths = paths
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = self.loader(self.paths[idx])
        if self.transform is not None:
            image = self.transform(image)
        # WARNING -1 indicates no target, it's useful to keep the same interface as torchvision
        return image, -1


def get_eurosat_dataloader(root_folder):
    # EuroSAT dataset

    dataset = EuroSAT(
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        )
    )

    trainval, test_ds = random_split(dataset, 0.9, random_state=42)
    # train_ds, val_ds = random_split(trainval, 0.9, random_state=7)

    train_dl = torch.utils.data.DataLoader(
        trainval,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=100, num_workers=2, pin_memory=True
    )

    return train_dl, test_dl