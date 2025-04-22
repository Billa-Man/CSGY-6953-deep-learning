import torch
from torch.utils.data import Dataset

class customDataset(Dataset):
    def __init__(self, images, labels=None, transform=None, is_test=False):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)

        if not self.is_test:
            label = self.labels[idx]
            label = torch.tensor(label).long()
            return image, label
        else:
            return image, idx