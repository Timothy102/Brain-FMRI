import os
import shutil
import torch
import cv2

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import INPUT_SHAPE

class BrainFMRIDataset(Dataset):
    """Brain FMRI dataset."""

    def __init__(self, images_path, masks_path, transform=None, test = False):
        """
        Args:
            images_path (string): Path to the directory containing MRI images.
            masks_path (string): Directory with all the segmentations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        image = cv2.imread(self.images_path[idx])
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_path[idx])
        if self.transform:
            image = self.transform(image)

        return image, mask


def paths(path):
    masks =  [f for f in os.listdir(path) if "_mask" in f]
    images = [f for f in os.listdir(path) if not "_mask" in f]
    return sorted(images), sorted(masks)

def all_paths(path):
    l, m = [], []
    for k in os.listdir(path):
        imgs, masks = paths(path + k)
        for idx in range(len(imgs)):
            l.append(path + k + "/" + imgs[idx])
            m.append(path + k + "/" + masks[idx])
    return l, m


class ReshapeTensor:
    def __init__(self, new_shape) -> None:
        self.new_shape = new_shape
    def __call__(self, img):
        return torch.reshape(img, self.new_shape)

def transformation():
    return transforms.Compose([
            transforms.ToTensor(),
            ReshapeTensor(INPUT_SHAPE),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])