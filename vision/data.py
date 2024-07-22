import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np

class WatermarkedImageDataset(Dataset):
    def __init__(self, image_folder='imgs', prompt_folder='prompts', scheme='invisible-watermark', sample_size=200, transform=None):
        """
        Args:
            image_folder (string): folder with all the images.
            prompt_folder (string): folder with all the prompts.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_folder = image_folder
        self.prompt_folder = prompt_folder
        self.transform = transform
        self.image_filenames = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('original.png')]
        with open(os.path.join(prompt_folder, 'prompts.txt'), 'r') as f:
            self.prompts = f.readlines()
        self.prompts = self.prompts[:sample_size]
        self.scheme = scheme

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        img_name = f'./imgs/{self.scheme}/{idx}-original.png'
        assert img_name in self.image_filenames
        return {'image': img_name, 'prompt': self.prompts[idx]}