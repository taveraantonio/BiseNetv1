from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torchvision.transforms as T
import json
import torch


class Cityscapes(Dataset):
    def __init__(self, path, image_size, task, device):
        """
        Inputs:
        - path: the base folder of the dataset
        - image_size: a tuple indicating the resizing for the image in the format (Height, Width)
        - task: a string in ['train', 'val']
        - device: a string in ['cpu', 'cuda']
        """
        self.path = path
        self.image_size = image_size
        self.task = task
        self.device = device
        self.load_labels_map()
        self.load_transformers()
        self.load_paths()

    def load_labels_map(self):
        with open(os.path.join(self.path, 'info.json'), 'r') as f:
            j = json.load(f)
        self.labels_map = np.array(j['label2train'], dtype=np.uint8)[:, 1]
        self.nb_classes = int(j['classes'])
        self.labels_map[self.labels_map == 255] = self.nb_classes

    def load_transformers(self):
        image_to_numpy = lambda image: self.labels_map[np.array(image, dtype=np.uint8)]
        self.lbl_transformer = T.Compose([
            T.Resize((512, 1024)),
            T.Lambda(image_to_numpy),
            T.ToTensor()
        ])
        self.img_transformer = T.Compose([
            T.Resize(self.image_size),
            T.PILToTensor()
        ])

    def load_paths(self):
        format_path = lambda path: path.split('/')[1][:-1]
        with open(os.path.join(self.path, f'{self.task}.txt')) as f:
            self.img_entry_names = [format_path(i) for i in f.readlines()]
            self.lbl_entry_names = [i.replace('leftImg8bit', 'gtFine_labelIds') for i in self.img_entry_names]
    
    def __getitem__(self, index):
        img_file_name = self.img_entry_names[index]
        lbl_file_name = self.lbl_entry_names[index]
        img_path = os.path.join(self.path, 'images', img_file_name)
        lbl_path = os.path.join(self.path, 'labels', lbl_file_name)
        img = self.img_transformer(Image.open(img_path)).to(torch.float32).to(self.device)
        lbl = self.lbl_transformer(Image.open(lbl_path)).to(torch.float32).to(self.device)[0]
        return img, lbl

    def __len__(self):
        return len(self.img_entry_names)
