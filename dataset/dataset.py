from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
import os
import numpy as np
import torchvision.transforms as T
import json
sep = os.path.sep

def images_to_tensors(path):
    """
    Converts all images and labels:
    - Downsize images to 1024x512 px
    - Convert labels to the proper format
    - Convert transformed images and labels to torch tensors
    - Save all tensors with pickle, dividing them in train and val folders accordingly to
      the txt files
    Input:
    - path: the root folder of the dataset
    - cuda: if True the function will send tensors to cuda device
    """
    # loading dictionary for mapping classes
    with open(f'{path}{sep}info.json', 'r') as f:
        labels_map = np.array(json.load(f)['label2train'], dtype=int)[:, 1]
    map_values = lambda x: labels_map[x]
    image_to_numpy = lambda image: map_values(np.array(image))
    lbl_transformer = T.Compose([
        T.Resize((512, 1024)),
        T.Lambda(image_to_numpy),
        T.ToTensor()
    ])
    img_transformer = T.Compose([
        T.Resize((512, 1024)),
        T.PILToTensor()
    ])
    output_paths = [
        f'{path}{sep}train{sep}images{sep}',
        f'{path}{sep}val{sep}images{sep}',
        f'{path}{sep}train{sep}labels{sep}',
        f'{path}{sep}val{sep}labels{sep}'
    ]
    for output_path in output_paths:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    for task in ['train', 'val']:
        with open(f'{path}{sep}{task}.txt', 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line[:-1]
            # open image
            image_name = line.split('/')[1]
            img = Image.open(f'{path}{sep}images{sep}{image_name}')
            # downsize image and to tensor
            img_tensor = img_transformer(img).cuda()
            # pickle dump
            image_name = image_name.replace('.png', '.pkl')
            with open(f'{path}{sep}{task}{sep}images{sep}{image_name}', 'wb') as fb:
                pickle.dump(img_tensor, fb)
            # open label
            label_name = line.split('/')[1].replace('leftImg8bit', 'gtFine_labelIds')
            label = Image.open(f'{path}{sep}labels{sep}{label_name}')
            # downsize label, mapping labels, to tensor
            label_tensor = lbl_transformer(label).cuda()
            label_name = label_name.replace('.png', '.pkl')
            # pickle dump
            with open(f'{path}{sep}{task}{sep}labels{sep}{label_name}', 'wb') as fb:
                pickle.dump(label_tensor, fb)


class Cityscapes(Dataset):
    """
    Load a dataset transformed with the function images_to_tensors,
    i.e., two folders, one with all GPU torch tensors representing images and
    one for labels.
    """
    def __init__(self, path, preproc):
        """
        Input:
        - path: the root folder of the dataset
        - converted is True iff the dataset items have been already pre-processed
        """
        self.path = path
        self.preproc_ = preproc

    def __preprocess__(self):
        images_to_tensors(self.path)
        self.preproc_ = True

    def __getitem__(self, index, section, mode):
        """
        Input:
        - index: the index of the item/label to be returned
        - section: string in ['train', 'val']
        - mode: string in ['images', 'labels']
        Output:
        - the cuda tensor associated to that item/label
        """
        if self.preproc_ is not True:
            self.__preprocess__()
        dir = f'{self.path}{sep}{section}{sep}{mode}{sep}'
        name = os.listdir(dir)[index]
        with open(dir+name, 'rb') as f:
            t = pickle.load(f)
        return t

    def __len__(self, section):
        """
        Input:
        - section: string in ['train', 'val']
        Output: the number of items in that section
        """
        return len(os.listdir(f'{self.path}{sep}{section}{sep}images{sep}'))
