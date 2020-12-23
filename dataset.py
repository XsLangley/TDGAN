import numpy as np
import random
import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import pdb

class ImageDataset(data.Dataset):
    def __init__(self, expr_data_dir, size=128, norm_flag=False, train=True, GRAY=True):

        self.train = train
        self.expr_data_dir = expr_data_dir
        self.size = size
        self.norm_flag = norm_flag
        self.GRAY_FLAG = GRAY

        self.data = np.load(expr_data_dir)

        self.train_data = self.data['tr_x']
        self.train_labels = self.data['tr_y']
        self.test_data = self.data['tt_x']
        self.test_labels = self.data['tt_y']

        self.transform = transforms.Compose([
                            transforms.Resize(144),
                            transforms.RandomCrop(self.size),
                            transforms.RandomHorizontalFlip(),
                            # transforms.Grayscale(),
                            transforms.ToTensor(),
                            ])
        self.transform_test = transforms.Compose([
                              transforms.Resize(144),
                              transforms.CenterCrop(self.size),
                              # transforms.Grayscale(),
                              transforms.ToTensor(),
                              ])

    def __getitem__(self, index):

        if self.train:
            label = self.train_labels[index].astype(np.long)
            image = self.train_data[index]
            image = Image.fromarray(image)
        else:
            label = self.test_labels[index].astype(np.long)
            image = self.test_data[index]
            image = Image.fromarray(image)


        if self.GRAY_FLAG:
            image = image.convert('L')
        else:
            image = image.convert('RGB')

        if self.train:
            image = self.transform(image)
        else:
            image = self.transform_test(image)

        image = np.array(image).astype(np.float32)
        # if self.norm_flag:
        #     image = image / 255
        image = torch.from_numpy(image)

        # if self.GRAY_FLAG:
        #     image = torch.unsqueeze(image, 0)

        return image, label


    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)
