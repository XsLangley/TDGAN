import numpy as np
import os
import scipy.misc as sm
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

class DualTrainDataset(data.Dataset):
    def __init__(self, root_dir_F, img_dir_list_F, img_lab_list_F, expr_data_dir, size=128, GRAY_flag=True):
        '''
        construct a dataset including CASIA database and other expression database
        :param root_dir_F: the root directory of the face database
        :param img_dir_list_F: the list that includes all image names and the corresponding folder dir of the face dataset
        :param img_lab_list_F: the list that contains the label corresponding to each image of the face dataset
        :param expr_data_dir: the directory of expression data
        :param size: the size of output image
        :param RGB_flag: the flag to indicate whether load the image in RGB type
        '''

        self.GRAY = GRAY_flag
        self.size = size

        # root directory of face dataset
        self.root_dir_F = root_dir_F
        self.img_dir_list_F = img_dir_list_F
        self.img_lab_list_F = img_lab_list_F

        # transform for face dataset
        self.transform_F = transforms.Compose([
            transforms.Resize([size, size]),
            transforms.ToTensor(),
        ])

        # data of expression dataset
        self.data_E = np.load(expr_data_dir, allow_pickle=True)
        self.expr_data = self.data_E['tr_x']
        self.expr_labels = self.data_E['tr_y']

        # transform for expression dataset
        self.transform_E = transforms.Compose([
            transforms.Resize(144),
            transforms.RandomCrop(self.size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        # the sample number of the dataset which with less images
        self.less_len = min(len(self.img_lab_list_F), len(self.expr_labels))
        if len(self.expr_labels) < len(self.img_lab_list_F):
            self.face_more = True
        else:
            self.face_more = False

    def __getitem__(self, idx):
        # image index for the less image dataset
        img_idx = idx % self.less_len

        if self.face_more:
            # load face image and label
            img_path_F = os.path.join(self.root_dir_F, self.img_dir_list_F[idx])
            img_F = self.img_prepro(img_path_F, self.transform_F, self.GRAY)
            lab_F = self.img_lab_list_F[idx].astype(np.long)

            # load expression image and label
            img_E = self.expr_data[img_idx]
            img_E = Image.fromarray(img_E)
            lab_E = self.expr_labels[img_idx].astype(np.long)
        else:
            # load face image and label
            img_path_F = os.path.join(self.root_dir_F, self.img_dir_list_F[img_idx])
            img_F = self.img_prepro(img_path_F, self.transform_F, self.GRAY)
            lab_F = self.img_lab_list_F[img_idx].astype(np.long)

            # load expression image and label
            img_E = self.expr_data[idx]
            img_E = Image.fromarray(img_E)
            lab_E = self.expr_labels[idx].astype(np.long)

        if self.GRAY:
            img_E = img_E.convert('L')
        else:
            img_E = img_E.convert('RGB')

        img_E = self.transform_E(img_E)
        img_E = np.array(img_E).astype(np.float32)
        img_E = torch.from_numpy(img_E)

        return img_F, lab_F, img_E, lab_E

    def __len__(self):
        return max(len(self.img_lab_list_F), len(self.expr_labels))

    def img_prepro(self, img_path, transform=None, GRAY=True):
        if GRAY:
            img = Image.open(img_path).convert('L')
        else:
            img = Image.open(img_path).convert('RGB')
        if transform is not None:
            img = transform(img)

        return img

class DualTrainDatasetRAF(data.Dataset):
    def __init__(self, root_dir_F, img_dir_list_F, img_lab_list_F, expr_data_dir, expr_img_dir, size=128, GRAY_flag=True):
        '''
        construct a dataset including CASIA database and the RAF-DB
        :param root_dir_F: the root directory of the face database
        :param img_dir_list_F: the list that include all image names and the corresponding folder of face dataset
        :param img_lab_list_F: the list that contain the label corresponding to each image of face dataset
        :param expr_data_dir: the directory of expression data
        :param expr_img_dir: the directory of expression images
        :param size: the size of output image
        :param RGB_flag: the flag to indicate whether load the image in RGB type
        '''

        self.GRAY = GRAY_flag
        self.size = size

        # root directory of face dataset
        self.root_dir_F = root_dir_F
        self.img_dir_list_F = img_dir_list_F
        self.img_lab_list_F = img_lab_list_F

        # transform for face dataset
        self.transform_F = transforms.Compose([
            transforms.Resize([size, size]),
            transforms.ToTensor(),
        ])

        # data of expression dataset
        self.expr_dir = expr_img_dir
        self.data_E = np.load(expr_data_dir, allow_pickle=True).item()
        self.expr_labels = self.data_E['train_y'].argmax(axis=0)

        # transform for expression dataset
        self.transform_E = transforms.Compose([
            transforms.Resize(144),
            transforms.RandomCrop(self.size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        # the sample number of the dataset which with less images
        self.less_len = min(len(self.img_lab_list_F), len(self.expr_labels))
        if len(self.expr_labels) < len(self.img_lab_list_F):
            self.face_more = True
        else:
            self.face_more = False

    def __getitem__(self, idx):
        # image index for the dataset with less images
        img_idx = idx % self.less_len

        if self.face_more:
            # load face image and label
            img_path_F = os.path.join(self.root_dir_F, self.img_dir_list_F[idx])
            img_F = self.img_prepro(img_path_F, self.transform_F, self.GRAY)
            lab_F = self.img_lab_list_F[idx].astype(np.long)

            # load expression image and label
            img_idx_str = '%05d' % (img_idx + 1)
            img_path_E = os.path.join(self.expr_dir, 'train_' + img_idx_str + '_aligned.jpg')
            img_E = self.img_prepro(img_path_E, self.transform_E, self.GRAY)
            lab_E = self.expr_labels[img_idx].astype(np.long)
        else:
            # load face image and label
            img_path_F = os.path.join(self.root_dir_F, self.img_dir_list_F[img_idx])
            img_F = self.img_prepro(img_path_F, self.transform_F, self.GRAY)
            lab_F = self.img_lab_list_F[img_idx].astype(np.long)

            # load expression image and label
            idx_str = '%05d' % (idx + 1)
            img_path_E = os.path.join(self.expr_dir, 'train_' + idx_str + '_aligned.jpg')
            img_E = self.img_prepro(img_path_E, self.transform_E, self.GRAY)
            lab_E = self.expr_labels[idx].astype(np.long)


        return img_F, lab_F, img_E, lab_E

    def __len__(self):
        return max(len(self.img_lab_list_F), len(self.expr_labels))

    def img_prepro(self, img_path, transform=None, GRAY=True):
        if GRAY:
            img = Image.open(img_path).convert('L')
        else:
            img = Image.open(img_path).convert('RGB')
        if transform is not None:
            img = transform(img)

        return img

class FaceDataset(data.Dataset):
    def __init__(self, root_dir, img_dir_list, img_lab_list, RGB_flag=False):
        '''

        :param root_dir: the root directory of the database
        :param img_dir_list: the list that include all image names and the corresponding folder
        :param img_lab_list: the list that contain labels of each image
        :param RGB_flag: the flag to indicate whether load the image in RGB type
        '''
        # RGB flag
        self.RGB = RGB_flag

        # root directory
        self.root_dir = root_dir

        self.img_dir_list = img_dir_list
        self.img_lab_list = img_lab_list

        # transform
        self.transform = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_dir_list[idx])
        img = self.img_prepro(img_path, self.transform, self.RGB)
        lab = self.img_lab_list[idx]
        return img, lab

    def __len__(self):
        return len(self.img_lab_list)

    def img_prepro(self, img_path, transform=None, RGB=False):
        if RGB:
            img = Image.open(img_path).convert('RGB')
        else:
            img = Image.open(img_path).convert('L')
        if transform is not None:
            img = transform(img)


        return img

class RAFDataset(data.Dataset):
    def __init__(self, root_dir, lab_dir, RGB_flag=False, train=False):
        '''
        :param root_dir: the root directory of the database
        :param lab_dir: the list that contain labels of each image
        :param RGB_flag: the flag to indicate whether load the image in RGB type
        :param train: the flag to indicate whether the object is the training or the testing set
        '''
        # RGB flag
        self.RGB = RGB_flag
        self.train = train

        # root directory
        self.root_dir = root_dir

        # label directory
        self.img_lab_list = np.load(lab_dir, allow_pickle=True).item()
        if train:
            self.img_lab_list = self.img_lab_list['train_y'].argmax(axis=0)
        else:
            self.img_lab_list = self.img_lab_list['test_y'].argmax(axis=0)

        # transform
        self.transform_train = transforms.Compose([
            transforms.Resize([144, 144]),
            transforms.CenterCrop([128, 128]),
            transforms.ToTensor(),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        if self.train:
            idx_str = '%05d' % (idx+1)
            img_path = os.path.join(self.root_dir, 'train_' + idx_str + '_aligned.jpg')
            img = self.img_prepro(img_path, self.transform_train, self.RGB)
        else:
            idx_str = '%04d' % (idx+1)
            img_path = os.path.join(self.root_dir, 'test_' + idx_str + '_aligned.jpg')
            img = self.img_prepro(img_path, self.transform_test, RGB=self.RGB)
        lab = self.img_lab_list[idx]
        return img, lab

    def __len__(self):
        return len(self.img_lab_list)

    def img_prepro(self, img_path, transform=None, RGB=False):
        if RGB:
            img = Image.open(img_path).convert('RGB')
        else:
            img = Image.open(img_path).convert('L')
        if transform is not None:
            img = transform(img)


        return img

def getFaceData(datapath, cls_num):
    '''
    truncate the image list
    :param datapath: the directory that stores image list
    :param cls_num: the number of classes used in the experiment
    :return: truncated image directory list and labels list
    '''
    face_data = np.load(datapath, allow_pickle=True)
    img_dir_list = face_data['img_dir_list']
    img_lab_list = face_data['img_lab_list']
    img_ind = np.argwhere(img_lab_list < cls_num)[:, 0]
    img_dir_list = img_dir_list[img_ind]
    img_lab_list = img_lab_list[img_ind]
    return img_dir_list, img_lab_list
