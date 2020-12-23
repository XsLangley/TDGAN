'''
some functions used in the model
'''
import numpy as np
from torch.autograd import Variable
import PIL.Image as Image
import torch

def Val_acc(loader, Dis, criterion, device):
    '''
    validation function based on self-built models
    :param loader: data loader
    :param Dis: discriminator object
    :param criterion: criterion to calculate the loss
    :return: accuracy and loss
    '''
    # predictions result
    pre_list = []
    # ground-truth
    GT_list = []
    val_ce = 0

    for i, (batch_val_x, batch_val_y) in enumerate(loader):
        GT_list = np.hstack((GT_list, batch_val_y.numpy()))
        batch_val_x = Variable(batch_val_x).to(device)
        batch_val_y = Variable(batch_val_y).to(device)
        # inference
        _, batch_p = Dis(batch_val_x)

        batch_result = batch_p.cpu().data.numpy().argmax(axis=1)
        pre_list = np.hstack((pre_list, batch_result))

        # classification loss
        val_ce += criterion(batch_p, batch_val_y).cpu().data.numpy()

    # calculate the accuracy
    val_acc = (np.sum((GT_list == pre_list).astype(float))) / len(GT_list)
    val_ce = val_ce / i

    return val_acc, val_ce


def combinefig_dualcon(FR_mat, ER_mat, Fake_mat, con_FR, con_ER, save_num=3):
    '''
    combine five images to one row, combine three row of images to one complete image
    :param FR_mat: face images
    :param ER_mat: expression images
    :param Fake_mat: fake images
    :param con_FR: consistent images with respect to FR
    :param con_ER: consistent images with respect to ER
    :return: combined images
    '''
    save_num = min(FR_mat.shape[0], save_num)
    imgsize = np.shape(FR_mat)[-1]
    img = np.zeros([imgsize * save_num, imgsize * 5, 3])
    for i in range(0, save_num):
        img[i * imgsize: (i + 1) * imgsize, 0 * imgsize: 1 * imgsize, :] = FR_mat[i, :, :, :].transpose([1, 2, 0])
        img[i * imgsize: (i + 1) * imgsize, 1 * imgsize: 2 * imgsize, :] = ER_mat[i, :, :, :].transpose([1, 2, 0])
        img[i * imgsize: (i + 1) * imgsize, 2 * imgsize: 3 * imgsize, :] = Fake_mat[i, :, :, :].transpose([1, 2, 0])
        img[i * imgsize: (i + 1) * imgsize, 3 * imgsize: 4 * imgsize, :] = con_FR[i, :, :, :].transpose([1, 2, 0])
        img[i * imgsize: (i + 1) * imgsize, 4 * imgsize: 5 * imgsize, :] = con_ER[i, :, :, :].transpose([1, 2, 0])

    return img


def preprocess_img(img_dir, device):
    img = Image.open(img_dir).convert('L').resize((128, 128))
    img = torch.from_numpy(np.array(img)/255).unsqueeze(0).unsqueeze(0).float()
    img = Variable(img).to(device)
    return img



def Val_acc_single(x_ER, Dis, device, name):
    # the meaning of each class in RAF-DB
    exprdict = {
        0: 'Surprise',
        1: 'Fear',
        2: 'Disgust',
        3: 'Happiness',
        4: 'Sadness',
        5: 'Anger',
        6: 'Neutral',
    }
    x_ER = Variable(x_ER).to(device)
    # inference
    _, x_p = Dis(x_ER)
    pred_cls = x_p.cpu().data.numpy().argmax(axis=1).item()
    print('the predicted class of model {} is: {}'.format(name, exprdict[pred_cls]))


def del_extra_keys(model_par_dir):
    # the pretrained model is trained on old version pytorch, some extra keys should be deleted before loading
    model_par_dict = torch.load(model_par_dir)
    model_par_dict_clone = model_par_dict.copy()
    # delete keys
    for key, value in model_par_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_par_dict[key]
    
    return model_par_dict

