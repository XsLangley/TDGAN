'''
the main function for training and testing
'''

import numpy as np
import os
import trainer
import argparse
import torch


def run(args):
    # construct the hyper-parameter dictionary
    hpar_dict = {
        # noise dimension
        'Nz': 256,
        # steps for discriminator update (F: Face, E: Expression)
        'D_GAP_FR': 5, 
        'D_GAP_ER': 10, 
        # steps for saving images
        'IMG_SAVE_GAP': 100,
        # gaps of epochs to save the parameters
        'PAR_SAVE_GAP': 50,
        # validation gap
        'VAL_GAP': 1,
        # batch size
        'BS': args.batchsize,
        # training epochs
        'epoch': args.epoch,
        # class number for face recognition (the first K_f persons)
        'FR_cls_num': 20,
        # learning rate
        'LR_D_FR': args.lr,  # face discriminator
        'LR_D_ER': args.lr,  # expression discriminator
        'LR_G_FR': args.lr,  # face encoder
        'LR_G_ER': args.lr,  # expression encoder
        # coefficients to balance the loss of generator or discriminator
        'H_G_FR_f': 0.2,  # lambda_G_f
        'H_G_ER_f': 0.8,  # lambda_G_e
        'H_G_FR_PER': 1,  # lambda_per_f
        'H_G_ER_PER': 0,  # lambda_per_e
        'H_G_CON_FR': 5,  # lambda_DIC
        'H_G_CON_ER': 5,  # lambda_DIC
        'H_D_FR_r': 1,
        'H_D_FR_f': 1,
        'H_D_ER_r': 1,
        'H_D_ER_f': 0,
        # flags to indicate whether to generate grayscale images
        'FLAG_GEN_GRAYIMG': True,
        # face dataset
        'FR_DB': args.facedata,
        # expression dataset
        'ER_DB': args.exprdata,
        # working mode
        'train': args.train,
        'device': torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() and args.gpu >= 0 else "cpu"),
    }

    # directory to save each fold of experiments
    save_dir = os.path.join('experiments', hpar_dict['FR_DB'], hpar_dict['ER_DB'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    hpar_dict['save_dir'] = save_dir

    print('---- START RUNNING ----')
    print('WORKING MODE: {}\n'.format('TRAIN' if hpar_dict['train'] else 'VALIDATION'))
    trainer.train(hpar_dict)
    # acc_mat, ce_mat = trainer.train(hpar_dict)
    # print('the best accuracy on {} dataset is: {}'.format(hpar_dict['ER_DB'], np.max(acc_mat)))
    print('--- END OF RUNNING ----')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='a demon code of model TDGAN')
    parser.add_argument('--facedata', type=str, default='CASIA', help='the face dataset; you should customize your own dataloader to feed different datasets')
    parser.add_argument('--exprdata', type=str, default='RAF', help='the expression dataset; you should customize your own dataloader to feed different datasets')
    parser.add_argument('--train', default=False, action='store_true', help='flag to indicate working mode: default False (validation mode)')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs for training, default: 100')
    parser.add_argument('--batchsize', type=int, default=32, help='batch size, defualt: 32')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default: 1e-4')
    parser.add_argument('--gpu', type=int, default=-1, help='set gpu device, default: -1 (cpu)')

    args = parser.parse_args()
    run(args)