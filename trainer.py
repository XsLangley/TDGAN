'''
the trainer function
'''

import os
import time
import torch
from torch import nn, optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import models as model
import numpy as np
import LoadData
import util
import PIL.Image as Image

def train(hpar_dict):

    # region: hyper-parameters for the model

    # noise channel
    Nz = hpar_dict['Nz']

    # steps gap to update discriminator
    D_GAP_FR = hpar_dict['D_GAP_FR']
    D_GAP_ER = hpar_dict['D_GAP_ER']
    # steps gap for saving images
    IMG_SAVE_GAP = hpar_dict['IMG_SAVE_GAP']
    # gaps to save parameters (epochs)
    PAR_SAVE_GAP = hpar_dict['PAR_SAVE_GAP']
    # validation gap
    VAL_GAP = hpar_dict['VAL_GAP']
    # batch size
    BS = hpar_dict['BS']
    # training epochs
    epoch = hpar_dict['epoch']
    # face recognition class number
    FR_cls_num = hpar_dict['FR_cls_num']

    # learning rate
    LR_D_FR = hpar_dict['LR_D_FR']
    LR_D_ER = hpar_dict['LR_D_ER']
    LR_G_FR = hpar_dict['LR_G_FR']
    LR_G_ER = hpar_dict['LR_G_ER']
    # weights to balance the loss of generator or discriminator
    H_G_FR_f = hpar_dict['H_G_FR_f']
    H_G_ER_f = hpar_dict['H_G_ER_f']
    H_G_FR_PER = hpar_dict['H_G_FR_PER']
    H_G_ER_PER = hpar_dict['H_G_ER_PER']
    H_G_CON_FR = hpar_dict['H_G_CON_FR']
    H_G_CON_ER = hpar_dict['H_G_CON_ER']

    H_D_FR_r = hpar_dict['H_D_FR_r']
    H_D_FR_f = hpar_dict['H_D_FR_f']
    H_D_ER_r = hpar_dict['H_D_ER_r']
    H_D_ER_f = hpar_dict['H_D_ER_f']
    # flag to indicate whether to generate grayscale images
    FLAG_GEN_GRAYIMG = hpar_dict['FLAG_GEN_GRAYIMG']
    # dataset
    ER_DB = hpar_dict['ER_DB']
    FR_DB = hpar_dict['FR_DB']
    save_dir = hpar_dict['save_dir']
    # directory to save images
    img_dir = os.path.join(save_dir, 'img')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    # directory to save parameters
    par_dir = os.path.join(save_dir, 'par')
    if not os.path.exists(par_dir):
        os.makedirs(par_dir)
    
    TRAIN_FLAG = hpar_dict['train']

    # endregion


    # region: construct dataloaders for training

    # construct face dataset's dataloader.
    if FR_DB == 'CASIA':
        # load the npz file that stores images' data (dir, label)
        face_data_dir = './Dataset/CASIA_WebFace/casia_data_example.npz'
        face_img_dir_list, face_img_lab_list = LoadData.getFaceData(face_data_dir, FR_cls_num)
        # the root directory of where images are stored (in your local machine)
        face_root_dir = './Dataset/CASIA_WebFace/img/'

    if ER_DB == 'RAF':
        ER_cls_num = 7 # the class of expressions inn RAF-DB
        acc_max = 0.1

        # npy file that stores the data of RAF-DB (img name, labels)
        expr_data_dir = './Dataset/RAF/RAF_example_label.npy'

        expr_root_dir = './Dataset/RAF/img/'
        lab_dir = expr_data_dir
        # the dataset instance for testing
        dataset_test = LoadData.RAFDataset(expr_root_dir, lab_dir, RGB_flag=not FLAG_GEN_GRAYIMG, train=False)

    # dataset instance for training
    dataset_train = LoadData.DualTrainDatasetRAF(face_root_dir, face_img_dir_list, face_img_lab_list, expr_data_dir, expr_root_dir, GRAY_flag=FLAG_GEN_GRAYIMG)

    # dataloader for training, which contains two datasets
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BS, shuffle=True, num_workers=2)
    # dataloader for testing, which only contains the expression dataset
    expr_tt_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BS, shuffle=False, num_workers=2)


    # endregion

    # region: module instantiate
    # instantiate Generator
    Gen = model.Gen(clsn_ER=ER_cls_num, Nz=Nz, GRAY=FLAG_GEN_GRAYIMG, Nb=6)

    # instantiate face discriminator
    Dis_FR = model.Dis(GRAY=FLAG_GEN_GRAYIMG, cls_num=FR_cls_num + 1)
    # instantiate expression discriminator
    Dis_ER = model.Dis(GRAY=FLAG_GEN_GRAYIMG, cls_num=ER_cls_num)

    # instantiate Expression Clssification Module (M_ER)
    Dis_ER_val = model.Dis()
    Dis_ER_val.enc = Gen.enc_ER
    Dis_ER_val.fc = Gen.fc_ER

    # push model into GPU
    Gen.to(hpar_dict['device'])
    Dis_FR.to(hpar_dict['device'])
    Dis_ER.to(hpar_dict['device'])
    Dis_ER_val.to(hpar_dict['device'])

    # endregion

    
    # region optimizer definition

    # parameters of the generator
    par_list_G_joint = [{'params': Gen.dec.parameters(), 'lr': LR_G_ER},
                        {'params': Gen.enc_FR.parameters(), 'lr': LR_G_FR},
                        {'params': Gen.enc_ER.parameters(), 'lr': LR_G_ER}
                        ]
    # parameters of the Expression Recognition Module
    par_list_G_ER_fc = [{'params': Gen.fc_ER.parameters(), 'lr': LR_D_ER},
                        ]
    # parameters of the two discriminators
    par_list_D_FR = [{'params': Dis_FR.parameters(), 'lr': LR_D_FR},
                    ]
    par_list_D_ER = [{'params': Dis_ER.parameters(), 'lr': LR_D_ER},
                    ]


    optG_joint = optim.Adam(par_list_G_joint)
    optG_ER_fc = optim.Adam(par_list_G_ER_fc)
    optD_FR = optim.Adam(par_list_D_FR)
    optD_ER = optim.Adam(par_list_D_ER)

    # endregion


    # criterion for loss
    CE = nn.CrossEntropyLoss()
    MSE = nn.MSELoss()
    L1_loss = nn.L1Loss()


    if not TRAIN_FLAG:
        # region: load pretrained model and do forward steps

        # pretrained model dir
        pre_root_dir = './Dataset/examples'
        par_Enc_FR_dir = os.path.join(pre_root_dir, 'Enc_FR_G.pkl')
        par_Enc_ER_dir = os.path.join(pre_root_dir, 'Enc_ER_G.pkl')
        par_dec_dir = os.path.join(pre_root_dir, 'dec.pkl')
        par_fc_ER_dir = os.path.join(pre_root_dir, 'fc_ER_G.pkl')

        par_Dis_ER_dir = os.path.join(pre_root_dir, 'Dis_ER.pkl')
        # load parameters
        print('loading pretrained models......')
        util.del_extra_keys(par_Enc_FR_dir)
        Gen.enc_FR.load_state_dict(util.del_extra_keys(par_Enc_FR_dir))
        Gen.enc_ER.load_state_dict(util.del_extra_keys(par_Enc_ER_dir))
        Gen.dec.load_state_dict(util.del_extra_keys(par_dec_dir))
        Gen.fc_ER.load_state_dict(util.del_extra_keys(par_fc_ER_dir))
        Dis_ER_val.enc.load_state_dict(util.del_extra_keys(par_Enc_ER_dir))
        Dis_ER_val.fc.load_state_dict(util.del_extra_keys(par_fc_ER_dir))

        Dis_ER.load_state_dict(util.del_extra_keys(par_Dis_ER_dir))

        # load example images for validation demo
        face_img_dir = '{}/face.jpg'.format(pre_root_dir)
        expression_img_dir = '{}/expression.jpg'.format(pre_root_dir)
        faceimg = util.preprocess_img(face_img_dir, hpar_dict['device'])
        exprimg = util.preprocess_img(expression_img_dir, hpar_dict['device'])

        with torch.no_grad():
            Gen.eval()
            gen_img = Gen.gen_img(faceimg, exprimg, device=hpar_dict['device'])
            gen_img_copy = gen_img.detach()
            gen_img = gen_img.squeeze().cpu().data.numpy()
            img_PIL = Image.fromarray((gen_img * 255).astype(np.uint8))
            img_PIL.save('{}/generated_example.png'.format(pre_root_dir))

            # validation using M_ER
            Dis_ER_val.eval()
            util.Val_acc_single(gen_img_copy, Dis_ER_val,device=hpar_dict['device'], name='M_ER')

            # validation using discriminator
            Dis_ER.eval()
            util.Val_acc_single(gen_img_copy, Dis_ER, device=hpar_dict['device'], name='Expr Dis')

        # endregion
    else:
        
        # buffer to store validation accuracy (Expression Classification Module)
        tt_acc_mat = []
        tt_ce_mat = []
        # buffer to store validation accuracy (Expression discriminator)
        tt_acc_mat_ExpDis = []
        tt_ce_mat_ExpDis = []

        # start training
        for e in range(1, epoch + 1):
            print('---- training ----')
            # the number of steps that an epoch goes
            step_total = train_loader.__len__()
            t_start = time.time()
            print('the %d-th training epoch' % (e))

            # set training mode
            Gen.train()
            Dis_ER.train()
            Dis_FR.train()

            for step, (batch_FR_x_r, batch_FR_y_r, batch_ER_x_r, batch_ER_y_r) in enumerate(train_loader):

                # region batch data preparation

                # batch_FR_x_r: real batch data for Face Recognition
                # batch_FR_y_r: real batch labels for Face Recognition
                # batch_ER_x_r: real batch data for expression Recognition
                # batch_ER_y_r: real batch labels for expression Recognition

                # labels for fake images
                batch_FR_y_f = FR_cls_num * torch.ones(len(batch_FR_y_r)).long()

                # convert all tensors to the form of torch.Variables
                batch_FR_x_r = Variable(batch_FR_x_r).to(hpar_dict['device'])
                batch_FR_y_r = Variable(batch_FR_y_r).long().to(hpar_dict['device'])

                batch_ER_x_r = Variable(batch_ER_x_r).to(hpar_dict['device'])
                batch_ER_y_r = Variable(batch_ER_y_r).long().to(hpar_dict['device'])

                batch_FR_y_f = Variable(batch_FR_y_f).long().to(hpar_dict['device'])

                # endregion


                # region forward step

                # go through the discriminators
                batch_FR_Dfea_r, batch_FR_Dp_r = Dis_FR(batch_FR_x_r)
                batch_FR_Dfea_r = Variable(batch_FR_Dfea_r.data, requires_grad=False)
                batch_ER_Dfea_r, batch_ER_Dp_r = Dis_ER(batch_ER_x_r)
                batch_ER_Dfea_r = Variable(batch_ER_Dfea_r.data, requires_grad=False)

                # loss of face discriminator (with respect to real samples)
                loss_D_FR_r = CE(batch_FR_Dp_r, batch_FR_y_r)
                # loss of expression discriminator (with respect to real samples)
                loss_D_ER_r = CE(batch_ER_Dp_r, batch_ER_y_r)

                # generat images
                batch_x_f = Gen.gen_img(batch_FR_x_r, batch_ER_x_r, device=hpar_dict['device'])
                batch_ER_Gfea_r = Variable(Gen.fea_ER.data, requires_grad=False)

                # region: update Expression Recognition Module
                optG_ER_fc.zero_grad()
                err_G_ER_r = CE(Gen.result_ER, batch_ER_y_r)
                err_G_ER_r.backward(retain_graph=True)
                optG_ER_fc.step()
                # endregion

                # endregion


                # region update discriminators
                
                # clear gradient buffer
                optD_FR.zero_grad()
                optD_ER.zero_grad()
                if step % D_GAP_FR == 0:
                    batch_FR_Dfea_f, batch_FR_Dp_f = Dis_FR(batch_x_f.detach())
                    # loss of face discriminator (with respect to fake samples)
                    loss_D_FR_f = CE(batch_FR_Dp_f, batch_FR_y_f)

                    # full loss of face discriminator
                    loss_D_FR = H_D_FR_r * loss_D_FR_r + H_D_FR_f * loss_D_FR_f

                    loss_D_FR.backward()
                    optD_FR.step()

                if step % D_GAP_ER == 0:

                    # loss of expression discriminator
                    loss_D_ER = H_D_ER_r * loss_D_ER_r

                    loss_D_ER.backward()
                    optD_ER.step()


                # endregion


                # region update generator

                optG_joint.zero_grad()

                # get the predicted results on fake samples
                batch_FR_Dfea_f, batch_FR_Dp_f = Dis_FR(batch_x_f)
                batch_ER_Dfea_f, batch_ER_Dp_f = Dis_ER(batch_x_f)

                err_G_FR_f = CE(batch_FR_Dp_f, batch_FR_y_r) # Equ.6 first part
                err_G_ER_f = CE(batch_ER_Dp_f, batch_ER_y_r) # Equ.6 second part
                err_G_FR_PER = MSE(batch_FR_Dfea_f, batch_FR_Dfea_r) # Equ.8

                # consistency loss (Fig.3 upper part): face branch input: the generated image, expression branch input: the original face image, expected output: same as the original face image
                batch_x_f_FR = Gen.gen_img(batch_x_f, batch_FR_x_r, device=hpar_dict['device'])
                # consistency loss (Fig.3 lower part): face branch input: the original expression image, expression branch: the generated image, expected output: same as the original expression image
                batch_x_f_ER = Gen.gen_img(batch_ER_x_r, batch_x_f, device=hpar_dict['device'])

                # expression perceptual error (unused)
                batch_ER_Gfea_f = Variable(Gen.fea_ER.data).to(hpar_dict['device'])
                err_G_ER_PER = MSE(batch_ER_Gfea_f, batch_ER_Gfea_r)

                err_G_con_FR = L1_loss(batch_x_f_FR, batch_FR_x_r)
                err_G_con_ER = L1_loss(batch_x_f_ER, batch_ER_x_r)
                err_G_con = H_G_CON_FR * err_G_con_FR + H_G_CON_ER * err_G_con_ER # Equ.7
                loss_G = H_G_FR_f * err_G_FR_f + H_G_ER_f * err_G_ER_f + \
                        H_G_FR_PER * err_G_FR_PER + H_G_ER_PER * err_G_ER_PER + err_G_con

                loss_G.backward()
                optG_joint.step()

                # endregion


                if step % 5 == 0:
                    print('the current information of the model:')
                    print('%d / %d' % (step, step_total))
                    print('the loss of G (total): %f' % (loss_G.cpu().data))
                    print('the loss of G (face): %f' % (err_G_FR_f.cpu().data))
                    print('the loss of G (expression): %f' % (err_G_ER_f.cpu().data))
                    print('the loss of G (face-per): %f' % (err_G_FR_PER.cpu().data))
                    print('the loss of G (expr-per): %f' % (err_G_ER_PER.cpu().data))
                    print('the loss of G (consistency): %f' % (err_G_con.cpu().data))
                    print('the loss of G (FR-cons): %f' % (err_G_con_FR.cpu().data))
                    print('the loss of G (ER-cons): %f' % (err_G_con_ER.cpu().data))
                    print('------------------------------------------------------------')
                    print('the loss of D (face-total): %f' % (loss_D_FR.cpu().data))
                    print('the loss of D (face-real): %f' % (loss_D_FR_r.cpu().data))
                    print('the loss of D (face-fake): %f' % (loss_D_FR_f.cpu().data))
                    print('the loss of D (expression): %f' % (loss_D_ER.cpu().data))

                # save the generated images
                if step % IMG_SAVE_GAP == 0:
                    # combine five images of real face, real expression and fake images
                    comb_img = util.combinefig_dualcon(batch_FR_x_r.cpu().data.numpy(),
                                                    batch_ER_x_r.cpu().data.numpy(),
                                                    batch_x_f.cpu().data.numpy(),
                                                    batch_x_f_FR.cpu().data.numpy(),
                                                    batch_x_f_ER.cpu().data.numpy())
                    # save figures
                    comb_img = Image.fromarray((comb_img * 255).astype(np.uint8))
                    comb_img.save(os.path.join(img_dir, str(e) + '_' + str(step) + '.jpg'))


            t_end = time.time()
            print('an epoch last for %f seconds\n' % (t_end - t_start))


            # region validation

            if e % VAL_GAP == 0:
                Dis_ER_val.eval()
                tt_acc, tt_ce = util.Val_acc(expr_tt_loader, Dis_ER_val, CE, device=hpar_dict['device'])
                tt_acc_mat.append(tt_acc)
                tt_ce_mat.append(tt_ce)
                if tt_acc > acc_max:
                    acc_max = tt_acc
                    torch.save(Gen.enc_ER.state_dict(), os.path.join(par_dir, 'Enc_ER_G.pkl'))
                    torch.save(Gen.enc_FR.state_dict(), os.path.join(par_dir, 'Enc_FR_G.pkl'))
                    torch.save(Gen.fc_ER.state_dict(), os.path.join(par_dir, 'fc_ER_G.pkl'))
                    torch.save(Gen.dec.state_dict(), os.path.join(par_dir, 'dec.pkl'))
                    torch.save(Dis_FR.state_dict(), os.path.join(par_dir, 'Dis_FR.pkl'))
                    torch.save(Dis_ER.state_dict(), os.path.join(par_dir, 'Dis_ER.pkl'))
                print('\n')
                print('the %d-th epoch' % (e))
                print('accuracy is : %f' % (tt_acc))
                print('validation cross enntropy is : %f' % (tt_ce))
                print('now the best accuracy is %f\n' % (np.max(tt_acc_mat)))

                # validation using discriminator
                Dis_ER.eval()
                tt_acc_ExpDis, tt_ce_ExpDis = util.Val_acc(expr_tt_loader, Dis_ER, CE, device=hpar_dict['device'])
                tt_acc_mat_ExpDis.append(tt_acc_ExpDis)
                tt_ce_mat_ExpDis.append(tt_ce_ExpDis)
                print('testing using discriminator:')
                print('accuracy is : %f' % (tt_acc_ExpDis))
                print('testing cross enntropy is : %f' % (tt_ce_ExpDis))
                print('now the best accuracy is %f\n' % (np.max(tt_acc_mat_ExpDis)))


            if e % PAR_SAVE_GAP == 0:
                torch.save(Gen.enc_ER.state_dict(), os.path.join(par_dir, 'Enc_ER_G_' + str(e) +'.pkl'))
                torch.save(Gen.enc_FR.state_dict(), os.path.join(par_dir, 'Enc_FR_G_' + str(e) +'.pkl'))
                torch.save(Gen.fc_ER.state_dict(), os.path.join(par_dir, 'fc_ER_G_' + str(e) +'.pkl'))
                torch.save(Gen.dec.state_dict(), os.path.join(par_dir, 'dec_' + str(e) +'.pkl'))

                torch.save(Dis_FR.state_dict(), os.path.join(par_dir, 'Dis_FR_' + str(e) +'.pkl'))
                torch.save(Dis_ER.state_dict(), os.path.join(par_dir, 'Dis_ER_' + str(e) +'.pkl'))
            # endregion
    
        print('end')
        
        np.savez(os.path.join(save_dir, 'val_data.npz'), tt_acc_mat=tt_acc_mat, tt_ce_mat=tt_ce_mat,
                    tt_acc_mat_ExpDis=tt_acc_mat_ExpDis, tt_ce_mat_ExpDis=tt_ce_mat_ExpDis)
        return tt_acc_mat, tt_ce_mat




