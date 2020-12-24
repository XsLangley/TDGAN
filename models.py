import torch
from torch import nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class BasicConv2d_Ins(nn.Module):
    '''
    BasicConv2d module with InstanceNorm
    '''
    def __init__(self, in_planes, out_planes, kernal_size, stride, padding):
        super(BasicConv2d_Ins, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernal_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.InstanceNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x= self.relu(x)
        return x


class block32_Ins(nn.Module):
    def __init__(self, scale=1.0):
        super(block32_Ins, self).__init__()

        self.scale = scale

        self.branch0 = nn.Sequential(BasicConv2d_Ins(64, 16, kernal_size=1, stride=1, padding=0))

        self.branch1 = nn.Sequential(
        BasicConv2d_Ins(64, 16, kernal_size=1, stride=1, padding=0),
        BasicConv2d_Ins(16, 16, kernal_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
        BasicConv2d_Ins(64, 16, kernal_size=1, stride=1, padding=0),
        BasicConv2d_Ins(16, 16, kernal_size=3, stride=1, padding=1),
        BasicConv2d_Ins(16, 16, kernal_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(48, 64, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    '''
    encoder structure: Inception + Instance Normalization
    '''
    def __init__(self, GRAY=False):
        super(Encoder, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        if GRAY:
            self.conv1 = nn.Sequential(BasicConv2d_Ins(1, 32, kernal_size=5, stride=1, padding=2))
        else:
            self.conv1 = nn.Sequential(BasicConv2d_Ins(3, 32, kernal_size=5, stride=1, padding=2))

        self.conv2 = nn.Sequential(BasicConv2d_Ins(32, 64, kernal_size=5, stride=1, padding=2))
        self.repeat = nn.Sequential(
            block32_Ins(scale=0.17),
            block32_Ins(scale=0.17),
            block32_Ins(scale=0.17),
            block32_Ins(scale=0.17)
        )
        self.conv3 = nn.Sequential(BasicConv2d_Ins(64, 128, kernal_size=5, stride=1, padding=2))
        self.conv4 = nn.Sequential(BasicConv2d_Ins(128, 128, kernal_size=5, stride=1, padding=2))

    def forward(self, x_in):
        # in_chanx128x128 -> 32x128x128
        self.conv1_out = self.conv1(x_in)
        # 32x128x128 -> 32x64x64
        self.ds1_out = self.maxpool(self.conv1_out)
        # 32x64x64 -> 64x64x64
        self.conv2_out = self.conv2(self.ds1_out)
        # 64x64x64 -> 64x32x32
        self.ds2_out = self.maxpool(self.conv2_out)
        # 64x32x32 -> 64x32x32
        self.incep_out = self.repeat(self.ds2_out)
        # 64x32x32 -> 128x32x32
        self.conv3_out = self.conv3(self.incep_out)
        # 128x32x32 -> 128x16x16
        self.ds3_out = self.maxpool(self.conv3_out)
        # 128x16x16 -> 128x16x16
        self.conv4_out = self.conv4(self.ds3_out)
        # 128x16x16 -> 128x8x8
        self.ds4_out = self.maxpool(self.conv4_out)
        return self.ds4_out


class fc_layer(nn.Module):
    def __init__(self, par=None, p=0.5, cls_num=10575):
        super(fc_layer, self).__init__()
        # activation function
        self.act = nn.ReLU()

        # network structure
        self.fc1 = nn.Linear(8 * 8 * 128, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, cls_num)

        self.dropout = nn.Dropout(p=p)

        # parameters initiation
        if par:
            # to load pre-trained model
            fc_dict = self.state_dict().copy()
            fc_list = list(self.state_dict().keys())

            fc_dict[fc_list[0]] = par['module.fc.weight']
            fc_dict[fc_list[1]] = par['module.fc.bias']

            # load pre-trained parameters into Encoder
            self.load_state_dict(fc_dict)
        else:
            # initiate parameters
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0, 0.02)
                elif isinstance(m, nn.ConvTranspose2d):
                    m.weight.data.normal_(0, 0.02)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.02)
                    m.bias.data.fill_(0)

    def forward(self, fea):
        # fc1: bsx8x8x128 -> bsx8192 -> bsx1024
        self.fc1_out = self.act(self.fc1(self.dropout(fea.view(fea.size(0), -1))))
        # fc2: bsx1024 -> bsx1024
        self.fc2_out = self.act(self.fc2(self.dropout(self.fc1_out)))
        # fc3: bsx1024 -> bsxcls_num
        self.fc3_out = self.fc3(self.fc2_out)
        return self.fc3_out


class resblock(nn.Module):
    '''
    residual block
    '''
    def __init__(self, n_chan):
        super(resblock, self).__init__()
        self.infer = nn.Sequential(*[
            nn.Conv2d(n_chan, n_chan, 3, 1, 1),
            nn.ReLU()
        ])

    def forward(self, x_in):
        self.res_out = x_in + self.infer(x_in)
        return self.res_out


class decoder(nn.Module):
    def __init__(self, Nz=100, Nb=3, Nc=128, GRAY=False):
        '''
        decoder to generate an image
        :param Nz: dimension of noises
        :param Nb: number of blocks
        :param Nc: channel number
        '''
        super(decoder, self).__init__()

        self.Nz = Nz

        # embedding layer
        self.emb1 = nn.Sequential(*[
            nn.Conv2d(128*2 + Nz, Nc, 3, 1, 1),
            nn.ReLU(),
        ])
        self.emb2 = self._make_layer(resblock, Nb, Nc)

        # decoding layers
        self.us1 = nn.Sequential(*[
            nn.ConvTranspose2d(Nc, 512, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(512),
            nn.ReLU(True),
        ])
        self.us2 = nn.Sequential(*[
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
        ])
        self.us3 = nn.Sequential(*[
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
        ])
        self.us4 = nn.Sequential(*[
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
        ])
        if GRAY:
            self.us5 = nn.Sequential(*[
                nn.ConvTranspose2d(64, 1, 3, 1, 1, bias=False),
                nn.Sigmoid()
            ])
        else:
            self.us5 = nn.Sequential(*[
                nn.ConvTranspose2d(64, 3, 3, 1, 1, bias=False),
                nn.Sigmoid()
            ])

    def _make_layer(self, block, num_blocks, n_chan):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(n_chan))
        return nn.Sequential(*layers)

    def forward(self, enc_FR, enc_ER, noise=None, device=None):
        # features of the branch
        fea_ER = enc_ER.ds4_out
        fea_FR = enc_FR.ds4_out

        # concatenate the inputs with noises
        if noise is not None:
            noise = noise
        else:
            noise = Variable(torch.rand(fea_ER.shape[0], self.Nz, 8, 8))
        
        if device is not None:
            noise = noise.to(device)

        if self.Nz == 0:
            emb_in = torch.cat((fea_ER, fea_FR), dim=1)
        else:
            emb_in = torch.cat((fea_ER, fea_FR, noise), dim=1)
        # embedding: bsx(256+Nz)x8x8 -> bsxNcx8x8
        self.emb1_out = self.emb1(emb_in)
        # bsxNcx8x8 -> bsxNcx8x8
        self.emb2_out = self.emb2(self.emb1_out)

        # decoding:
        # bsxNcx8x8 -> bsx512x16x16
        self.us1_out = self.us1(self.emb2_out)
        # bsx512x16x16 -> bsx256x32x32
        self.us2_out = self.us2(self.us1_out)
        # bsx256x32x32 -> bsx128x64x64
        self.us3_out = self.us3(self.us2_out)
        # bsx128x64x64 -> bsx64x128x128
        self.us4_out = self.us4(self.us3_out)
        # bsx64x128x128 -> bsxout_chanx128x128
        self.img = self.us5(self.us4_out)

        return self.img


class Dis(nn.Module):
    '''
    the class of discriminator to handle classification
    '''
    def __init__(self, fc=None, GRAY=True, cls_num=6):
        super(Dis, self).__init__()

        # initiate encoder
        self.enc = Encoder(GRAY=GRAY)

        # initiate fc layer
        self.fc = fc_layer(cls_num=cls_num)

    def forward(self, x_in):
        self.fea = self.enc(x_in)
        self.result = self.fc(self.fea)
        return self.fea, self.result


class Gen(nn.Module):
    '''
    the class of generator
    '''
    def __init__(self, clsn_ER=7, Nz=100, Nb=3, GRAY=False):
        super(Gen, self).__init__()
        # encoders for the two branches
        self.enc_FR = Encoder(GRAY=GRAY)
        self.enc_ER = Encoder(GRAY=GRAY)

        # Expression Classification Module (M_ER)
        self.fc_ER = fc_layer(cls_num=clsn_ER)

        # decoder in generator
        self.dec = decoder(Nz=Nz, GRAY=GRAY, Nb=Nb)
        self.dec.apply(weights_init)

    def infer_FR(self, x_FR):
        fea_FR = self.enc_FR(x_FR)
        return fea_FR

    def infer_ER(self, x_ER):
        fea_ER = self.enc_ER(x_ER)
        result_ER = self.fc_ER(fea_ER)
        return fea_ER, result_ER

    def gen_img(self, x_FR, x_ER, noise=None, device=None):
        self.fea_FR = self.infer_FR(x_FR=x_FR)
        self.fea_ER, self.result_ER = self.infer_ER(x_ER=x_ER)
        self.img = self.dec(enc_FR=self.enc_FR, enc_ER=self.enc_ER, noise=noise, device=device)
        return self.img

    def gen_img_withfea(self, fea_FR, fea_ER):
        self.enc_FR.ds4_out = fea_FR
        self.enc_ER.ds4_out = fea_ER
        self.img = self.dec(enc_FR=self.FR, enc_ER=self.enc_ER)
        return self.img


