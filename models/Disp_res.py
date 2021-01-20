#this edition use same resnet 50 as the normal one but configure the decoder as the left-right unsupervised one
#thus there something in this decoder is not reasonable(should change layer1 in encoder with stride of 2 for 
#consistancy with left-right edition, but that one have not tried with pretrained weight)

#before 23.05.19, all the test about disp_res is built upon this edition of code

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
import torch.utils.model_zoo as model_zoo
import pdb

def upsample_nn_nearest(x):
    return F.upsample(x, scale_factor=2, mode='nearest')

def predict_disp(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )
# this conv and upconv are just for decoder since the encoder use the conv1x1 and conv3x3
def conv(in_planes, out_planes, leaky=True):
    if leaky:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
def upconv(in_planes, out_planes, leaky=True):
    if leaky:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

def crop_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]

def maxpool(kernel_size):
    p = (kernel_size - 1)//2
    return nn.MaxPool2d(kernel_size, stride=None, padding=p)

class Disp_res(nn.Module):

    def __init__(self, datasets ='kitti'):
        super(Disp_res, self).__init__()
        #not sure about the setting of these two parameter since the left-right consistancy one use alpha 0.3 and beta 0.01
        if datasets == 'kitti':
            self.alpha = 10
            self.beta = 0.01
        elif datasets == 'nyu':
            self.alpha = 10#not sure about this number choice(I just think nyu should be more detailed)
            self.beta = 0.1
        
        self.only_train_dec = False
        #resnet encoder
        self.inplanes = 64
        conv_planes = [64, 64, 128, 256, 512]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        #not sure with or without BN
        self.bn1 = nn.BatchNorm2d(64) #tinghui trick for vgg similiar struture did not have that part in order to improve
        
        self.relu = nn.ReLU(inplace=True)
        self.pool1=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.resblock(conv_planes[1], 3)
        self.layer2 = self.resblock(conv_planes[2], 4, stride=2)
        self.layer3 = self.resblock(conv_planes[3], 6, stride=2)
        self.layer4 = self.resblock(conv_planes[4], 3, stride=2)
        
        #decode
        upconv_planes = [512, 256, 128, 64, 32, 16]
        self.upconv6 = upconv(conv_planes[4]*4,   upconv_planes[0])
        self.upconv5 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv4 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv3 = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv2 = upconv(upconv_planes[3], upconv_planes[4])
        self.upconv1 = upconv(upconv_planes[4], upconv_planes[5])	

        self.iconv6 = conv(upconv_planes[0] + conv_planes[3]*4, upconv_planes[0])
        self.iconv5 = conv(upconv_planes[1] + conv_planes[2]*4, upconv_planes[1])
        self.iconv4 = conv(upconv_planes[2] + conv_planes[1]*4, upconv_planes[2])
        #with depth result from last layer
        self.iconv3 = conv(1 + upconv_planes[3] + conv_planes[1], upconv_planes[3])
        self.iconv2 = conv(1 + upconv_planes[4] + conv_planes[0], upconv_planes[4])
        self.iconv1 = conv(1 + upconv_planes[5], upconv_planes[5])

        self.predict_disp4 = predict_disp(upconv_planes[2])
        self.predict_disp3 = predict_disp(upconv_planes[3])
        self.predict_disp2 = predict_disp(upconv_planes[4])
        self.predict_disp1 = predict_disp(upconv_planes[5])

    def resblock(self, planes, num_blocks, stride=1):
        downsample=None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * 4, stride),
                nn.BatchNorm2d(planes * 4),
            )

        layers=[]
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * 4#all the 4 is originally block.expansion
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(self.inplanes, planes))#inplanes, planes, stride=1
	    #layers.append(Bottleneck(self.inplanes, planes, stride=2))

        return nn.Sequential(*layers)
    
    def init_weights(self, use_pretrained_weights=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
        if use_pretrained_weights:
            print("loading pretrained weights downloaded from pytorch.org")
            self.load_res_params(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
        else:
            print("do not load pretrained weights for the monocular model")

    def load_res_params(self, params):
        model_dict = self.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in params.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.load_state_dict(model_dict)

    def forward(self, x):
    	#encoder
        conv1 = self.conv1(x)

        bn1   = self.bn1(conv1)

        relu1 = self.relu(conv1)     # H/2  -   64D
        pool1 = self.pool1(relu1)    # H/4  -   64D

        conv2 = self.layer1(pool1)   # H/4  -  256D   H/8  -  256D
        # H/4  -  256D is for the resnet 50 and H/8  -  256D is for the left-right paper which utilize stride in first resblock layer
        conv3 = self.layer2(conv2)   # H/8  -  512D   H/16 -  512D
        conv4 = self.layer3(conv3)   # H/16 -  1024D  H/32 - 1024D
        conv5 = self.layer4(conv4)   # H/32 -  2048D  H/64 - 2048D

        if self.only_train_dec:
            relu1 = relu1.detach()
            pool1 = pool1.detach()
            conv2 = conv2.detach()
            conv3 = conv3.detach()
            conv4 = conv4.detach()
            conv5 = conv5.detach()
        #skips

        skip1 = relu1
        skip2 = pool1
        skip3 = conv2
        skip4 = conv3
        skip5 = conv4

        #decoder
        upconv6 = crop_like(self.upconv6(conv5), skip5)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = crop_like(self.upconv5(iconv6), skip4)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = crop_like(self.upconv4(iconv5), skip3)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        disp4 = self.alpha * self.predict_disp4(iconv4) + self.beta

        upconv3 = crop_like(self.upconv3(iconv4), skip2)
        #disp4_up = upsample_nn_nearest(disp4)
        disp4_up = crop_like(F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), skip2)
        concat3 = torch.cat((upconv3, skip2, disp4_up), 1)
        iconv3 = self.iconv3(concat3)
        disp3 = self.alpha * self.predict_disp3(iconv3) + self.beta

        upconv2 = crop_like(self.upconv2(iconv3), skip1)
        #disp3_up = upsample_nn_nearest(disp3)
        disp3_up = crop_like(F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), skip1)
        concat2 = torch.cat((upconv2, skip1, disp3_up), 1)
        iconv2 = self.iconv2(concat2)
        disp2 = self.alpha * self.predict_disp2(iconv2) + self.beta

        upconv1 = crop_like(self.upconv1(iconv2), x)
        #disp2_up = upsample_nn_nearest(disp2)
        disp2_up = crop_like(F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        concat1 = torch.cat((upconv1, disp2_up), 1)
        iconv1 = self.iconv1(concat1)
        disp1 = self.alpha * self.predict_disp1(iconv1) + self.beta

        if self.training:
            return disp1, disp2, disp3, disp4
        else:
            return disp1

#******* about resdual block used in resblock function
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
        	identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)

        return out
#residual block


