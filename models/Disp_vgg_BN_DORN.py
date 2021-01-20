import torch.cuda
import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torchvision.models as models
import pdb
#from .model_utils import * #use . represent relative address
#from utils.util_functions import unsqueeze_dim0_tensor

def upsample_nn_nearest(x):
    return F.upsample(x, scale_factor=2, mode='nearest')
#    return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
def initilize_modules(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

def Conv2dBlock2(c_in, c_out, k_size, stride, padding, leaky=True):
    if leaky:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, k_size, stride, padding),
            nn.LeakyReLU(0.1),
            nn.Conv2d(c_out, c_out, k_size, 1, padding),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
        	nn.Conv2d(c_in, c_out, k_size, stride, padding),
	        nn.ReLU(inplace=True),
	        nn.Conv2d(c_out, c_out, k_size, 1, padding),
	        nn.ReLU(inplace=True)
	    )


def Conv2dBlock1(c_in, c_out, k_size, stride, padding, leaky=True):
    if leaky:
    	return nn.Sequential(
            nn.Conv2d(c_in, c_out, k_size, stride, padding),
            nn.LeakyReLU(0.1)
        )
    else:
    	return nn.Sequential(
            nn.Conv2d(c_in, c_out, k_size, stride, padding),
            nn.ReLU(inplace=True)
        )


def ConvTranspose2dBlock1(c_in, c_out, k_size, stride, padding, output_padding, leaky=True):

    if leaky:
    	return nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, k_size, stride, padding, output_padding),
            nn.LeakyReLU(0.1)
        )
    else:
    	return nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, k_size, stride, padding, output_padding),
            nn.ReLU(inplace=True)
        )

def predict_disp(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )

class Disp_vgg_BN_DORN(nn.Module):
    def __init__(self, datasets ='kitti', ordinal_c=71):
        super(Disp_vgg_BN_DORN, self).__init__()
        self.only_train_dec = False

        if datasets == 'kitti':
            self.alpha = 10
            self.beta = 0.01
        elif datasets == 'nyu':
            self.alpha = 10#not sure about this number choice(I just think nyu should be more detailed)
            self.beta = 0.1

        self.features = models.vgg16_bn(pretrained=False)
        # self.vgg16_model = models.vgg16(pretrained=True)
        # self.conv1 = self.vgg16_model._modules['features'][0:5]
        # self.conv2 = self.vgg16_model._modules['features'][5:10]
        # self.conv3 = self.vgg16_model._modules['features'][10:17]
        # self.conv4 = self.vgg16_model._modules['features'][17:24]
        # self.conv5 = self.vgg16_model._modules['features'][24:31]

        self.upconv4 = ConvTranspose2dBlock1(512, 256, 4, 2, 1, 0)
        self.iconv4 = Conv2dBlock1(256 + 512, 256, 3, 1, 1)

        self.upconv3 = ConvTranspose2dBlock1(256, 128, 4, 2, 1, 0)
        self.iconv3 = Conv2dBlock1(128 + 256, 128, 3, 1, 1)

        self.upconv2 = ConvTranspose2dBlock1(128, 64, 4, 2, 1, 0)
        self.iconv2 = Conv2dBlock1(64 + 128 + 1, 64, 3, 1, 1)

        self.upconv1 = ConvTranspose2dBlock1(64, 32, 4, 2, 1, 0)
        self.iconv1 = Conv2dBlock1(32 + 64 + 1, 32, 3, 1, 1)

        self.upconv0 = ConvTranspose2dBlock1(32, 16, 4, 2, 1, 0)
        self.iconv0 = Conv2dBlock1(16 + 1, 16, 3, 1, 1)

        self.disp3 = predict_disp(128)
        self.disp2 = predict_disp(64)
        self.disp1 = predict_disp(32)

        self.dropout = nn.Dropout2d(p=0.5)
        self.conv_ord = nn.Conv2d(16, 2*ordinal_c, 1)#142
        self.orl = OrdinalRegressionLayer()

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
            self.load_vgg_params(model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'))
        else:
            print("do not load pretrained weights for the monocular model")

    def load_vgg_params(self, params):
        model_dict = self._modules['features'].state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in params.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self._modules['features'].load_state_dict(model_dict)

    def forward(self, x):
        conv1 = self.features._modules['features'][0:7](x)
        conv2 = self.features._modules['features'][7:14](conv1)
        conv3 = self.features._modules['features'][14:24](conv2)
        conv4 = self.features._modules['features'][24:34](conv3)
        conv5 = self.features._modules['features'][34:44](conv4)

        # conv1 = self.conv1(x)
        # conv2 = self.conv2(conv1)
        # conv3 = self.conv3(conv2)
        # conv4 = self.conv4(conv3)
        # conv5 = self.conv5(conv4)

        if self.only_train_dec:
            conv1 = conv1.detach()
            conv2 = conv2.detach()
            conv3 = conv3.detach()
            conv4 = conv4.detach()
            conv5 = conv5.detach()

        skip1 = conv1
        skip2 = conv2
        skip3 = conv3
        skip4 = conv4

        upconv4 = self.upconv4(conv5)#;pdb.set_trace() # H/16
        concat4 = torch.cat((upconv4, skip4), 1)
        iconv4  = self.iconv4(concat4)

        upconv3 = self.upconv3(iconv4)  # H/8
        concat3 = torch.cat((upconv3, skip3), 1)
        iconv3  = self.iconv3(concat3)
        disp3   = self.alpha * self.disp3(iconv3)+self.beta
        disp3up = upsample_nn_nearest(disp3)

        upconv2 = self.upconv2(iconv3)  # H/4
        concat2 = torch.cat((upconv2, skip2, disp3up), 1)
        iconv2  = self.iconv2(concat2)
        disp2   = self.alpha * self.disp2(iconv2)+self.beta
        disp2up = upsample_nn_nearest(disp2)

        upconv1 = self.upconv1(iconv2)  # H/2
        concat1 = torch.cat((upconv1, skip1, disp2up), 1)
        iconv1  = self.iconv1(concat1)
        disp1   = self.alpha * self.disp1(iconv1)+self.beta
        disp1up = upsample_nn_nearest(disp1)

        upconv0 = self.upconv0(iconv1)
        concat0 = torch.cat((upconv0, disp1up), 1)
        iconv0  = self.iconv0(concat0)
        
        pre_ord = self.conv_ord(self.dropout(iconv0))
        depth_labels, ord_labels = self.orl(pre_ord)

        return depth_labels, ord_labels

class OrdinalRegressionLayer(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLayer, self).__init__()

    def forward(self, x):
        """
        :param x: N X H X W X C, N is batch_size, C is channels of features
        :return: ord_labels is ordinal outputs for each spatial locations , size is N x H X W X C (C = 2K, K is interval of SID)
                 decode_label is the ordinal labels for each position of Image I
        """
        N, C, H, W = x.size()
        ord_num = C // 2

        """
        replace iter with matrix operation
        fast speed methods
        """
        A = x[:, ::2, :, :].clone()
        B = x[:, 1::2, :, :].clone()

        A = A.view(N, 1, ord_num * H * W)
        B = B.view(N, 1, ord_num * H * W)

        C = torch.cat((A, B), dim=1)
        C = torch.clamp(C, min=1e-8, max=1e8)  # prevent nans

        ord_c = nn.functional.softmax(C, dim=1)

        ord_c1 = ord_c[:, 1, :].clone()
        ord_c1 = ord_c1.view(-1, ord_num, H, W)
        decode_c = torch.sum((ord_c1 > 0.5), dim=1).view(-1, 1, H, W)
        return decode_c, ord_c1
