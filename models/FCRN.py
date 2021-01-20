# adapted from https://github.com/XPFly1989/FCRN

import torch
import torch.nn as nn
import torch.nn.functional
import math
from torch.nn.init import xavier_uniform_, zeros_
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
### fcrn for depth ###
# based on ResNet50
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class UpProject(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpProject, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, (2, 3))
        self.conv1_3 = nn.Conv2d(in_channels, out_channels, (3, 2))
        self.conv1_4 = nn.Conv2d(in_channels, out_channels, 2)

        self.conv2_1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv2_2 = nn.Conv2d(in_channels, out_channels, (2, 3))
        self.conv2_3 = nn.Conv2d(in_channels, out_channels, (3, 2))
        self.conv2_4 = nn.Conv2d(in_channels, out_channels, 2)

        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.bn1_2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        batch_size = x.size()[0]
        out1_1 = self.conv1_1(nn.functional.pad(x, (1, 1, 1, 1)))
        #out1_2 = self.conv1_2(nn.functional.pad(x, (1, 1, 0, 1)))#right interleaving padding
        out1_2 = self.conv1_2(nn.functional.pad(x, (1, 1, 1, 0)))#author's interleaving pading in github
        #out1_3 = self.conv1_3(nn.functional.pad(x, (0, 1, 1, 1)))#right interleaving padding
        out1_3 = self.conv1_3(nn.functional.pad(x, (1, 0, 1, 1)))#author's interleaving pading in github
        #out1_4 = self.conv1_4(nn.functional.pad(x, (0, 1, 0, 1)))#right interleaving padding
        out1_4 = self.conv1_4(nn.functional.pad(x, (1, 0, 1, 0)))#author's interleaving pading in github

        out2_1 = self.conv2_1(nn.functional.pad(x, (1, 1, 1, 1)))
        #out2_2 = self.conv2_2(nn.functional.pad(x, (1, 1, 0, 1)))#right interleaving padding
        out2_2 = self.conv2_2(nn.functional.pad(x, (1, 1, 1, 0)))#author's interleaving pading in github
        #out2_3 = self.conv2_3(nn.functional.pad(x, (0, 1, 1, 1)))#right interleaving padding
        out2_3 = self.conv2_3(nn.functional.pad(x, (1, 0, 1, 1)))#author's interleaving pading in github
        #out2_4 = self.conv2_4(nn.functional.pad(x, (0, 1, 0, 1)))#right interleaving padding
        out2_4 = self.conv2_4(nn.functional.pad(x, (1, 0, 1, 0)))#author's interleaving pading in github

        height = out1_1.size()[2]
        width = out1_1.size()[3]

        out1_1_2 = torch.stack((out1_1, out1_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)
        out1_3_4 = torch.stack((out1_3, out1_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)

        out1_1234 = torch.stack((out1_1_2, out1_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
            batch_size, -1, height * 2, width * 2)

        out2_1_2 = torch.stack((out2_1, out2_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)
        out2_3_4 = torch.stack((out2_3, out2_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)

        out2_1234 = torch.stack((out2_1_2, out2_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
            batch_size, -1, height * 2, width * 2)

        out1 = self.bn1_1(out1_1234)
        out1 = self.relu(out1)
        out1 = self.conv3(out1)
        out1 = self.bn2(out1)

        out2 = self.bn1_2(out2_1234)

        out = out1 + out2
        out = self.relu(out)

        return out


class FCRN(nn.Module):

    def __init__(self, datasets ='kitti'):
        super(FCRN, self).__init__()
        if datasets == 'kitti':
            self.alpha = 10
            self.beta = 0.01
        elif datasets == 'nyu':
            self.alpha = 10#not sure about this number choice(I just think nyu should be more detailed)
            self.beta = 0.1

        self.inplanes = 64
        # self.n_classes = n_classes

        # ResNet with out avrgpool & fc
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # Up-Conv layers
        self.conv2 = nn.Conv2d(2048, 1024, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1024)

        self.up1 = self._make_upproj_layer(UpProject, 1024, 512)
        self.up2 = self._make_upproj_layer(UpProject, 512, 256)
        self.up3 = self._make_upproj_layer(UpProject, 256, 128)
        self.up4 = self._make_upproj_layer(UpProject, 128, 64)

        self.drop = nn.Dropout2d()

        self.conv3 = nn.Conv2d(64, 1, 3, padding=1)
        self.Sigmoid = nn.Sigmoid()
        # initialize
    def init_weights(self, use_pretrained_weights=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if use_pretrained_weights:
            print("loading pretrained weights downloaded from pytorch.org")
            resnet50 = models.resnet50(pretrained=True)
            init_state_dict = self.init_resnet50_params(resnet50)
            self.load_state_dict(init_state_dict, strict=False)
        else:
            print("do not load pretrained weights for the monocular model")
        #mine
    # def init_weights(self, use_pretrained_weights=False):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
    #             torch.nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 torch.nn.init.constant_(m.bias, 0)
    #             elif isinstance(m, nn.BatchNorm2d):
    #                 torch.nn.init.constant_(m.weight, 1)
    #                 torch.nn.init.constant_(m.bias, 0)
    #     if use_pretrained_weights:
    #         print("loading pretrained weights downloaded from pytorch.org")
    #         self.load_res_params(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    #     else:
    #         print("do not load pretrained weights for the monocular model")

    def init_resnet50_params(self, resnet50):
        initial_state_dict = resnet50.state_dict()
        return initial_state_dict

    def load_res_params(self, params):
        model_dict = self.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in params.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.load_state_dict(model_dict)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_upproj_layer(self, block, in_channels, out_channels):
        return block(in_channels, out_channels)

    def forward(self, x):
        inp_shape = x.shape[2:]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        x = self.drop(x)

        x = self.conv3(x)
        #x = self.relu(x)
        #try with original last layer for this whole code base
        x = self.alpha*self.Sigmoid(x)+self.beta
        

        x = nn.functional.interpolate(x, size=inp_shape, mode='bilinear', align_corners=True)

        # for the convenience of loss function, turn the result shape to become same as other network
        if self.training:
            return [x]
        else:
            return x

    # if you want to load from downloaded pretrained model:
    # def init_resnet50_params(self, model_path):
    #     init_state_dict = torch.load(model_path)
#     return init_state_dict


# def predict_disp(in_planes):
#     return nn.Sequential(
#         nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
#         nn.Sigmoid()
#     )