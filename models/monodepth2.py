import torch.cuda
import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torchvision.models as models

class monodepth2(nn.Module):
    def __init__(self, encoder, decoder):
        super(monodepth2, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.decoder(features)
        return outputs