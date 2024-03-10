import torch.nn as nn
import torch
import pretrainedmodels


class PPNet(nn.Module):
    def __init__(self, base_encoder: str, pretrained=True, num_classes=2):
        super(PPNet, self).__init__()

        # create the encoder
        if pretrained:
            print("use pretrained model!")
            self.encoder = pretrainedmodels.__dict__[base_encoder](pretrained='imagenet')
        else:
            self.encoder = pretrainedmodels.__dict__[base_encoder](pretrained=None)

        self.inp_encoder = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(*list(self.encoder.children())[1:-2])

        self.classify_fc = nn.Linear(512 * 4, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x1, mask):
        '''baseline'''
        x1 = torch.cat((x1, mask), dim=1)
        x1 = self.inp_encoder(x1)
        z1 = self.encoder(x1)
        z1 = self.avgpool(z1)
        z1 = z1.view(z1.size(0), -1)
        t1 = self.classify_fc(z1)

        return t1


class emptyLayer(nn.Module):
    def __init__(self):
        super(emptyLayer, self).__init__()

    def forward(self, x):
        return x
