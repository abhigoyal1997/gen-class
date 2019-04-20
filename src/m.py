import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class Normalize(nn.Module):
    def __init__(self, mean=None, std=None):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
        if self.mean is None:
            self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        if self.std is None:
            self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    def forward(self, x):
        if x.size(1) == 1:
            x = torch.cat([x]*3, dim=1)
        return (x-self.mean)/self.std

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()


class CNormalize(nn.Module):
    def __init__(self, out_features=3):
        super(CNormalize, self).__init__()
        if out_features == 3:
            self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
            self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
            self.conv = nn.Conv2d(1, 3, 1)
        else:
            self.mean = torch.Tensor([0.5]).view(1,1,1,1)
            self.std = torch.Tensor([0.5]).view(1,1,1,1)
            self.conv = nn.Conv2d(1, 1, 1)

        self.conv.weight.data = (1/self.std).view_as(self.conv.weight)
        self.conv.bias.data = (-self.mean/self.std).view_as(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class Flat(nn.Module):
    def __init__(self):
        super(Flat, self).__init__()

    def forward(self, x):
        return x.view(x.size(0),-1)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, block_type='conv', activation='relu', first_block=False, strides=[2,1]):
        super(Block, self).__init__()

        layers = []

        if not first_block:
            layers.append(create_module(['norm2d'], in_channels))
            if activation == 'lrelu':
                layers.append(create_module(['lrelu', 0.001]))
            else:
                layers.append(create_module(['relu']))

        layers.append(create_module(['conv', out_channels, 3, strides[0], 1], in_channels))
        layers.append(create_module(['norm2d'], out_channels))
        if activation == 'lrelu':
            layers.append(create_module(['lrelu', 0.001]))
        else:
            layers.append(create_module(['relu']))

        layers.append(create_module(['conv', out_channels, 3, strides[1], 1], out_channels))

        self.block = nn.Sequential(*layers)

        if block_type == 'res':
            short_layers = [
                create_module(['conv', out_channels, 1, strides[0]], in_channels),
                create_module(['norm2d'], out_channels)
            ]

            self.short_block = nn.Sequential(*short_layers)

        self.res_block = (block_type == 'res')

    def forward(self, x):
        if self.res_block:
            return self.short_block(x) + self.block(x)
        else:
            return self.block(x)


class DownBlock(Block):
    def __init__(self, in_channels, out_channels, block_type='conv', first_block=False):
        super(DownBlock, self).__init__(in_channels, out_channels, block_type, 'lrelu', first_block)

        # self.pool = create_module(['max2d',2])

    def forward(self, x):
        f = super(DownBlock, self).forward(x)
        # x = self.pool(f)
        return f,f


class UpBlock(Block):
    def __init__(self, in_channels, out_channels, block_type='conv', side_channels=None, crop=True):
        channels = int(in_channels/2)
        if side_channels is None:
            side_channels = channels
        self.crop = crop

        super(UpBlock, self).__init__(channels + side_channels, out_channels, block_type, 'lrelu', False, [1,1])

        self.tconv = create_module(['tconv', channels, 2, 2], in_channels)

    def center_crop(self, x, cs):
        _, _, h, w = x.size()
        diff_y = (h - cs[0]) // 2
        diff_x = (w - cs[1]) // 2
        return x[:, :, diff_y:(diff_y + cs[0]), diff_x:(diff_x + cs[1])]

    def forward(self, x1, x2):
        x1 = self.tconv(x1)

        if self.crop:
            x2 = self.center_crop(x2, x1.shape[2:])
        else:
            x2 = F.interpolate(x2, x1.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x1,x2], dim=1)
        x = super(UpBlock, self).forward(x)
        return x


MODULES = {
    'nn': {
        'conv': nn.Conv2d,
        'linear': nn.Linear,
        'norm1d': nn.BatchNorm1d,
        'norm2d': nn.BatchNorm2d,
        'block': Block,
        'dblock': DownBlock,
        'ublock': UpBlock,
        'tconv': nn.ConvTranspose2d
    },
    'f': {
        'max2d': nn.MaxPool2d,
        'relu': nn.ReLU,
        'lrelu': nn.LeakyReLU,
        'tanh': nn.Tanh,
        'drop1d': nn.Dropout,
        'drop2d': nn.Dropout2d,
        'flat': Flat,
        'pad2d': nn.ZeroPad2d,
        'normalize': Normalize,
        'cnorm': CNormalize
    }
}


def create_module(config, in_features=None):
    if config[0] == 'resnet18':
        module = models.resnet18(True)
        module.fc = nn.Linear(512, 1)
    elif config[0] == 'resnet50':
        module = models.resnet50(True)
        module.fc = nn.Linear(2048, 1)
    else:
        try:
            if config[0] in MODULES['nn']:
                module = MODULES['nn'][config[0]](in_features, *config[1:])
            else:
                module = MODULES['f'][config[0]](*config[1:])
        except KeyError:
            print('Module {} not found!'.format(config[0]))
            raise KeyError
        except Exception:
            print('Error while creating module {}'.format(config[0]))
            raise Exception

    return module
