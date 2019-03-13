import torch
import torch.nn as nn
import torch.nn.functional as F


class Flat(nn.Module):
    def __init__(self):
        super(Flat, self).__init__()

    def forward(self, x):
        return x.view(x.size(0),-1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.layers = nn.ModuleList([
            create_module(['conv', out_channels, 3], in_channels),
            create_module(['relu']),
            create_module(['norm2d'], out_channels),
            create_module(['pad2d', 1])
        ])

        self.layers.append(create_module(['conv', out_channels, 3], out_channels))
        self.layers.append(create_module(['relu']))
        self.layers.append(create_module(['norm2d'], out_channels))
        self.layers.append(create_module(['pad2d', 1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DownConvBlock(ConvBlock):
    def __init__(self, in_channels, out_channels):
        super(DownConvBlock, self).__init__(in_channels, out_channels)

        self.pool = create_module(['max2d',2])

    def forward(self, x):
        f = super(DownConvBlock, self).forward(x)
        x = self.pool(f)
        return x,f


class UpConvBlock(ConvBlock):
    def __init__(self, in_channels, out_channels, side_channels=None):
        channels = int(in_channels/2)
        if side_channels is None:
            side_channels = channels

        super(UpConvBlock, self).__init__(channels + side_channels, out_channels)

        self.tconv = create_module(['tconv', channels, 2, 2], in_channels)

    def forward(self, x1, x2):
        x1 = self.tconv(x1)

        x2 = F.interpolate(x2, x1.shape[2:], mode='bilinear', align_corners=True)
        # th,tw = x1.shape[2:]
        # h,w = x2.shape[2:]
        # h = h//2-th//2
        # w = w//2-tw//2
        # x2 = x2[:,:,h:h+th,w:w+tw]
        x = torch.cat([x1,x2], dim=1)
        x = super(UpConvBlock, self).forward(x)
        return x


MODULES = {
    'nn': {
        'conv': nn.Conv2d,
        'linear': nn.Linear,
        'norm1d': nn.BatchNorm1d,
        'norm2d': nn.BatchNorm2d,
        'cblock': ConvBlock,
        'dblock': DownConvBlock,
        'ublock': UpConvBlock,
        'tconv': nn.ConvTranspose2d
    },
    'f': {
        'max2d': nn.MaxPool2d,
        'relu': nn.ReLU,
        'drop1d': nn.Dropout,
        'drop2d': nn.Dropout2d,
        'flat': Flat,
        'pad2d': nn.ZeroPad2d
    }
}


def create_module(config, in_features=None):
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
