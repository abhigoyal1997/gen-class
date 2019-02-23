import torch
import torch.nn as nn
import torch.nn.functional as F


class Flat(nn.Module):
    def __init__(self):
        super(Flat, self).__init__()

    def forward(self, x):
        return x.view(x.size(0),-1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False, kernel_size=3):
        super(ConvBlock, self).__init__()

        self.layers = nn.ModuleList([
            create_module(['conv', out_channels, kernel_size], in_channels),
            create_module(['relu'])
        ])
        if batch_norm:
            self.layers.append(create_module(['norm2d'], out_channels))

        self.layers.append(create_module(['conv', out_channels, kernel_size], out_channels))
        self.layers.append(create_module(['relu']))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DownConvBlock(ConvBlock):
    def __init__(self, in_channels, out_channels, batch_norm=False, kernel_size=3):
        super(DownConvBlock, self).__init__(in_channels, out_channels, kernel_size=kernel_size, batch_norm=batch_norm)

        self.pool = create_module(['max2d',2])

    def forward(self, x):
        f = super(DownConvBlock, self).forward(x)
        x = self.pool(f)
        return x,f


class UpConvBlock(ConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3, side_channels=None):
        channels = int(in_channels/2)
        if side_channels is None:
            side_channels = channels

        super(UpConvBlock, self).__init__(channels + side_channels, out_channels, kernel_size=kernel_size)

        self.tconv = create_module(['tconv', channels, 2, 2], in_channels)

    def forward(self, x1, x2):
        x1 = self.tconv(x1)

        x2 = F.interpolate(x2, x1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1,x2], dim=1)
        x = super(UpConvBlock, self).forward(x)
        return x


MODULES = {
    'nn': {
        'conv': nn.Conv2d,
        'linear': nn.Linear,
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
        'flat': Flat
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
