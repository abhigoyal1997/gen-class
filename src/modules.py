import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

    def cuda(self, device=None):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return super(Normalize, self).cuda(device)


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
    def __init__(self, in_channels, out_channels, block_type='conv', activation='relu', first_block=False, strides=[1,1]):
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

        self.pool = create_module(['max2d',2])

    def forward(self, x):
        f = super(DownBlock, self).forward(x)
        x = self.pool(f)
        return x,f


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


class Crop(nn.Module):
    def __init__(self, in_size, out_size=None, crop_size=None, square=True, augment=False):
        super(Crop, self).__init__()

        self.in_size = in_size
        self.out_size = out_size if out_size is not None else self.in_size
        self.crop_size = crop_size if crop_size is not None else self.out_size

        self.augment = augment
        self.square = square
        self.is_cuda = False

    def cuda(self, device=None):
        self.is_cuda = True
        return super(Crop, self).cuda(device)

    def find_bb(self, z, threshold=10):
        zv = z.cumsum(-2)
        height = zv[:,0,-1,:].max(1)[0].int()

        zvf = z.flip(-2).cumsum(-2).flip(-2)
        zvf[zvf<20] = z.shape[-2]
        zvc = (zvf-zv).abs()<threshold

        zh = z.cumsum(-1)
        width = zh[:,0,:,-1].max(1)[0].int()

        zhf = z.flip(-1).cumsum(-1).flip(-1)
        zhf[zhf<20] = z.shape[-1]
        zhc = (zhf-zh).abs()<threshold

        zpos = zhc*zvc

        zpos = zpos.view(zpos.shape[0],-1).argmax(1)
        py,px = (zpos//z.shape[-1]).int(),(zpos%z.shape[-1]).int()

        if self.square:
            height = torch.max(height,width)
            width = height

        return py,px,height,width

    def forward(self, x, z):
        with torch.no_grad():
            batch_size = x.size(0)

            py,px,height,width = self.find_bb(z)

            tx = ((2*px).float()/x.size(-1) - 1)
            ty = ((2*py).float()/x.size(-2) - 1)
            x_scale = width.float()/x.size(-1)
            y_scale = height.float()/x.size(-2)

            if self.augment and self.training:
                tx += (torch.randn_like(tx)*2-1).cuda()/40
                ty += (torch.randn_like(ty)*2-1).cuda()/40
                rotation_angle = torch.rand(batch_size).cuda()*np.pi*2/20
            else:
                rotation_angle = torch.zeros(batch_size).cuda()

            scale_factor = torch.stack([torch.stack([x_scale,x_scale],dim=-1), torch.stack([y_scale,y_scale],dim=-1)],dim=-2)

            rcos = torch.cos(rotation_angle).view(batch_size,1,1)
            rsin = torch.sin(rotation_angle).view(batch_size,1,1)

            rotation = torch.cat([torch.cat([rcos,rsin], dim=2), torch.cat([-rsin,rcos], dim=2)], dim=1)

            translation = torch.stack([tx,ty], dim=-1).unsqueeze(-1)

            transform = torch.cat([scale_factor*rotation, translation], dim=2)

            grid = F.affine_grid(transform, (batch_size, 1, self.out_size, self.out_size))
            xf = F.grid_sample(x, grid, padding_mode='border')

        return xf


class STN(nn.Module):
    def __init__(self, in_size, out_size=None, augment=False, cuda=True):
        super(STN, self).__init__()

        self.in_size = in_size
        self.out_size = out_size if out_size is not None else self.in_size
        self.is_cuda = cuda

        x = torch.rand(2, 1, self.in_size, self.in_size)
        if self.is_cuda:
            x = x.cuda()

        self.localization = nn.Sequential(
            create_module(['conv', 8, 7], 1),
            create_module(['max2d',2]),
            create_module(['relu']),
            create_module(['conv', 10, 5], 8),
            create_module(['max2d',2]),
            create_module(['relu'])
        )
        if self.is_cuda:
            self.localization = self.localization.cuda()

        self.flat = create_module(['flat'])
        x = self.flat(self.localization(x))

        self.fc = nn.Sequential(
            create_module(['linear', 32], x.size(1)),
            create_module(['relu']),
            create_module(['linear', 3*2], 32),
        )
        if self.is_cuda:
            self.fc = self.fc.cuda()

        self.fc[2].weight.data.zero_()
        self.fc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.augment = augment

    def forward(self, x, z, debug=False):
        xs = self.flat(self.localization(z))
        theta = self.fc(xs)
        theta = theta.view(-1, 2, 3)

        if self.augment:
            pass  # TODO: implement data augmentation

        if self.out_size is None:
            grid = F.affine_grid(theta, x.shape)
        else:
            grid = F.affine_grid(theta, (*x.shape[:2],self.out_size, self.out_size))
        x = F.grid_sample(x, grid, padding_mode='border')

        if debug:
            return x, theta
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
        'tconv': nn.ConvTranspose2d,
        'stn': STN,
        'crop': Crop
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


def create_module(config, args=None, cuda=True):
    if config[0] == 'resnet18':
        module = models.resnet18(True)
        module.fc = nn.Linear(512, int(config[1]))
    elif config[0] == 'resnet50':
        module = models.resnet50(True)
        module.fc = nn.Linear(2048, int(config[1]))
    else:
        try:
            if config[0] in MODULES['nn']:
                module = MODULES['nn'][config[0]](args, *config[1:])
            else:
                module = MODULES['f'][config[0]](*config[1:])
        except KeyError:
            print('Module {} not found!'.format(config[0]))
            raise KeyError
        except Exception:
            print('Error while creating module {}'.format(config[0]))
            raise Exception

    if cuda:
        module = module.cuda()

    return module
