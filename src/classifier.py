import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules import create_module
from tqdm import tqdm


class Classifier(nn.Module):
    def __init__(self, config, cuda=True):
        super(Classifier, self).__init__()

        self.config = config
        self.is_cuda = cuda

        i = 1
        self.in_channels = config[i][0]
        self.in_size = config[i][1]
        self.augment = config[i][2]
        i += 1
        x = torch.rand(2, self.in_channels, self.in_size, self.in_size)

        if self.is_cuda:
            x = x.cuda()

        if config[i][0] != 'none':
            self.loc = create_module(config[i], self.in_size, self.is_cuda)
            self.loc.augment = self.augment
            x = self.loc(x,x[:,0,None,:,:])
        else:
            self.loc = None

        layers = []
        i += 1
        while i < len(config):
            layers.append(create_module(config[i], x.size(1), self.is_cuda))
            with torch.no_grad():
                x = layers[-1](x)
            i += 1

        self.net = nn.Sequential(*layers)

        self.out_features = x.size(1)

    def cuda(self, device=None):
        self.is_cuda = True
        if self.loc is not None:
            self.loc = self.loc.cuda()

        return super(Classifier, self).cuda(device)

    def forward(self, x, z=None, return_loc=False):
        if self.loc is not None and z is not None:
            x = self.loc(x,z)

        out = self.net(x)

        if return_loc:
            return out, x
        else:
            return out

    def augment_batch(self, x, y):
        if self.training:
            x = torch.cat([x, x.flip(-1)])

            scale_ratio = self.image_size/self.crop_size
            scale = torch.Tensor([[scale_ratio,0],[0,scale_ratio]]).cuda()
            translation = ((torch.rand(x.size(0),2)*2-1)*(1-scale_ratio)).cuda().unsqueeze(dim=-1)
            transform = torch.cat([scale.expand(x.size(0),*scale.size()), translation], dim=-1)

            grid = F.affine_grid(transform, (*x.shape[:2], self.image_size, self.image_size))
            x = F.grid_sample(x, grid, padding_mode='border')

            y = torch.cat([y]*(x.size(0)//y.size(0))).long()
        else:
            scale_ratio = self.image_size/self.crop_size
            transform = torch.Tensor([[scale_ratio,0,0],[0,scale_ratio,0]]).cuda().expand(x.size(0),2,3)

            grid = F.affine_grid(transform, (*x.shape[:2], self.image_size, self.image_size))
            x = F.grid_sample(x, grid)
        return (x,y)

    def run_epoch(self, mode, batches, epoch=None, criterion=None, optimizer=None, writer=None, log_interval=None):
        if mode == 'train':
            self.train()
        else:
            self.eval()

        loss = 0.0
        correct_predictions = 0
        data_size = 0
        i = 0
        if epoch is not None:
            batches = tqdm(batches, desc='Epoch {}: '.format(epoch), total=len(batches))
        else:
            batches = tqdm(batches, total=len(batches))

        for data in batches:
            if self.is_cuda:
                for k in range(len(data)):
                    data[k] = data[k].cuda().float()
            if len(data) > 2:
                x,y,z = data[:3]
                if self.augment and self.training:
                    x = torch.cat([x,x.flip(-1)],dim=0)
                    z = torch.cat([z,z.flip(-1)],dim=0)
                    y = torch.cat([y]*2,dim=0)
            else:
                x,y = data
                z = None
                if self.augment and self.training:
                    x = torch.cat([x,x.flip(-1)],dim=0)
                    y = torch.cat([y]*2,dim=0)

            y = y.long()

            with torch.set_grad_enabled(self.training):
                if self.training:
                    optimizer.zero_grad()

                # Forward Pass
                logits = self.forward(x,z)
                if logits.shape[0] != y.shape[0]:
                    y = torch.cat([y]*2,dim=0)
                batch_loss = criterion(logits, y)
                predictions = logits.argmax(-1)

                if self.training:
                    # Backward Pass
                    batch_loss.backward()
                    optimizer.step()

            # Update metrics
            loss += (batch_loss*y.shape[0]).item()
            correct_predictions += (predictions.long() == y.long()).sum().item()
            data_size += y.shape[0]

            if self.training and (log_interval is not None) and (i % log_interval == 0):
                writer.add_scalar('c/{}_loss'.format(mode), loss/data_size, epoch*len(batches)+i)
            i += 1

        loss = loss/data_size
        accuracy = correct_predictions/data_size

        if writer is not None:
            writer.add_scalar('c/{}_acc'.format(mode), accuracy, epoch)
            if mode == 'valid':
                writer.add_scalar('c/{}_loss'.format(mode), loss, epoch)

        return {'loss': loss, 'acc': accuracy}

    def get_criterion(self, no_reduction=False):
        if no_reduction:
            if self.out_features == 1:
                return nn.BCEWithLogitsLoss(reduction='none')
            else:
                return nn.CrossEntropyLoss(reduction='none')
        else:
            if self.out_features == 1:
                return nn.BCEWithLogitsLoss()
            else:
                return nn.CrossEntropyLoss()


def crop_images_old(img, mask, sz=(80,80), cuda=True):
    mask = mask.byte().squeeze().cpu().numpy()
    cnt = [cv.findContours(x,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[1] for x in mask]
    pts = [torch.Tensor([[np.min(ci.squeeze(axis=1),axis=0), np.max(ci.squeeze(axis=1),axis=0)] for ci in c]).int() for c in cnt]

    size = [(p[:,1,:]-p[:,0,:]).prod(dim=1) if p.dim() > 1 else torch.Tensor([0]) for p in pts]
    i = [sz.argmax().item() for sz in size]
    tx = torch.Tensor([pts[j][i[j],0,0].item() if pts[j].dim() > 1 else 0 for j in range(len(i))])
    ty = torch.Tensor([pts[j][i[j],0,1].item() if pts[j].dim() > 1 else 0 for j in range(len(i))])
    bx = torch.Tensor([pts[j][i[j],1,0].item() if pts[j].dim() > 1 else img.shape[3] for j in range(len(i))])
    by = torch.Tensor([pts[j][i[j],1,1].item() if pts[j].dim() > 1 else img.shape[2] for j in range(len(i))])

    if cuda:
        tx = tx.cuda()
        ty = ty.cuda()
        bx = bx.cuda()
        by = by.cuda()

    tx = torch.clamp((tx+bx)/2-sz[1]/2, 0, img.shape[3] - sz[1]).long()
    ty = torch.clamp((ty+by)/2-sz[0]/2, 0, img.shape[2] - sz[0]).long()

    attention_img = torch.cat([img[i, None, :, ty[i]:ty[i]+sz[1], tx[i]:tx[i]+sz[0]] for i in range(img.size(0))])
    return attention_img
