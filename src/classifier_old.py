import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules import create_module
from tqdm import tqdm


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()

        self.config = config

        i = 1
        self.in_channels = config[i][0]
        self.image_size = config[i][1:]

        i += 1
        self.use_masks = (config[i][0] == 1)
        if self.use_masks:
            self.crop_size = config[i][1]
        else:
            self.crop_size = self.image_size[0]

        x = torch.rand(2, self.in_channels, self.crop_size, self.crop_size)
        self.layers = nn.ModuleList()
        i += 1
        while i < len(config):
            self.layers.append(create_module(config[i], x.size(1)))
            with torch.no_grad():
                x = self.layers[-1](x)
            i += 1

        self.is_cuda = False
        self.crop_filter = torch.ones(1,1,1,self.crop_size)
        self.crop_filter_t = self.crop_filter.transpose(-2,-1)

    def cuda(self, device=None):
        self.is_cuda = True
        self.crop_filter = self.crop_filter.cuda()
        self.crop_filter_t = self.crop_filter_t.cuda()
        return super(Classifier, self).cuda(device)

    def forward(self, x, z=None, debug=False):
        if self.use_masks and z is not None:
            x = self.crop_images(x,z)
            # x = crop_images_old(x, z, self.crop_size, self.is_cuda)

        if debug:
            outputs = [x]
        for layer in self.layers:
            x = layer(x)
            if debug:
                outputs.append(x)

        if debug:
            return x, outputs
        else:
            return x

    def crop_images(self, x, z):
        # zf = F.conv1d(z,self.crop_filter)
        # zf = F.conv1d(z,self.crop_filter_t)
        # zf = (zf == zf.max(-1,keepdim=True)[0].max(-2,keepdim=True)[0])
        # nz = [zf[i].nonzero() for i in range(zf.shape[0])]
        # p = [k[k.shape[0]//2][1:].min(torch.Tensor([z.shape[-2]-self.crop_size, z.shape[-1]-self.crop_size]).cuda().long()) for k in nz]
        # xf = torch.stack([x[i,:,p[i][0]:p[i][0]+self.crop_size,p[i][1]:p[i][1]+self.crop_size] for i in range(zf.shape[0])])
        zf = F.conv1d(z,self.crop_filter)
        zf = F.conv1d(zf,self.crop_filter_t)
        _,zi = zf.view(zf.shape[0],-1).max(-1,keepdim=True)
        py,px = zi//zf.shape[-1],zi%zf.shape[-1]
        xf = torch.stack([x[i,:,py[i]:py[i]+self.crop_size,px[i]:px[i]+self.crop_size] for i in range(zf.shape[0])])
        # zi = zf.view(zf.shape[0],-1).argmax(-1)
        # py = (zi//zf.shape[-1]).min(torch.Tensor([z.shape[-2]-self.crop_size]).cuda().long())
        # px = (zi%zf.shape[-1]).min(torch.Tensor([z.shape[-1]-self.crop_size]).cuda().long())
        # xf = torch.stack([x[i,:,py[i]:py[i]+self.crop_size,px[i]:px[i]+self.crop_size] for i in range(zf.shape[0])])
        return xf

    def preprocess(self, x, z=None, y=None):
        batch_size = x.size(0)

        zf = F.conv1d(z,self.crop_filter)
        zf = F.conv1d(zf,self.crop_filter_t)
        _,zi = zf.view(batch_size,-1).max(-1,keepdim=True)
        py,px = zi//zf.shape[-1],zi%zf.shape[-1]

        tx = ((2*px + self.crop_size).float()/x.size(-1) - 1)
        ty = ((2*py + self.crop_size).float()/x.size(-2) - 1)
        x_scale = self.crop_size/x.size(-1)
        y_scale = self.crop_size/x.size(-2)
        if self.augment and self.training:
            tx += (torch.rand(batch_size, 1)*2-1).cuda()/10
            ty += (torch.rand(batch_size, 1)*2-1).cuda()/10
            rotation_angle = torch.rand(batch_size).cuda()*np.pi*2
        else:
            rotation_angle = torch.zeros(batch_size).cuda()

        scale_factor = torch.Tensor([[x_scale,x_scale],[y_scale,y_scale]]).cuda().expand(batch_size,2,2)
        rcos = torch.cos(rotation_angle).view(batch_size,1,1)
        rsin = torch.sin(rotation_angle).view(batch_size,1,1)

        rotation = torch.cat([torch.cat([rcos,rsin], dim=2), torch.cat([-rsin,rcos], dim=2)], dim=1)

        translation = torch.cat([tx,ty], dim=-1).unsqueeze(-1)
        transform = torch.cat([scale_factor*rotation, translation], dim=2)

        grid = F.affine_grid(transform, (batch_size, 1, self.crop_size, self.crop_size))
        xf = F.grid_sample(x, grid, padding_mode='border')

        return xf, y

    def run_epoch(self, mode, batches, epoch, criterion=None, optimizer=None, writer=None, log_interval=None):
        if mode == 'train':
            self.train()
        else:
            self.eval()

        loss = 0.0
        correct_predictions = 0
        data_size = 0
        i = 0
        for data in tqdm(batches, desc='Epoch {}: '.format(epoch), total=len(batches)):
            if self.is_cuda:
                for k in range(len(data)):
                    data[k] = data[k].cuda()
            if len(data) > 2:
                x,y,z = data[:3]
            else:
                x,y = data
                z = None

            if mode == 'train':
                optimizer.zero_grad()

                # Forward Pass
                logits = self.forward(x,z)
                batch_loss = criterion(logits, y)

                # Backward Pass
                batch_loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    # Forward Pass
                    logits = self.forward(x,z)
                    batch_loss = criterion(logits, y)

            # Update metrics
            loss += batch_loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == y.long()).sum().item()
            data_size += x.shape[0]

            if mode == 'train' and (log_interval is not None) and (i % log_interval == 0):
                writer.add_scalar('c/{}_loss'.format(mode), batch_loss.item(), epoch*len(batches)+i)
            i += 1

        loss = loss/len(batches)
        accuracy = correct_predictions/data_size
        if writer is not None:
            writer.add_scalar('c/{}_acc'.format(mode), accuracy, epoch)
            if mode == 'valid':
                writer.add_scalar('c/{}_loss'.format(mode), loss, epoch)
        return {'loss': loss, 'acc': accuracy}

    def get_criterion(self):
        return nn.CrossEntropyLoss()

    def predict(self, batches, labels=True):
        predictions = None
        if labels:
            y_true = None
        with torch.no_grad():
            for data in tqdm(batches):
                if self.is_cuda:
                    for k in range(len(data)):
                        data[k] = data[k].cuda()
                if labels:
                    if len(data) == 3:
                        x,y,z = data
                    else:
                        x,y = data
                        z = None
                else:
                    if len(data) > 1:
                        x,z = data[:2]
                    else:
                        x = data[0]
                        z = None

                # Forward Pass
                logits = self.forward(x,z)

                # Update metrics
                if predictions is None:
                    predictions = torch.argmax(logits,dim=1)
                    if labels:
                        y_true = y
                else:
                    predictions = torch.cat([predictions, torch.argmax(logits,dim=1)])
                    if labels:
                        y_true = torch.cat([y_true, y])

            if labels:
                accuracy = (predictions == y_true.long()).double().mean().item()
                return {'predictions': predictions, 'acc': accuracy}
            else:
                return {'predictions': predictions}


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