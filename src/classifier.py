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
            self.crop_size = config[i][1:]
        else:
            self.crop_size = self.image_size

        x = torch.rand(1, self.in_channels, *self.crop_size)
        self.layers = nn.ModuleList()
        i += 1
        while i < len(config):
            self.layers.append(create_module(config[i], x.size(1)))
            with torch.no_grad():
                x = self.layers[-1](x)
            i += 1

        self.is_cuda = False

    def cuda(self, device=None):
        self.is_cuda = True
        return super(Classifier, self).cuda(device)

    def forward(self, x, z=None, debug=False):
        if self.use_masks and z is not None:
            x = crop_images(x,z, self.crop_size)
            if self.is_cuda:
                x = x.cuda()

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


def find_attention_xy(z):
    size = 0
    # tx, ty, bx, by = 0, 0, 0, 0
    # import pudb; pudb.set_trace()
    p = np.asarray([
        [np.min(c.squeeze(axis=1),axis=0), np.max(c.squeeze(axis=1),axis=0)] for c in cv.findContours(z,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[1]
    ])
    size = p[:,1,:]-p[:,0,:]
    i = np.argmax(size[:,0]*size[:,1])
    # for cnt in contours:
    #     cnt = np.squeeze(cnt, axis=1)
    #     x1, y1 = np.min(cnt, axis=0)
    #     x2, y2 = np.max(cnt, axis=0)
    #     if (y2 - y1) * (x2 - x1) > size:
    #         tx, ty = x1, y1
    #         bx, by = x2, y2
    #         size = (y2 - y1) * (x2 - x1)
    return [p[i,0,0], p[i,0,1], p[i,1,0], p[i,1,1]]


def get_attention_img(img, mask, sz=(80,80)):
    cnt = cv.findContours(mask,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[1]
    tx, ty, bx, by = find_attention_xy(cnt)
    if not (tx == 0 and ty == 0 and bx == 0 and by == 0):
        img = img[:, ty:by, tx:bx]

    return F.interpolate(img.unsqueeze(0), size=sz, mode='bilinear', align_corners=True)


def crop_images(img, mask, sz=(80,80)):
    mask = mask.byte().squeeze().cpu().numpy()
    # return torch.cat([get_attention_img(img[i], mask[i], sz) for i in range(img.size(0))])
    # cnt = map(lambda x:cv.findContours(x,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[1], mask)
    cnt = [cv.findContours(x,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[1] for x in mask]
    # attention_xy = map(find_attention_xy, cnt)
    # attention_xy = [find_attention_xy(z) for z in mask]
    pts = [torch.Tensor([[np.min(ci.squeeze(axis=1),axis=0), np.max(ci.squeeze(axis=1),axis=0)] for ci in c]).int() for c in cnt]

    size = [(p[:,1,:]-p[:,0,:]).prod(dim=1) if p.dim() > 1 else torch.Tensor([0]) for p in pts]
    i = [sz.argmax().item() for sz in size]
    tx = [pts[j][i[j],0,0].item() if pts[j].dim() > 1 else 0 for j in range(len(i))]
    ty = [pts[j][i[j],0,1].item() if pts[j].dim() > 1 else 0 for j in range(len(i))]
    bx = [pts[j][i[j],1,0].item() if pts[j].dim() > 1 else 0 for j in range(len(i))]
    by = [pts[j][i[j],1,1].item() if pts[j].dim() > 1 else 0 for j in range(len(i))]
    # import pudb; pudb.set_trace()
    attention_img = [img[i,:,:,:] if (bx[i] - tx[i])*(by[i] - ty[i]) == 0 else img[i, :, ty[i]:by[i], tx[i]:bx[i]] for i in range(img.size(0))]
    attention_img = torch.cat([F.interpolate(img.unsqueeze(0), size=sz, mode='bilinear', align_corners=True) for img in attention_img])

    # attention_img = torch.empty(*img.shape[:2],*sz)
    # interpolate = F.interpolate
    # for i,d in enumerate(attention_xy):
    #     tx, ty, bx, by = d
    #     if tx == 0 and ty == 0 and bx == 0 and by == 0:
    #         this_attention_img = img[i, :, :, :]
    #     else:
    #         this_attention_img = img[i, :, ty:by, tx:bx]
    #
    #     this_attention_img = this_attention_img.unsqueeze(0)
    #     attention_img[i,:,:,:] = interpolate(this_attention_img, size=sz, mode='bilinear', align_corners=True)
    return attention_img
