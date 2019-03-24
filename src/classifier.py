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
        self.image_size = tuple(config[i][1:])

        i += 1
        self.use_masks = (config[i][0] == 1)
        if self.use_masks:
            self.crop_size = config[i][1]
            self.crop_filter = torch.ones(1,1,1,self.crop_size)
            self.crop_filter_t = self.crop_filter.transpose(-2,-1)

        x = torch.rand(2, self.in_channels, *self.image_size)
        self.layers = nn.ModuleList()
        i += 1
        while i < len(config):
            self.layers.append(create_module(config[i], x.size(1)))
            with torch.no_grad():
                x = self.layers[-1](x)
            i += 1

        self.out_features = x.size(1)
        print(self.out_features)

        self.is_cuda = False

    def cuda(self, device=None):
        self.is_cuda = True
        if self.use_masks:
            self.crop_filter = self.crop_filter.cuda()
            self.crop_filter_t = self.crop_filter_t.cuda()
        return super(Classifier, self).cuda(device)

    def forward(self, x, z=None, debug=False):
        if self.use_masks and z is not None:
            x = self.crop_images(x,z)

        if x.shape[2:] != self.image_size:
            x = F.interpolate(x, size=self.image_size, mode='bilinear')

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
        zf = F.conv1d(z,self.crop_filter)
        zf = F.conv1d(zf,self.crop_filter_t)
        _,zi = zf.view(zf.shape[0],-1).max(-1,keepdim=True)
        py,px = zi//zf.shape[-1],zi%zf.shape[-1]
        xf = torch.stack([x[i,:,py[i]:py[i]+self.crop_size,px[i]:px[i]+self.crop_size] for i in range(zf.shape[0])])
        return xf

    def run_epoch(self, mode, batches, epoch, criterion=None, optimizer=None, writer=None, log_interval=None):
        if mode == 'train':
            self.train()
        else:
            self.eval()

        loss = 0.0
        correct_predictions = 0
        true_positives = 0
        false_negatives = 0
        false_positives = 0
        data_size = 0
        i = 0
        for data in tqdm(batches, desc='Epoch {}: '.format(epoch), total=len(batches)):
            if self.is_cuda:
                for k in range(len(data)):
                    data[k] = data[k].cuda().float()
            if len(data) > 2:
                x,y,z = data[:3]
            else:
                x,y = data
                z = None
            # x = x.float()

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
            if self.out_features == 1:
                predictions = torch.ge(torch.sigmoid(logits), 0.5)
                true_positives += (predictions[y.byte()] == 1).sum().item()
                false_negatives += (predictions[y.byte()] == 0).sum().item()
                false_positives += (predictions[1 - y.byte()] == 0).sum().item()
            else:
                predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions.long() == y.long()).sum().item()
            data_size += x.shape[0]

            if mode == 'train' and (log_interval is not None) and (i % log_interval == 0):
                writer.add_scalar('c/{}_loss'.format(mode), batch_loss.item(), epoch*len(batches)+i)
            i += 1

        loss = loss/len(batches)
        accuracy = correct_predictions/data_size
        if self.out_features == 1:
            recall = true_positives/(true_positives+false_negatives)
            precision = true_positives/(true_positives+false_positives)
        if writer is not None:
            writer.add_scalar('c/{}_acc'.format(mode), accuracy, epoch)
            if mode == 'valid':
                writer.add_scalar('c/{}_loss'.format(mode), loss, epoch)
            if self.out_features == 1:
                writer.add_scalar('c/{}_recall'.format(mode), recall, epoch)
                writer.add_scalar('c/{}_precision'.format(mode), precision, epoch)

        if self.out_features == 1:
            return {'loss': loss, 'acc': accuracy, 'recall': recall, 'precision': precision}
        else:
            return {'loss': loss, 'acc': accuracy}

    def get_criterion(self):
        if self.out_features == 1:
            return nn.BCEWithLogitsLoss()
        else:
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
                    if self.out_features == 1:
                        predictions = torch.ge(torch.sigmoid(logits), 0.5)
                    else:
                        predictions = torch.argmax(logits, dim=1)
                    if labels:
                        y_true = y
                else:
                    if self.out_features == 1:
                        preds = torch.ge(torch.sigmoid(logits), 0.5)
                    else:
                        preds = torch.argmax(logits, dim=1)
                    predictions = torch.cat([predictions, preds])
                    if labels:
                        y_true = torch.cat([y_true, y])

            if labels:
                accuracy = (predictions.long() == y_true.long()).double().mean().item()
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
