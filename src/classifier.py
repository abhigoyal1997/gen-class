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

    def forward(self, x, z=None, debug=False):
        if self.use_masks and z is not None:
            x = self.get_attn_img(x,z)

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

    def get_attn_img(self, x, z):
        z = (F.interpolate(z,size=x.shape[2:],mode='bilinear',align_corners=True)>0).float()
        attention_xy = get_attention_xy(z)
        return get_attention_img(x, attention_xy, self.crop_size)


def run_epoch(mode, model, criterion, optimizer, batches, epoch, writer=None, log_interval=None, device=torch.device('cpu')):
    if mode == 'train':
        model.train()
    else:
        model.eval()

    loss = 0.0
    predictions = None
    y_true = None
    i = 0
    for data in tqdm(batches, desc='Epoch {}: '.format(epoch), total=len(batches)):
        for k in range(len(data)):
            data[k] = data[k].to(device)
        if len(data) == 3:
            x,z,y = data
        else:
            x,y = data
            z = None

        if mode == 'train':
            optimizer.zero_grad()

        # Forward Pass
        logits = model(x,z)
        batch_loss = criterion(logits, y)

        if mode == 'train':
            # Backward Pass
            batch_loss.backward()
            optimizer.step()

        # Update metrics
        loss += batch_loss.item()
        if predictions is None:
            predictions = torch.argmax(logits,dim=1)
            y_true = y
        else:
            predictions = torch.cat([predictions, torch.argmax(logits,dim=1)])
            y_true = torch.cat([y_true, y])

        if mode == 'train' and (log_interval is not None) and (i % log_interval == 0):
            writer.add_scalar('{}_loss'.format(mode), batch_loss.item(), epoch*len(batches)+i)
        i += 1

    loss = loss/len(batches)
    accuracy = (predictions == y_true.long()).double().mean().item()
    if writer is not None:
        writer.add_scalar('{}_acc'.format(mode), accuracy, epoch)
        if mode == 'valid':
            writer.add_scalar('{}_loss'.format(mode), loss, epoch)
    return {'loss': loss, 'acc': accuracy}


def get_criterion():
    return nn.CrossEntropyLoss()


def get_attention_xy(mask):
    mask = mask.to('cpu').numpy()
    mask_shape = mask.shape
    mask = np.uint8(mask)
    attention_xy = np.zeros((mask_shape[0], 4))
    for i in range(mask_shape[0]):
        this_mask = mask[i, :, :, :]
        this_mask = this_mask.reshape(mask_shape[2], mask_shape[3])
        contours = cv.findContours(this_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        attention_xy[i, :] = find_attention_xy(contours)
    return attention_xy


def find_attention_xy(contours):
    size = 0
    tx, ty, bx, by = 0, 0, 0, 0
    for cnt in contours[1]:
        cnt = np.squeeze(cnt, axis=1)
        x1, y1 = np.min(cnt, axis=0)
        x2, y2 = np.max(cnt, axis=0)
        if (y2 - y1) * (x2 - x1) > size:
            tx, ty = x1, y1
            bx, by = x2, y2
            size = (y2 - y1) * (x2 - x1)
    return tx, ty, bx, by


def get_attention_img(img, attention_xy, sz=(28,28)):
    samples = attention_xy.shape[0]
    for i in range(samples):
        tx, ty, bx, by = attention_xy[i, :]
        if tx == 0 and ty == 0 and bx == 0 and by == 0:
            this_attention_img = img[i, :, :, :]
        else:
            this_attention_img = img[i, :, int(ty):int(by), int(tx):int(bx)]

        this_attention_img = this_attention_img.unsqueeze(0)
        this_attention_img = F.interpolate(this_attention_img, size=sz, mode='bilinear', align_corners=True)

        if i == 0:
            attention_img = this_attention_img
        else:
            attention_img = torch.cat((attention_img, this_attention_img), dim=0)
    return attention_img
