import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules import create_module
from tqdm import tqdm


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()

        self.config = config

        i = 1
        self.in_channels = config[i][0]
        self.image_size = config[i][1:]

        x = torch.rand(1, self.in_channels, *self.image_size)
        self.down_blocks = nn.ModuleList()
        i += 1
        f_stack = []
        while config[i][0] != 'cblock':
            self.down_blocks.append(create_module(config[i], x.size(1)))
            with torch.no_grad():
                x,f = self.down_blocks[-1](x)
            i += 1
            f_stack.append(f)

        self.conv_block = create_module(config[i], x.size(1))
        with torch.no_grad():
            x = self.conv_block(x)
        i += 1

        self.up_blocks = nn.ModuleList()
        while config[i][0] != 'conv':
            self.up_blocks.append(create_module(config[i], x.size(1)))
            with torch.no_grad():
                x = self.up_blocks[-1](x,f_stack.pop())
            i += 1

        self.final_layers = nn.ModuleList()
        while i < len(config):
            self.final_layers.append(create_module(config[i], x.size(1)))
            with torch.no_grad():
                x = self.final_layers[-1](x)
            i += 1

        self.is_cuda = False

    def cuda(self, device=None):
        self.is_cuda = True
        return super(Generator, self).cuda(device)

    def forward(self, x):
        sz = x.shape[2:]
        if sz != self.image_size:
            x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=True)
        f_stack = []
        for block in self.down_blocks:
            x,f = block(x)
            f_stack.append(f)

        x = self.conv_block(x)

        for block in self.up_blocks:
            x = block(x, f_stack.pop())

        for layer in self.final_layers:
            x = layer(x)

        if sz != x.shape[2:]:
            x = F.interpolate(x, size=sz, mode='bilinear', align_corners=True)
        return x

    def run_epoch(self, mode, batches, epoch, criterion=None, optimizer=None, writer=None, log_interval=None):
        if mode == 'train':
            self.train()
        else:
            self.eval()

        loss = 0.0
        predictions = None
        z_true = None
        i = 0
        for data in tqdm(batches, desc='Epoch {}: '.format(epoch), total=len(batches)):
            if len(data) > 2:
                x = data[0]
                z = data[2]
            else:
                print('Data instance must have masks for training!')
                exit(0)

            if self.is_cuda:
                x = x.cuda()
                z = z.cuda()

            if mode == 'train':
                optimizer.zero_grad()

                # Forward Pass
                logits = self.forward(x)
                batch_loss = criterion(logits, z)

                # Backward Pass
                batch_loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    # Forward Pass
                    logits = self.forward(x)
                    batch_loss = criterion(logits, z)

            # Update metrics
            loss += batch_loss.item()
            if predictions is None:
                predictions = torch.sigmoid(logits)
                z_true = z
            else:
                predictions = torch.cat([predictions, torch.sigmoid(logits)])
                z_true = torch.cat([z_true, z])

            if mode == 'train' and (log_interval is not None) and (i % log_interval == 0):
                writer.add_scalar('g/{}_loss'.format(mode), batch_loss.item(), epoch*len(batches)+i)
            i += 1

        loss = loss/len(batches)
        dice = soft_dice(predictions, z_true.float())
        if writer is not None:
            writer.add_scalar('g/{}_dice'.format(mode), dice, epoch)
            if mode == 'valid':
                writer.add_scalar('g/{}_loss'.format(mode), loss, epoch)
        return {'loss': loss, 'dice': dice}

    def get_criterion(self):
        return nn.BCEWithLogitsLoss()

    def predict(self, batches, labels=True):
        self.eval()
        with torch.no_grad():
            predictions = None
            if labels:
                z_true = None
            for data in tqdm(batches):
                if labels:
                    if len(data) == 3:
                        x,z,_ = data
                    else:
                        x,z = data

                    if self.is_cuda:
                        x = x.cuda()
                        z = z.cuda()
                else:
                    if len(data) == 3:
                        x,_,_ = data
                    elif len(data) == 2:
                        x,_ = data
                    else:
                        x = data[0]

                    if self.is_cuda:
                        x = x.cuda()

                # Forward Pass
                logits = self.forward(x)

                # Update metrics
                if predictions is None:
                    predictions = torch.sigmoid(logits)
                    if labels:
                        z_true = z
                else:
                    predictions = torch.cat([predictions, torch.sigmoid(logits)])
                    if labels:
                        z_true = torch.cat([z_true, z])

            if labels:
                return {'predictions': predictions, 'dice': soft_dice(predictions, z_true.float())}
            else:
                return {'predictions': predictions}


def soft_dice(predictions, z_true):
    with torch.no_grad():
        predictions = predictions.view(predictions.size(0),-1)
        z_true = z_true.view(z_true.size(0),-1)
        return (2*(predictions*z_true).sum(dim=1)/(predictions**2 + z_true**2).sum(dim=1)).mean().item()
