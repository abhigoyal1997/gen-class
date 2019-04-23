import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules import create_module
from tqdm import tqdm


class Generator(nn.Module):
    def __init__(self, config, cuda=True):
        super(Generator, self).__init__()

        self.config = config
        self.is_cuda = cuda

        i = 1
        self.in_channels = config[i][0]
        self.image_size = config[i][1]
        self.augment = config[i][2]

        x = torch.rand(2, self.in_channels, self.image_size, self.image_size)
        if self.is_cuda:
            x = x.cuda()

        self.layers = nn.ModuleList()
        i += 1
        while config[i][0] != 'dblock':
            self.layers.append(create_module(config[i], x.size(1), self.is_cuda))
            with torch.no_grad():
                x = self.layers[-1](x)
            i += 1

        self.down_blocks = nn.ModuleList()
        f_stack = []
        while config[i][0] != 'block':
            self.down_blocks.append(create_module(config[i], x.size(1), self.is_cuda))
            with torch.no_grad():
                x,f = self.down_blocks[-1](x)
            i += 1
            f_stack.append(f)

        self.conv_block = create_module(config[i], x.size(1), self.is_cuda)
        with torch.no_grad():
            x = self.conv_block(x)
        i += 1

        self.up_blocks = nn.ModuleList()
        while config[i][0] != 'conv':
            self.up_blocks.append(create_module(config[i], x.size(1), self.is_cuda))
            with torch.no_grad():
                x = self.up_blocks[-1](x,f_stack.pop())
            i += 1

        self.final_layers = nn.ModuleList()
        while i < len(config):
            self.final_layers.append(create_module(config[i], x.size(1), self.is_cuda))
            with torch.no_grad():
                x = self.final_layers[-1](x)
            i += 1

        self.out_features = x.size(1)

    def cuda(self, device=None):
        self.is_cuda = True
        return super(Generator, self).cuda(device)

    def forward(self, x):
        sz = x.shape[2:]
        if sz != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=True)
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
        dice = 0.0
        correct_predictions = 0
        data_size = 0
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

            if self.training:
                optimizer.zero_grad()
                if self.augment:
                    x = torch.cat([x,x.flip(-1)],dim=0)
                    z = torch.cat([z,z.flip(-1)],dim=0)

            with torch.set_grad_enabled(self.training):
                # Forward Pass
                logits = self.forward(x)
                if self.out_features == 1:
                    p = logits.sigmoid()
                    predictions = torch.ge(p,0.5).view(z.size(0),-1).float()
                else:
                    p = logits.softmax(dim=1)[:,1,:,:]
                    predictions = logits.argmax(dim=1).view(z.size(0),-1).float()
                batch_loss = criterion(logits, z)
                p = p.view(p.size(0), -1)
                z = z.view(z.size(0), -1).float()

            if self.training:
                # Backward Pass
                batch_loss.backward()
                optimizer.step()

            # Update metrics
            loss += batch_loss.item()

            dice += (2*(p*z).sum(dim=1).float()/(p.sum(dim=1) + z.sum(dim=1)).float()).sum().item()
            correct_predictions += (predictions == z).sum().float().item()
            data_size += x.shape[0]

            if self.training and (log_interval is not None) and (i % log_interval == 0):
                writer.add_scalar('g/{}_loss'.format(mode), batch_loss.item(), epoch*len(batches)+i)
            i += 1

        loss = loss/len(batches)
        accuracy = correct_predictions/(data_size*self.image_size*self.image_size)
        dice = dice/data_size
        if writer is not None:
            writer.add_scalar('g/{}_dice'.format(mode), dice, epoch)
            writer.add_scalar('g/{}_accuracy'.format(mode), accuracy, epoch)
            if not self.training:
                writer.add_scalar('g/{}_loss'.format(mode), loss, epoch)
        return {'loss': loss, 'dice': dice, 'accuracy': accuracy}

    def get_criterion(self, no_reduction=False, l2_penalty=0):
        cc_loss = nn.BCEWithLogitsLoss(reduction='none')

        def loss(logits, labels):
            batch_size = labels.shape[0]
            ret = cc_loss(logits, labels).view(batch_size,-1).sum(1)
            ret += l2_penalty*(logits.view(batch_size,-1)**2).sum(1)
            if no_reduction:
                return ret
            return ret.mean()

        return loss

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
