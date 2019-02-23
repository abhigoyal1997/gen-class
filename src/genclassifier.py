import torch
import torch.nn as nn

from tqdm import tqdm


class GenClassifier(nn.Module):
    def __init__(self, config, generator, classifier):
        super(GenClassifier, self).__init__()

        self.config = config
        self.num_mask_samples = config[1][0]

        self.generator = generator
        self.classifier = classifier

        self.image_size = self.generator.image_size

    # def run_epoch(self, mode, batches, epoch, criterion=None, optimizer=None, writer=None, log_interval=None, device=torch.device('cpu')):
    #     if mode == 'train':
    #         self.train()
    #     else:
    #         self.eval()
    #
    #     loss = 0.0
    #     predictions = None
    #     y_true = None
    #     i = 0
    #     for data in tqdm(batches, desc='Epoch {}: '.format(epoch), total=len(batches)):
    #         for k in range(len(data)):
    #             data[k] = data[k].to(device)
    #         if len(data) == 3:
    #             x,z,y = data
    #         else:
    #             x,y = data
    #             z = None
    #
    #         if mode == 'train':
    #             optimizer.zero_grad()
    #
    #         # Forward Pass
    #         logits = self.forward(x,z)
    #         batch_loss = criterion(logits, y)
    #
    #         if mode == 'train':
    #             # Backward Pass
    #             batch_loss.backward()
    #             optimizer.step()
    #
    #         # Update metrics
    #         loss += batch_loss.item()
    #         if predictions is None:
    #             predictions = torch.argmax(logits,dim=1)
    #             y_true = y
    #         else:
    #             predictions = torch.cat([predictions, torch.argmax(logits,dim=1)])
    #             y_true = torch.cat([y_true, y])
    #
    #         if mode == 'train' and (log_interval is not None) and (i % log_interval == 0):
    #             writer.add_scalar('{}_loss'.format(mode), batch_loss.item(), epoch*len(batches)+i)
    #         i += 1
    #
    #     loss = loss/len(batches)
    #     accuracy = (predictions == y_true.long()).double().mean().item()
    #     if writer is not None:
    #         writer.add_scalar('{}_acc'.format(mode), accuracy, epoch)
    #         if mode == 'valid':
    #             writer.add_scalar('{}_loss'.format(mode), loss, epoch)
    #     return {'loss': loss, 'acc': accuracy}

    def get_criterion(self):
        return nn.CrossEntropyLoss()

    def predict(self, batches, labels=True, device=torch.device('cpu')):
        predictions = None
        if labels:
            y_true = None
        with torch.no_grad():
            for data in tqdm(batches):
                for k in range(len(data)):
                    data[k] = data[k].to(device)
                x = data[0]
                if labels:
                    y = data[-1]

                # Forward Pass
                z_logits = self.generator(x)
                zp = torch.sigmoid(z_logits)
                if self.num_mask_samples == 1:
                    z = torch.ge(zp, 0.5).float()
                    yl = self.classifier(x,z)
                    yp = torch.softmax(yl, dim=1)
                else:
                    zp = zp.repeat(self.num_mask_samples, 1, 1, 1)
                    z = torch.bernoulli(zp)
                    x = x.repeat(self.num_mask_samples, 1, 1, 1)
                    yl = self.classifier(x, z)
                    yp = torch.softmax(yl, dim=1).reshape(self.num_mask_samples, int(yl.shape[0]/self.num_mask_samples), *yl.shape[1:]).mean(dim=0)

                # Update metrics
                if predictions is None:
                    predictions = [torch.argmax(yp,dim=1), z]
                    if labels:
                        y_true = y
                else:
                    predictions[0] = torch.cat([predictions[0], torch.argmax(yp,dim=1)])
                    predictions[1] = torch.cat([predictions[1], z])
                    if labels:
                        y_true = torch.cat([y_true, y])

            if labels:
                accuracy = (predictions[0] == y_true.long()).double().mean().item()
                return {'predictions': predictions, 'acc': accuracy}
            else:
                return {'predictions': predictions}
