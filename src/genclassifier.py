import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


class GenClassifier(nn.Module):
    def __init__(self, config, generator, classifier):
        super(GenClassifier, self).__init__()

        self.config = config
        self.num_mask_samples = config[1][0]

        self.generator = generator
        self.classifier = classifier

        self.image_size = self.generator.image_size

    def get_criterion(self):
        return self.e_step

    def run_epoch(self, mode, batches, epoch, criterion=None, optimizer=None, writer=None, log_interval=None, device=torch.device('cpu')):
        if mode == 'train':
            self.train()
        else:
            self.eval()

        loss = 0.0
        predictions = None
        y_true = None
        i = 0
        for data in tqdm(batches, desc='Epoch {}: '.format(epoch), total=len(batches)):
            for k in range(len(data)):
                data[k] = data[k].to(device)
            x,y,z,m = data
            num_masks = m.sum()

            if mode == 'train':
                optimizer.zero_grad()

                # Forward Pass
                zl = self.generator(x)
                batch_loss = self.e_step(x, y, z[:num_masks], zl, device=device)

                # Backward Pass
                self.m_step(batch_loss, optimizer)
            else:
                with torch.no_grad():
                    # Forward Pass
                    pred, zs, batch_loss = self.predict_batch(x, y, device, True)

                # Update metrics
                if predictions is None:
                    predictions = pred
                    y_true = y
                else:
                    predictions = torch.cat([predictions, pred])
                    y_true = torch.cat([y_true, y])

            loss += batch_loss.item()
            if mode == 'train' and (log_interval is not None) and (i % log_interval == 0):
                writer.add_scalar('{}_loss'.format(mode), batch_loss.item(), epoch*len(batches)+i)
            i += 1

        loss = loss/len(batches)
        if mode != 'train':
            accuracy = (predictions == y_true.long()).double().mean().item()
            if writer is not None:
                writer.add_scalar('{}_acc'.format(mode), accuracy, epoch)
                writer.add_scalar('{}_loss'.format(mode), loss, epoch)
            return {'loss': loss, 'acc': accuracy}
        else:
            return {'loss': loss}

    def e_step(self, x, y, z, zl, device=torch.device('cuda')):
        batch_size = x.shape[0]
        num_masks = z.shape[0] if z is not None else 0

        if num_masks == 0:
            z = mh_sample(zl, x, y, classifier=self.classifier, device=device, num_samples=self.num_mask_samples)
            x = x.repeat(self.num_mask_samples, 1, 1, 1)
            y = y.repeat(self.num_mask_samples)
            zl = zl.repeat(self.num_mask_samples, 1, 1, 1)
        elif num_masks != batch_size:
            z = torch.cat([z, mh_sample(zl[num_masks:], x[num_masks:], y[num_masks:], classifier=self.classifier, device=device, num_samples=self.num_mask_samples)])
            x = torch.cat([x[:num_masks], x[num_masks:].repeat(self.num_mask_samples, 1, 1, 1)])
            y = torch.cat([y[:num_masks], y[num_masks:].repeat(self.num_mask_samples)])
            zl = torch.cat([zl[:num_masks], zl[num_masks:].repeat(self.num_mask_samples, 1, 1, 1)])

        yl = self.classifier(x,z)
        nll = segmentation_nll(zl, z) + classification_nll(yl, y)

        if num_masks == 0:
            nll = nll.sum()/self.num_mask_samples
        elif num_masks != batch_size:
            nll = nll[:num_masks].sum() + nll[num_masks:].sum()/self.num_mask_samples
        else:
            nll = nll.sum()

        return nll/batch_size

    def m_step(self, nll, optimizer):
        nll.backward()
        optimizer.step()

    def predict_batch(self, x, y, device=torch.device('cpu'), loss=False):
        z_logits = self.generator(x)
        if loss:
            batch_loss = self.e_step(x, y, None, z_logits, device=device)

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

        if loss:
            return torch.argmax(yp, dim=1), z, batch_loss
        else:
            return torch.argmax(yp, dim=1), z

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

                pred, z = self.predict_batch(x,None,device)
                # Update metrics
                if predictions is None:
                    predictions = [pred, z]
                    if labels:
                        y_true = y
                else:
                    predictions[0] = torch.cat([predictions[0], pred])
                    predictions[1] = torch.cat([predictions[1], z])
                    if labels:
                        y_true = torch.cat([y_true, y])

            if labels:
                accuracy = (predictions[0] == y_true.long()).double().mean().item()
                return {'predictions': predictions, 'acc': accuracy}
            else:
                return {'predictions': predictions}


def segmentation_nll(logits, masks, mean=False):
    batch_size = logits.shape[0]
    nll = F.binary_cross_entropy_with_logits(logits, masks, reduction='none').view(batch_size, -1).mean(dim=1)
    if mean:
        nll = nll.mean()
    return nll


def classification_nll(logits, labels, mean=False):
    nll = F.cross_entropy(logits, labels, reduction='none')
    if mean:
        nll = nll.mean()
    return nll


def p_z(p, x, y, z, classifier):
    pz = torch.prod((p*z + (1-p)*(1-z)).view(p.shape[0], -1), dim=1)
    p = torch.softmax(classifier(x, z), dim=1)
    py = p[range(p.shape[0]),y]

    return pz*py


def mh_sample(zl, x, y, classifier, burnin=4, device=torch.device('cpu'), num_samples=1):
    with torch.no_grad():
        zp = torch.sigmoid(zl)
        samples = None
        s1 = torch.ge(zp, 0.5).float()

        for _ in range(num_samples):
            for _ in range(burnin):
                s2 = torch.bernoulli(zp).float()
                pz1 = p_z(zp, x, y, s1, classifier)
                pz2 = p_z(zp, x, y, s2, classifier)
                r = pz2/pz1
                u = torch.rand(r.shape, device=device)
                ns = (u>r).nonzero()
                if ns.shape[0] == 0:
                    continue
                s2[ns[:,0]] = s1[ns[:,0]]
                s1 = s2
            if samples is None:
                samples = s2
            else:
                samples = torch.cat([samples, s2])
        return samples
