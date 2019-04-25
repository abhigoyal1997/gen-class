import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm


class GenClassifier(nn.Module):
    def __init__(self, config, generator=None, classifier=None, fix_generator=False, fix_classifier=False):
        super(GenClassifier, self).__init__()

        self.config = config
        self.num_mask_samples = config[1][0]
        self.augment = config[1][1]

        self.generator = generator
        self.classifier = classifier

        self.generator.augment = self.augment
        self.classifier.augment = self.augment

        self.image_size = self.generator.image_size

        self.is_cuda = False

        self.fix_generator = fix_generator
        self.fix_classifier = fix_classifier

        if self.fix_generator:
            for p in self.generator.parameters():
                p.requires_grad = False

        if self.fix_classifier:
            for p in self.classifier.parameters():
                p.requires_grad = False

    def cuda(self, device=None):
        self.is_cuda = True
        self.generator.cuda(device)
        self.classifier.cuda(device)
        return super(GenClassifier, self).cuda(device)

    def get_criterion(self):
        self.classifier_criterion = self.classifier.get_criterion(no_reduction=True)
        self.generator_criterion = self.generator.get_criterion(no_reduction=True, l2_penalty=0 if not hasattr(self,'l2_penalty') else self.l2_penalty)
        return self.e_step

    def run_epoch(self, mode, batches, epoch=None, criterion=None, optimizer=None, writer=None, log_interval=None):
        if mode == 'train':
            self.train()
        else:
            self.eval()

        loss = 0.0
        i = 0
        correct_predictions = 0
        data_size = 0
        if epoch is not None:
            b_iter = tqdm(batches, desc='Epoch {}: '.format(epoch), total=len(batches))
        else:
            b_iter = tqdm(batches, total=len(batches))
        for data in b_iter:
            if self.is_cuda:
                for k in range(min(3,len(data))):
                    data[k] = data[k].cuda()

            if mode == 'train':
                optimizer.zero_grad()

                x,y,z,m = data
                if self.augment:
                    to_flip = torch.rand(x.shape[0])>0.5
                    x[to_flip,:,:,:] = x[to_flip,:,:,:].flip(-1)
                    z[to_flip,:,:,:] = z[to_flip,:,:,:].flip(-1)
                    # x = torch.cat([x,x.flip(-1)])
                    # z = torch.cat([z,z.flip(-1)])
                    # y = torch.cat([y]*2)
                    # m = torch.cat([m]*2)

                num_masks = sum(m)

                # Forward Pass
                zl = self.generator(x)
                batch_loss = self.e_step(x, y, z[:num_masks], zl)

                # For logging training accuracy
                pred = self.predict_batch(x, y, zl=zl, loss=False, return_masks=False)

                # Backward Pass
                self.m_step(batch_loss, optimizer)
            else:
                x,y = data[:2]
                pred,batch_loss = self.predict_batch(x, y, zl=None, loss=True, return_masks=False)

            # Update metrics
            correct_predictions += (pred == y.long()).sum().item()
            data_size += x.shape[0]

            loss += batch_loss.item()
            if mode == 'train' and (log_interval is not None) and (i % log_interval == 0):
                writer.add_scalar('gc/{}_loss'.format(mode), batch_loss.item(), epoch*len(batches)+i)
            i += 1

        accuracy = correct_predictions/data_size
        if writer is not None:
            writer.add_scalar('gc/{}_acc'.format(mode), accuracy, epoch)
            if mode != 'train':
                writer.add_scalar('gc/{}_loss'.format(mode), loss, epoch)
        return {'loss': loss, 'acc': accuracy}

    def e_step(self, x, y, z, zl):
        batch_size = x.shape[0]
        num_masks = z.shape[0] if z is not None else 0

        if num_masks == 0:
            z = mh_sample(zl, x, y, classifier=self.classifier, cuda=self.is_cuda, num_samples=self.num_mask_samples)
            x = x.repeat(self.num_mask_samples, 1, 1, 1)
            y = y.repeat(self.num_mask_samples)
            zl = zl.repeat(self.num_mask_samples, 1, 1, 1)
        elif num_masks != batch_size:
            z = torch.cat([z, mh_sample(zl[num_masks:], x[num_masks:], y[num_masks:], classifier=self.classifier, cuda=self.is_cuda, num_samples=self.num_mask_samples)])
            x = torch.cat([x[:num_masks], x[num_masks:].repeat(self.num_mask_samples, 1, 1, 1)])
            y = torch.cat([y[:num_masks], y[num_masks:].repeat(self.num_mask_samples)])
            zl = torch.cat([zl[:num_masks], zl[num_masks:].repeat(self.num_mask_samples, 1, 1, 1)])

        yl = self.classifier(x,z)
        nll = self.segmentation_nll(zl, z) + self.classification_nll(yl, y)

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

    def predict_batch(self, x, y, zl=None, loss=False, return_masks=True):
        with torch.no_grad():
            if zl is None:
                zl = self.generator(x)
            if loss:
                batch_loss = self.e_step(x, y, None, zl)

            zp = torch.sigmoid(zl)
            if self.num_mask_samples == 1:
                z = torch.ge(zp, 0.5).float()
                yl = self.classifier(x,z)
                yp = torch.softmax(yl, dim=1)
            else:
                zp = zp.repeat(self.num_mask_samples, 1, 1, 1)
                z = torch.bernoulli(zp)
                x = x.repeat(self.num_mask_samples, 1, 1, 1)
                yl = self.classifier(x,z)
                yp = torch.softmax(yl, dim=1).reshape(self.num_mask_samples, int(yl.shape[0]/self.num_mask_samples), *yl.shape[1:]).mean(dim=0)

            if return_masks:
                if loss:
                    return torch.argmax(yp, dim=1), z, batch_loss
                else:
                    return torch.argmax(yp, dim=1), z
            else:
                if loss:
                    return torch.argmax(yp, dim=1), batch_loss
                else:
                    return torch.argmax(yp, dim=1)

    def predict(self, batches, labels=True, return_predictions=False):
        self.eval()
        if return_predictions:
            predictions = None
        elif not labels:
            return

        if labels:
            correct_predictions = 0
            data_size = 0

        with torch.no_grad():
            for data in tqdm(batches):
                if self.is_cuda:
                    for k in range(len(data)):
                        data[k] = data[k].cuda()
                x = data[0]
                if labels:
                    y = data[-1]

                pred, z = self.predict_batch(x,None)
                # Update metrics
                if labels:
                    correct_predictions += (pred == y.long()).sum().item()
                    data_size += x.shape[0]

                if return_predictions:
                    if predictions is None:
                        predictions = [pred.cpu().numpy(), z.cpu().numpy()]
                    else:
                        predictions[0] = np.concatenate([predictions[0], pred], axis=0)
                        predictions[1] = np.concatenate([predictions[1], z], axis=0)

            if labels:
                accuracy = correct_predictions/data_size
                if return_predictions:
                    return {'predictions': predictions, 'acc': accuracy}
                else:
                    return {'acc': accuracy}
            else:
                return {'predictions': predictions}

    def segmentation_nll(self, logits, masks, mean=False):
        nll = self.generator_criterion(logits, masks)
        if mean:
            nll = nll.mean()
        return nll

    def classification_nll(self, logits, labels, mean=False):
        nll = self.classifier_criterion(logits, labels)
        if mean:
            nll = nll.mean()
        return nll


def lp_z(zl, x, y, z, classifier):
    lpy = -F.cross_entropy(classifier(x,z), y, reduction='none')
    return lpy


def mh_sample(zl, x, y, classifier, burnin=4, cuda=True, num_samples=1):
    classifier_state = classifier.training
    classifier.eval()
    with torch.no_grad():
        zp = torch.sigmoid(zl)
        s1 = torch.ge(zp, 0.5).float()
        batch_size = x.shape[0]

        zpr = zp.repeat(burnin*num_samples,1,1,1)
        zlr = zl.repeat(1+burnin*num_samples,1,1,1)
        xr = x.repeat(1+burnin*num_samples,1,1,1)
        yr = y.repeat(1+burnin*num_samples)

        candidates = torch.bernoulli(zpr).float()
        pz = lp_z(zlr, xr, yr, torch.cat([s1,candidates],dim=0), classifier)

        batch_size = x.shape[0]

        for i in range(1, num_samples*burnin+1):
            r = pz[i*batch_size:(i+1)*batch_size] - pz[(i-1)*batch_size:i*batch_size]
            if cuda:
                u = torch.rand(r.shape, device=torch.device('cuda')).log()
            else:
                u = torch.rand(r.shape).log()
            ns = (u>r).nonzero()
            if ns.shape[0] != 0:
                candidates[(i-1)*batch_size+ns[:,0]] = s1[ns[:,0]]
                pz[i*batch_size+ns[:,0]] = pz[(i-1)*batch_size+ns[:,0]]
            s1 = candidates[(i-1)*batch_size:(i+1)*batch_size]

        idx = sum((list(range((i*burnin-1)*batch_size,i*burnin*batch_size)) for i in range(1,num_samples+1)),[])
        samples = candidates[idx]
        classifier.train(classifier_state)
        return samples


def mh_sample_old(zl, x, y, classifier, burnin=4, cuda=True, num_samples=1):
    classifier_state = classifier.training
    classifier.eval()
    with torch.no_grad():
        zp = torch.sigmoid(zl)
        samples = None
        s1 = torch.ge(zp, 0.5).float()
        pz1 = lp_z(zl, x, y, s1, classifier)

        for _ in range(num_samples):
            for _ in range(burnin):
                s2 = torch.bernoulli(zp).float()
                pz2 = lp_z(zl, x, y, s2, classifier)
                r = pz2 - pz1
                if cuda:
                    u = torch.rand(r.shape, device=torch.device('cuda')).log()
                else:
                    u = torch.rand(r.shape).log()
                ns = (u>r).nonzero()
                if ns.shape[0] != 0:
                    s2[ns[:,0]] = s1[ns[:,0]]
                    pz2[ns[:,0]] = pz1[ns[:,0]]
                s1 = s2
                pz1 = pz2
            if samples is None:
                samples = s1
            else:
                samples = torch.cat([samples, s1])
        classifier.train(classifier_state)
        return samples
