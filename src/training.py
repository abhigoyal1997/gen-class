import os
import torch
import shutil

from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torchvision.datasets import ImageFolder
from tensorboardX import SummaryWriter
from time import time
from src import classifier


def save_model(model, model_path):
    with open(os.path.join(model_path,'config.txt'),'w') as f:
        f.write('\n'.join([' '.join([str(j) for j in i]) for i in model.config]))

    torch.save(model.state_dict(), os.path.join(model_path, '{}.pth'.format(model.config[0][0])))
    print('{} saved to {}'.format(model.config[0][0], model_path))


def train(model, hparams, dataset, model_path=None, log_interval=None, device=torch.device('cpu')):
    batch_size = hparams['batch_size']
    num_epochs = hparams['num_epochs']
    train_ratio = hparams['train_ratio']
    mask_ratio = hparams['mask_ratio']
    num_workers = hparams['num_workers']

    train_size = int(train_ratio*len(dataset))
    train_set, valid_set = random_split(dataset, [train_size, len(dataset) - train_size])

    # TODO: create batches of data with masks
    train_batches = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_batches = DataLoader(valid_set, batch_size=batch_size, num_workers=num_workers)

    module = eval(model.__class__.__name__.lower())

    criterion = module.get_criterion()
    optimizer = Adam(model.parameters())

    log_path = model_path.replace('models','logs')
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    writer = SummaryWriter(log_dir=log_path)

    best_acc = 0.0
    for epoch in range(num_epochs):
        # Train
        metrics = module.run_epoch('train', model, criterion, optimizer, train_batches, epoch, writer, log_interval, device)
        print('Train: {}'.format(metrics))

        # Validate
        metrics = module.run_epoch('valid', model, criterion, optimizer, valid_batches, epoch, writer, log_interval, device)
        print('Validation: {}'.format(metrics))

        if metrics['acc'] > best_acc:
            best_acc = metrics['acc']
            save_model(model, model_path)
