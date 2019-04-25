from torch.utils.data import Subset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tensorboardX import SummaryWriter
from time import time
from src.model_utils import save_model
from src.dataset import SBatchSampler


def train(model, hparams, dataset, worker_init_fn, model_path=None, log_interval=None):
	batch_size = hparams['batch_size']
	num_epochs = hparams['num_epochs']
	train_ratio = hparams['train_ratio']
	num_workers = hparams['num_workers']
	if model.config[0][0] == 'c':
		(lr,min_lr,decay,step) = hparams['lr_c']
	elif model.config[0][0] == 'g':
		(lr,min_lr,decay,step) = hparams['lr_g']
	else:
		(lr,min_lr,decay,step) = hparams['lr_c']
		batch_size = hparams['batch_size_gc']

	train_size = int(train_ratio*len(dataset))
	train_set = Subset(dataset, list(range(train_size)))
	valid_set = Subset(dataset, list(range(train_size, len(dataset))))

	if model.config[0][0] != 'gc':
		train_batches = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True, worker_init_fn=worker_init_fn)
	else:
		train_sampler = SBatchSampler(dataset.masks_idx[:train_size], batch_size, shuffle=True)
		train_batches = DataLoader(train_set, batch_sampler=train_sampler, num_workers=num_workers, worker_init_fn=worker_init_fn)

	valid_batches = DataLoader(valid_set, batch_size=batch_size, num_workers=num_workers)

	criterion = model.get_criterion()
	optimizer = Adam(model.parameters(), lr=lr)
	scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: max((decay**(epoch//step)), min_lr))

	log_path = model_path.replace('models','logs')+'_'+str(time())
	writer = SummaryWriter(log_dir=log_path)

	best_val = None
	best_loss = None
	for epoch in range(num_epochs):
		scheduler.step(epoch)
		writer.add_scalar(f'{model.config[0][0]}/lr',optimizer.param_groups[0]['lr'],epoch)

		# Train
		metrics = model.run_epoch('train', train_batches, criterion=criterion, optimizer=optimizer, epoch=epoch, writer=writer, log_interval=log_interval)
		print('Train: {}'.format(metrics))

		# Validate
		metrics = model.run_epoch('valid', valid_batches, criterion=criterion, epoch=epoch, writer=writer, log_interval=log_interval)
		print('Validation: {}'.format(metrics))

		if 'acc' in metrics:
			if best_val is None or metrics['acc'] > best_val:
				best_val = metrics['acc']
				best_loss = metrics['loss']
				save_model(model, model_path)
			elif metrics['acc'] == best_val and metrics['loss'] < best_loss:
				best_loss = metrics['loss']
				save_model(model, model_path)
		elif 'score' in metrics:
			if best_val is None or metrics['score'] > best_val:
				best_val = metrics['score']
				best_loss = metrics['loss']
				save_model(model, model_path)
			elif metrics['score'] == best_val and metrics['loss'] < best_loss:
				best_loss = metrics['loss']
				save_model(model, model_path)
		elif 'dice' in metrics:
			if best_val is None or metrics['dice'] > best_val:
				best_val = metrics['dice']
				best_loss = metrics['loss']
				save_model(model, model_path)
			elif metrics['dice'] == best_val and metrics['loss'] < best_loss:
				best_loss = metrics['loss']
				save_model(model, model_path)
		elif 'mse' in metrics:
			if best_val is None or metrics['mse'] < best_val:
				best_val = metrics['mse']
				best_loss = metrics['loss']
				save_model(model, model_path)
			elif metrics['mse'] == best_val and metrics['loss'] < best_loss:
				best_loss = metrics['loss']
				save_model(model, model_path)
