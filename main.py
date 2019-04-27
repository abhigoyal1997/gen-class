import os
import argparse
import torch
import numpy as np
import random

from src.dataset import FacesDataset as Dataset
from src.training import train
from src.testing import test
from src.model_utils import read_config, read_hparams, load_model, create_model

RANDOM_SEED = 0
IMAGE_SIZE = 300


def _init_fn(worker_id):
	np.random.seed(RANDOM_SEED)


def main(args):
	if args.command == 'train':
		if args.comment is not None:
			model_path = args.model_path + '_' + args.comment
		else:
			model_path = args.model_path
		if not os.path.exists(model_path):
			os.makedirs(model_path)

		config = read_config(args.model_config)
		hparams = read_hparams(args.train_specs)

		print('Creating new model...', flush=True)
		model = create_model(config, cuda=args.cuda)
		model.l2_penalty = args.l2_penalty
		print('Model initialized!', flush=True)

		print('Loading data...', flush=True)
		if config[0][0] == 'gc':
			dataset = Dataset(args.data_file, image_size=IMAGE_SIZE, masks=True, size=args.ds, num_masks=args.ms, random_seed=RANDOM_SEED, dataset=args.dataset)
		elif model.config[0][0] == 'c':
			dataset = Dataset(args.data_file, image_size=IMAGE_SIZE, masks=(model.loc is not None), mask_only=(model.loc is not None), size=args.ds, random_seed=RANDOM_SEED, dataset=args.dataset)
		else:
			dataset = Dataset(args.data_file, image_size=IMAGE_SIZE, masks=True, mask_only=True, size=args.ds, random_seed=RANDOM_SEED, dataset=args.dataset)

		print('Training model...', flush=True)
		train(model, hparams, dataset, _init_fn, model_path, log_interval=2)
	elif args.command == 'test':
		if not os.path.exists(args.model_path):
			print("Model doesn't exist!")
			exit(0)

		print('Loding model...', flush=True)
		model = load_model(args.model_path, cuda=args.cuda, weights=not args.test_init)
		print('Model loaded!', flush=True)

		print('Loading data...', flush=True)
		if model.config[0][0] == 'gc':
			dataset = Dataset(args.data_file, image_size=IMAGE_SIZE, masks=False, dataset=args.dataset)
		else:
			dataset = Dataset(args.data_file, image_size=IMAGE_SIZE, masks=False, mask_only=False, dataset=args.dataset)

		print('Testing model...', flush=True)
		test(model, dataset, _init_fn, args.model_path)


def parse_args():
	global RANDOM_SEED

	parser = argparse.ArgumentParser()
	parser.add_argument('-p','--device',dest='device',default='0')
	parser.add_argument('-r','--random-seed',dest='random_seed',default=RANDOM_SEED,type=int)
	subparsers = parser.add_subparsers(dest='command')

	parser_train = subparsers.add_parser('train')
	parser_train.add_argument('model_path')
	parser_train.add_argument('-mc','--model-config',dest='model_config',default='model_config.txt')
	parser_train.add_argument('-c','--comment',dest='comment',default=None)
	parser_train.add_argument('-dataset',dest='dataset',default='mnist')
	parser_train.add_argument('-d','--data-file',dest='data_file',default='data/mnist/train.csv')
	parser_train.add_argument('-ds','--train-size',dest='ds',default=None)
	parser_train.add_argument('-ms','--num-masks',dest='ms',default=None)
	parser_train.add_argument('-l2','--l2-penalty',dest='l2_penalty',default=0,type=float)
	parser_train.add_argument('-s','--train-specs',dest='train_specs',default='train_specs.txt')

	parser_test = subparsers.add_parser('test')
	parser_test.add_argument('model_path')
	parser_test.add_argument('-dataset',dest='dataset',default='mnist')
	parser_test.add_argument('-d','--data-file',dest='data_file',default='data/mnist/test.csv')
	parser_test.add_argument('-i','--test-init',dest='test_init',default=False,action='store_true')

	args = parser.parse_args()
	if args.command[:2] == 'tr':
		args.command = 'train'
	elif args.command[:2] == 'te':
		args.command = 'test'
	else:
		args.command = 'predict'

	try:
		args.ds = int(args.ds)
	except Exception:
		args.ds = None

	try:
		args.ms = int(args.ms)
	except Exception:
		args.ms = None

	try:
		cuda = int(args.device)
		os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
		args.cuda = True
		print(f'Using cuda:{cuda}')
	except Exception:
		args.cuda = False
		print('Not using cuda!')

	RANDOM_SEED = args.random_seed

	return args


if __name__ == '__main__':
	args = parse_args()
	torch.multiprocessing.set_sharing_strategy('file_system')
	if args.cuda:
		torch.cuda.manual_seed(RANDOM_SEED)
		torch.cuda.manual_seed_all(RANDOM_SEED)
	torch.manual_seed(RANDOM_SEED)
	np.random.seed(RANDOM_SEED)
	random.seed(RANDOM_SEED)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	main(args)
