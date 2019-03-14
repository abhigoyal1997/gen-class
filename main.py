import os
import argparse
import torch

from src.dataset import BloodCellsDataset as Dataset
from src.training import train
from src.testing import test
from src.model_utils import read_config, read_hparams, load_model, create_model

RANDOM_SEED = 12345


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
        print('Model initialized!', flush=True)

        print('Loading data...', flush=True)
        if config[0][0] == 'gc':
            dataset = Dataset(args.data_file, image_size=model.image_size, masks=True, size=args.ds, num_masks=args.ms, random_seed=RANDOM_SEED)
        elif model.config[0][0] == 'c':
            dataset = Dataset(args.data_file, image_size=model.image_size, masks=model.use_masks, mask_only=model.use_masks, size=args.ds)
        else:
            dataset = Dataset(args.data_file, image_size=model.image_size, masks=True, mask_only=True, size=args.ds)

        print('Training model...', flush=True)
        train(model, hparams, dataset, model_path, log_interval=2)
    elif args.command == 'test':
        if not os.path.exists(args.model_path):
            print("Model doesn't exist!")
            exit(0)

        print('Loding model...', flush=True)
        model = load_model(args.model_path, cuda=args.cuda, weights=not args.test_init)
        print('Model loaded!', flush=True)

        print('Loading data...', flush=True)
        if model.config[0][0] == 'gc':
            dataset = Dataset(args.data_file, image_size=model.image_size, masks=False)
        else:
            dataset = Dataset(args.data_file, image_size=model.image_size, masks=True, mask_only=True)

        print('Testing model...', flush=True)
        test(model, dataset, args.model_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--device',dest='device',default='0')
    subparsers = parser.add_subparsers(dest='command')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('model_path')
    parser_train.add_argument('-mc','--model-config',dest='model_config',default='model_config.txt')
    parser_train.add_argument('-c','--comment',dest='comment',default=None)
    parser_train.add_argument('-d','--data-file',dest='data_file',default='data/ddsm/train.csv')
    parser_train.add_argument('-ds','--train-size',dest='ds',default=None,type=int)
    parser_train.add_argument('-ms','--num-masks',dest='ms',default=None,type=int)
    parser_train.add_argument('-s','--train-specs',dest='train_specs',default='train_specs.txt')

    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('model_path')
    parser_test.add_argument('-d','--data-file',dest='data_file',default='data/ddsm/test.csv')
    parser_test.add_argument('-i','--test-init',dest='test_init',default=False,action='store_true')

    args = parser.parse_args()
    if args.command[:2] == 'tr':
        args.command = 'train'
    elif args.command[:2] == 'te':
        args.command = 'test'
    else:
        args.command = 'predict'

    if '0' in args.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        args.cuda = True
        print('Using cuda:0')
    elif '1' in args.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        args.cuda = True
        print('Using cuda:1')
    else:
        args.cuda = False
        print('Not using cuda!')

    return args


if __name__ == '__main__':
    args = parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    if args.cuda:
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    main(args)
