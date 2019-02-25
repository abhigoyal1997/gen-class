import os
import argparse
import torch

from src.dataset import BloodCellsDataset as Dataset
from src.training import train
from src.testing import test
from src.model_utils import read_config, read_hparams, load_model, create_model

RANDOM_SEED = 1234


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
        model = create_model(config, device=args.device)
        print('Model initialized!', flush=True)

        print('Loading data...', flush=True)
        if config[0][0] == 'genclassifier':
            dataset = Dataset(args.data_file, image_size=model.image_size, masks=True, size=args.ds, num_masks=args.ms, random_seed=RANDOM_SEED)
        else:
            dataset = Dataset(args.data_file, image_size=model.image_size, masks=True, mask_only=True, size=args.ds)

        print('Training model...', flush=True)
        train(model, hparams, dataset, model_path, log_interval=10, device=args.device)
    elif args.command == 'test':
        if not os.path.exists(args.model_path):
            print("Model doesn't exist!")
            exit(0)

        print('Loding model...', flush=True)
        model = load_model(args.model_path, args.device)
        print('Model loaded!', flush=True)

        print('Loading data...', flush=True)
        dataset = Dataset(args.data_file, image_size=model.image_size, masks=False)

        print('Testing model...', flush=True)
        test(model, dataset, args.model_path, device=args.device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--device',dest='device',default='cuda:0')
    subparsers = parser.add_subparsers(dest='command')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('model_path')
    parser_train.add_argument('-mc','--model-config',dest='model_config',default='model_config.txt')
    parser_train.add_argument('-c','--comment',dest='comment',default=None)
    parser_train.add_argument('-d','--data-file',dest='data_file',default='data/train_data.csv')
    parser_train.add_argument('-ds','--train-size',dest='ds',default=None,type=int)
    parser_train.add_argument('-ms','--num-masks',dest='ms',default=None,type=int)
    parser_train.add_argument('-s','--train-specs',dest='train_specs',default='train_specs.txt')

    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('model_path')
    parser_test.add_argument('-d','--data-file',dest='data_file',default='data/test_data2.csv')

    args = parser.parse_args()
    if args.command[:2] == 'tr':
        args.command = 'train'
    elif args.command[:2] == 'te':
        args.command = 'test'
    else:
        args.command = 'predict'

    if '0' in args.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        args.device = torch.device('cuda')
        print('Using {}:0'.format(args.device))
    elif '1' in args.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        args.device = torch.device('cuda')
        print('Using {}:1'.format(args.device))
    else:
        print('Not using cuda!')
        args.device = torch.device('cpu')
        print('Using {}'.format(args.device))

    return args


if __name__ == '__main__':
    args = parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    main(args)
