import os
import argparse
import torch

from src.dataset import BloodCellsDataset as Dataset
from src.classifier import Classifier
from src.training import train

RANDOM_SEED = 1234


def read_config(config_file):
    def cast(x):
        if '.' in x:
            return float(x)
        else:
            try:
                return int(x)
            except Exception:
                return x

    with open(config_file, 'r') as f:
        config = f.readlines()
    for i in range(len(config)):
        config[i] = list(map(cast, config[i].split()))
    return config


def create_model(config):
    if config[0][0] == 'generator':
        pass  # TODO: create and return a generator
    elif config[0][0] == 'classifier':
        model = Classifier(config)
        print('Classifier created!')
    return model


def get_model(model_path):
    config = read_config(os.path.join(model_path, 'config.txt'))
    model = create_model(config)
    state_path = os.path.join(model_path, config[0][0]+'.pth')
    model.load_state_dict(torch.load(state_path))
    return model


def read_hparams(spec_file):
    with open(spec_file,'r') as f:
        spec = f.readlines()
    param_keys = [
        'batch_size',
        'num_epochs',
        'train_ratio',
        'mask_ratio',
        'num_workers'
    ]
    hparams = {}
    for i in range(len(param_keys)):
        if '.' in spec[i]:
            hparams[param_keys[i]] = float(spec[i])
        else:
            hparams[param_keys[i]] = int(spec[i])
    return hparams


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

        if config[0] == 'generator-classifier':
            pass  # TODO: create a generator-classifier
        else:
            model = create_model(config).to(args.device)
        print('Model Initialized!')

        print('Loading data...')
        dataset = Dataset(args.data_dir, image_size=model.image_size, masks=True)

        print('Training model...')
        train(model, hparams, dataset, model_path, log_interval=10, device=args.device)
    else:
        if not os.path.exists(model_path):
            print("Model doesn't exist!")
            exit(0)

        model = get_model(config)
        print('Model Loaded!')

        print('Loading data...')
        dataset = Dataset(args.data_dir, image_size=model.image_size)

        print('Testing model...')
        # TODO: test the model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--device',dest='device',default='cuda:1')
    subparsers = parser.add_subparsers(dest='command')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('model_path')
    parser_train.add_argument('-mc','--model-config',dest='model_config',default='model_config.txt')
    parser_train.add_argument('-c','--comment',dest='comment',default=None)
    parser_train.add_argument('-d','--data-dir',dest='data_dir',required=True)
    parser_train.add_argument('-ds','--data-size',dest='data_size',default=None,type=int)
    parser_train.add_argument('-s','--train-specs',dest='train_specs',default='train_specs.txt')

    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('model_path')
    parser_test.add_argument('-d','--data-dir',dest='data_dir',required=True)

    args = parser.parse_args()
    if args.command[0] == 't':
        args.command = 'train'
    else:
        args.command = 'test'

    if '0' in args.device:
        args.device = torch.device('cuda:0')
    elif '1' in args.device:
        args.device = torch.device('cuda:1')
    else:
        print('Not using cuda!')
        args.device = torch.device('cpu')

    return args


if __name__ == '__main__':
    args = parse_args()
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    main(args)
