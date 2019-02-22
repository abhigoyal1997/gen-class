import os
import argparse
import torch

RANDOM_SEED = 1234


def create_model(config):
    if config[0] == 'generator':
        pass  # TODO: create and return a generator
    elif config[0] == 'classifier':
        pass  # TODO: create and return a classifier


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
        model_path = args.model_path + '_' + args.comment
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        with open(args.model_config,'r') as f:
            config = f.readlines()

        with open(os.path.join(model_path,'config.txt'),'w') as f:
            f.writelines([line+'\n' for line in config])

        hparams = read_hparams(args.train_specs)

        if config[0] == 'generator-classifier':
            pass  # TODO: create a generator-classifier
        else:
            model = create_model(config)
        print('Model Initialized!')

        print('Loading data...')
        # TODO: load the data

        print('Training model...')
        # TODO: train the model
    else:
        pass  # TODO: read data and create and test model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--device',dest='device',default='cuda:0')
    subparsers = parser.add_subparsers(dest='command')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('model_path',required=True)
    parser_train.add_argument('-mc','--model-config',dest='model_config',default='model_config.txt')
    parser_train.add_argument('-c','--comment',dest='comment',default='')
    parser_train.add_argument('-d','--data-dir',dest='data_dir',required=True)
    parser_train.add_argument('-ds','--data-size',dest='data_size',default=None,type=int)
    parser_train.add_argument('-s','--train-specs',dest='train_specs',default='train_specs.txt')

    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('model_path',required=True)
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
        args.device = torch.device('cpu')

    return args


if __name__ == '__main__':
    args = parse_args()
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    main(args)
