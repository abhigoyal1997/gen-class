import os
import argparse
import torch

from src.dataset import BloodCellsDataset as Dataset
from src.classifier import Classifier
from src.generator import Generator
from src.training import train
from src.testing import test

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


def create_model(config, device):
    if config[0][0] == 'generator':
        model = Generator(config)
    elif config[0][0] == 'classifier':
        model = Classifier(config)
    return model.to(device)


def get_model(model_path, device):
    config = read_config(os.path.join(model_path, 'config.txt'))
    model = create_model(config, device)
    state_path = os.path.join(model_path, config[0][0]+'.pth')
    model.load_state_dict(torch.load(state_path, map_location=device))
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

        print('Creating new model...', flush=True)
        if config[0] == 'generator-classifier':
            pass  # TODO: create a generator-classifier
        else:
            model = create_model(config, args.device)
        print('Model initialized!', flush=True)

        print('Loading data...', flush=True)
        dataset = Dataset(args.data_dir, image_size=model.image_size, masks=True)

        print('Training model...', flush=True)
        train(model, hparams, dataset, model_path, log_interval=10, device=args.device)
    elif args.command == 'test':
        if not os.path.exists(args.model_path):
            print("Model doesn't exist!")
            exit(0)

        print('Loding model...', flush=True)
        model = get_model(args.model_path, args.device)
        print('Model loaded!', flush=True)

        print('Loading data...', flush=True)
        dataset = Dataset(args.data_dir, image_size=model.image_size, masks=False)

        print('Testing model...', flush=True)
        test(model, dataset, args.model_path, device=args.device)
    else:
        if not os.path.exists(args.model_path):
            print("Model doesn't exist!")
            exit(0)

        print('Loding model...', flush=True)
        model = get_model(args.model_path, args.device)
        print('Model loaded!', flush=True)

        print('Loading data...', flush=True)
        dataset = Dataset(args.data_dir, image_size=model.image_size, masks=False)

        print('Predicting...', flush=True)
        predictions = test(model, dataset, args.model_path, device=args.device, predictions=True)
        if args.output_file is not None:
            # TODO: save predictions
            pass
        else:
            print(predictions)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--device',dest='device',default='cuda:1')
    subparsers = parser.add_subparsers(dest='command')

    default_data_dir = '/data1/abhinav/blood-cells/dataset2-master/subset'
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('model_path')
    parser_train.add_argument('-mc','--model-config',dest='model_config',default='model_config.txt')
    parser_train.add_argument('-c','--comment',dest='comment',default=None)
    parser_train.add_argument('-d','--data-dir',dest='data_dir',default=default_data_dir)
    parser_train.add_argument('-ds','--data-size',dest='data_size',default=None,type=int)
    parser_train.add_argument('-s','--train-specs',dest='train_specs',default='train_specs.txt')

    default_data_dir = '/data1/abhinav/blood-cells/dataset2-master/images/TEST'
    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('model_path')
    parser_test.add_argument('-d','--data-dir',dest='data_dir',default=default_data_dir)

    parser_predict = subparsers.add_parser('predict')
    parser_predict.add_argument('model_path')
    parser_predict.add_argument('-d','--data-dir',dest='data_dir',default=default_data_dir)
    parser_predict.add_argument('-o','--output-file',dest='output_file',default=None)

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
    elif '1' in args.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        args.device = torch.device('cuda')
    else:
        print('Not using cuda!')
        args.device = torch.device('cpu')

    return args


if __name__ == '__main__':
    args = parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    main(args)
