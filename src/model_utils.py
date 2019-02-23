import os
import torch

from src.classifier import Classifier
from src.generator import Generator
from src.genclassifier import GenClassifier

MODELS = {
    'generator': Generator,
    'classifier': Classifier
}


def save_model(model, model_path):
    with open(os.path.join(model_path,'config.txt'),'w') as f:
        f.write('\n'.join([' '.join([str(j) for j in i]) for i in model.config]))

    torch.save(model.state_dict(), os.path.join(model_path, '{}.pth'.format(model.config[0][0])))
    print('{} saved to {}'.format(model.config[0][0], model_path))


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


def create_model(config, device=torch.device('cpu')):
    if config[0][0] == 'genclassifier':
        generator = load_model(config[2][0], device)
        classifier = load_model(config[3][0], device)
        return GenClassifier(config, generator, classifier).to(device)
    elif config[0][0] in MODELS:
        return MODELS[config[0][0]](config).to(device)
    else:
        print('{} not implemented!'.format(config[0][0]))


def load_model(model_path, device):
    config = read_config(os.path.join(model_path, 'config.txt'))
    model = create_model(config, device)
    if config[0][0] != 'genclassifier':
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
