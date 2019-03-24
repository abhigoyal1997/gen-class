import os
import torch
import torch.nn as nn

from src.classifier import Classifier
from src.generator import Generator
from src.genclassifier import GenClassifier

MODELS = {
    'g': Generator,
    'c': Classifier
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


def init_weights(m):
    try:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    except AttributeError:
        pass


def create_model(config, cuda=True):
    if config[0][0] == 'gc':
        generator = load_model(config[2][0], False)
        classifier = load_model(config[3][0], False)
        model = GenClassifier(config, generator, classifier, fix_generator=(config[2][-1] == 'fix'), fix_classifier=(config[3][-1] == 'fix'))
    elif config[0][0] in MODELS:
        model = MODELS[config[0][0]](config)
    else:
        print('{} not implemented!'.format(config[0][0]))
        exit(0)

    if cuda:
        model.cuda()

    # model.apply(init_weights)
    return model


def load_model(model_path, cuda, weights=True):
    config = read_config(os.path.join(model_path, 'config.txt'))
    model = create_model(config, cuda)
    state_path = os.path.join(model_path, config[0][0]+'.pth')
    if os.path.exists(state_path) and weights:
        print('Loading weights from {}...'.format(state_path))
        if cuda:
            model.load_state_dict(torch.load(state_path, map_location='cuda'))
        else:
            model.load_state_dict(torch.load(state_path, map_location='cpu'))
    return model


def read_hparams(spec_file):
    with open(spec_file,'r') as f:
        spec = f.readlines()
    param_keys = [
        'batch_size',
        'num_epochs',
        'train_ratio',
        'lr',
        'num_workers'
    ]
    hparams = {}
    for i in range(len(param_keys)):
        if '.' in spec[i]:
            hparams[param_keys[i]] = float(spec[i])
        else:
            hparams[param_keys[i]] = int(spec[i])
    return hparams
