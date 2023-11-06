import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
from data_loader.cifar_data_loaders import TestAgnosticImbalanceCIFAR100DataLoader
import model.model as module_arch
import numpy as np
from parse_config import ConfigParser
import torch.nn.functional as F

from utils import adjusted_model_wrapper
from transformers import ViTFeatureExtractor, ViTForImageClassification

def vit_model_init(device, vit_name='google/vit-large-patch16-384-in21k'):
    model = ViTForImageClassification.from_pretrained(vit_name)
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model

def main(config, posthoc_bias_correction=False):
    logger = config.get_logger('test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setup data_loader instances
    model = vit_model_init(device)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    # checkpoint = torch.load(config.resume)
    # state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    # model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    train_data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=128,
        shuffle=False,
        training=True,
        num_workers=8,
        imb_factor=config['data_loader']['args']['imb_factor']
    )
    train_cls_num_list = train_data_loader.cls_num_list
    train_cls_num_list = torch.tensor(train_cls_num_list)
    many_shot = train_cls_num_list > 100
    few_shot = train_cls_num_list < 20
    medium_shot = ~many_shot & ~few_shot

    num_classes = config._config["arch"]["args"]["num_classes"]

    total_feat = torch.empty((0)).cuda()
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(train_data_loader)):
            data, target = data.to(device), target.to(device)
            feat = model(data)
            feat = model(data)
            total_feat = torch.cat((total_feat, feat))



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-l', '--log-config', default='logger/logger_config.json', type=str,
                      help='logging config file path (default: logger/logger_config.json)')
    args.add_argument("--posthoc_bias_correction", dest="posthoc_bias_correction", action="store_true", default=False)

    # dummy arguments used during training time
    args.add_argument("--validate")
    args.add_argument("--use-wandb")

    config, args = ConfigParser.from_args(args)
    main(config, args.posthoc_bias_correction)
