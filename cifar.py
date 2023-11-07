import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
from data_loader.cifar_data_loaders import TestAgnosticImbalanceCIFAR100DataLoader
import model.model as module_arch
import numpy as np
from parse_config import ConfigParser
import torch.nn.functional as F
from data_loader.imbalance_cifar import IMBALANCECIFAR100
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

image_size=224
batch_size = 256
def vit_model_init(device, vit_name='google/vit-large-patch16-224-in21k'):
    model = ViTModel.from_pretrained(vit_name)
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
        batch_size=batch_size,
        shuffle=False,
        training=True,
        num_workers=8,
        imb_factor=config['data_loader']['args']['imb_factor'],
        image_size=image_size
    )
    train_cls_num_list = train_data_loader.cls_num_list
    train_cls_num_list = torch.tensor(train_cls_num_list)
    many_shot = train_cls_num_list > 100
    few_shot = train_cls_num_list < 20
    medium_shot = ~many_shot & ~few_shot

    evaluate_dist_image()
    # evaluate(train_data_loader, device, model)


def evaluate(train_data_loader, device, model):
    class_number = 100

    total_feat = torch.empty((0)).cuda()
    total_labels = torch.empty((0)).cuda()
    class_centers = torch.empty((0)).cuda()
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(train_data_loader)):
            data, target = data.to(device), target.to(device)
            feat = model(data)['pooler_output']
            total_feat = torch.cat((total_feat, feat))
            total_labels = torch.cat((total_labels, target))
        # calculate class centers and distances
        total_dists = torch.zeros_like(total_labels).cuda()
        for i in range(class_number):
            class_centers = torch.cat((class_centers, total_feat[total_labels == i].mean(dim=0)[None, :]), dim=0)
            total_dists[total_labels == i] = torch.cdist(class_centers[i][None, :], total_feat[total_labels == i], p=2)
        # calculate distances

        torch.save(total_feat, "total_feat_cifar.pth")
        torch.save(class_centers, 'class_centers.pth')
        torch.save(total_labels, 'total_labels.pth')
        torch.save(total_dists, 'total_dists.pth')

def evaluate_dist_image():
    class_centers = torch.load("class_centers.pth")
    total_feat = torch.load('total_feat_cifar.pth')
    total_labels = torch.load('total_labels.pth')
    total_dists = torch.load('total_dists.pth')
    data_all = torch.empty((0))
    transform = transforms.Compose(
        [transforms.Resize(image_size), 
        transforms.ToTensor()
        ])
    imcifar100dataset = IMBALANCECIFAR100(root='./data', download=True, transform=transform)
    dataloader = DataLoader(imcifar100dataset, batch_size=1, shuffle=False)

    # 创建一个列表来存储所有图像
    all_images = []

    # 遍历整个数据集并将图像添加到列表中
    for data in dataloader:
        images, labels = data
        all_images.append(images)
    data_all = imcifar100dataset.data
    targets_all = imcifar100dataset.targets
    import os

    for k in range(100):
        k_images = data_all[total_labels.cpu() == k]
        k_dists = total_dists[total_labels == k]
        sort_k_dists_val, sort_k_dists_ind = torch.sort(k_dists)
        
        dir = "./images/" + str(k)
        if not os.path.exists(dir):
            os.mkdir(dir)
        for i in range(len(sort_k_dists_ind)):
            image = Image.fromarray(k_images[i])

            # 保存图像
            image.save(os.path.join(dir,f'{sort_k_dists_val[i].item():.3f}' + ".png"))  # 将图像保存为PNG文件




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
