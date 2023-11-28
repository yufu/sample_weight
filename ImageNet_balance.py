
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
from data_loader.imagenet_lt_data_loaders import ImageNetLTDataLoader
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import numpy as np
from parse_config import ConfigParser
import torch.nn.functional as F
from transformers import ViTFeatureExtractor, ViTModel

from utils import adjusted_model_wrapper
import os
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

    # build model architecture
    model = vit_model_init(device)

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)


    # prepare model for testing
    model = model.to(device)
    model.eval()
 
    num_classes = config._config["arch"]["args"]["num_classes"]



    for i in range(num_classes):
        test_txt  = 'data_txt/ImageNet/%s.txt'%(i)
        print(test_txt)
        data_loader = ImageNetLTDataLoader(
            config['data_loader']['args']['data_dir'],
            batch_size=128,
            shuffle=False,
            training=False,
            num_workers=2,
            test_txt=test_txt
        )

        if posthoc_bias_correction:
            test_prior = torch.tensor(data_loader.cls_num_list).float().to(device)
            test_prior = test_prior / test_prior.sum()
            test_bias = test_prior.log()
        else:
            test_bias = None

        root = "./ImageNet_dists"
        calculate_distances(data_loader, device, model, num_classes, i, root)

def calculate_distances(train_data_loader, device, model, class_number, index, root):
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
        
        class_center = total_feat.mean(dim=0)
        total_dists = torch.cdist(class_center.unsqueeze(0), total_feat, p=2)
        # calculate distances

        torch.save(total_dists.squeeze(dim=0), os.path.join(root, str(index) + "_dist.pth"))

def mic_acc_cal(preds, labels):
    if isinstance(labels, tuple):
        assert len(labels) == 3
        targets_a, targets_b, lam = labels
        acc_mic_top1 = (lam * preds.eq(targets_a.data).cpu().sum().float() \
                       + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()) / len(preds)
    else:
        acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1
   

def validation(data_loader, model, num_classes, device):
    b = np.load("./data/imagenet_lt_shot_list.npy")
    many_shot = b[0]
    medium_shot = b[1] 
    few_shot = b[2]
    confusion_matrix = torch.zeros(num_classes, num_classes).cuda()
    total_logits = torch.empty((0, num_classes)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            for t, p in zip(target.view(-1), output.argmax(dim=1).view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            total_logits = torch.cat((total_logits, output))
            total_labels = torch.cat((total_labels, target))  

    probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)

    # Calculate the overall accuracy and F measurement
    eval_acc_mic_top1= mic_acc_cal(preds[total_labels != -1],
                                        total_labels[total_labels != -1])
        
    print('All top-1 Acc:', np.round(eval_acc_mic_top1 * 100, decimals=2))
    acc_per_class = confusion_matrix.diag()/confusion_matrix.sum(1)
    acc = acc_per_class.cpu().numpy() 
    many_shot_acc = acc[many_shot].mean()
    medium_shot_acc = acc[medium_shot].mean()
    few_shot_acc = acc[few_shot].mean()
    print("{}, {}, {}".format(np.round(many_shot_acc * 100, decimals=2), np.round(medium_shot_acc * 100, decimals=2), np.round(few_shot_acc * 100, decimals=2)))
    return np.round(many_shot_acc * 100, decimals=2), np.round(medium_shot_acc * 100, decimals=2), np.round(few_shot_acc * 100, decimals=2), np.round(eval_acc_mic_top1 * 100, decimals=2)
 
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
