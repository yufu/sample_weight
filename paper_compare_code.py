# Generate compare images for papers

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
from PIL import Image
from scipy import stats
from scipy.stats import norm
from scipy.stats import probplot
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# Load Vit Balpoe Dists models and data
image_size=224
batch_size = 256
def vit_model_init(device, vit_name='google/vit-large-patch16-224-in21k'):
    model = ViTModel.from_pretrained(vit_name)
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model

colors = [[199,192,208], [215,203,177], [177,148,167],\
          [148,83,87], [86,132,145]]
colors = [[y/255 for y in x] for x in colors]


def main(config, posthoc_bias_correction=False):
   
    compare_distrib()



def compare_distrib():
    vit = torch.load("pth/total_dists_cifar100_p3_vit.pth").cpu()
    bal = torch.load("pth/total_dists_cifar100_balpoe_p2.pth").cpu()
    labels = torch.load("pth/total_labels_cifar100_p3_vit.pth").cpu()
    import os
    import matplotlib.pyplot as plt

    def compare_class_specified_distrib():
        save_dir = "./Comparisons/"
        # compare vit and bal distrib
        # for i in range(100):
        i = 33
        vit_i = vit[labels == i]
        bal_i = bal[labels == i]

        plt.clf()

        # Vit Figure
        fig, ax = plt.subplots()

        ax.hist(vit_i, bins=20, color=colors[0], alpha=0.7)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
        plt.savefig(os.path.join(save_dir, 'vit_balance_dataset_CLS_' + str(i)+".png"))
        plt.close()

        plt.clf()
        fig, ax = plt.subplots()
        probplot(vit_i,dist='norm', plot=ax)
        ax.get_lines()[0].set_color(colors[0])
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.savefig(os.path.join(save_dir, 'vit_balance_dataset_QQPLOT_CLS_' + str(i)+".png"))
        plt.close()


        # BalPoE Figure
        plt.clf()
        fig, ax = plt.subplots()
        ax.hist(bal_i, bins=20, color=colors[1], alpha=0.7)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

        plt.savefig(os.path.join(save_dir, 'BalPoE_CLS_' + str(i)+".png"))
        plt.close()

        plt.clf()
        fig, ax = plt.subplots()
        probplot(bal_i, dist='norm', plot=ax)
        ax.get_lines()[0].set_color(colors[1])
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.savefig(os.path.join(save_dir, 'BalPoE_QQPLOT_CLS_' + str(i)+".png"))
        plt.close()

        # PowerLaw Figure
        plt.clf()
        fig, ax = plt.subplots()
        power_law_dists = torch.pow(bal_i, 1/4)
        ax.hist(power_law_dists, bins=20, color=colors[2], alpha=0.7)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

        plt.savefig(os.path.join(save_dir, 'PowerLaw_0.25_CLS_'+ str(i)+".png"))
        plt.close()

        plt.clf()
        fig, ax = plt.subplots()
        probplot(power_law_dists, dist='norm', plot=ax)
        ax.get_lines()[0].set_color(colors[2])
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.savefig(os.path.join(save_dir, 'PowerLaw_0.25_QQPLOT_CLS_' + str(i)+".png"))
        plt.close()


        # KDE
        kernel = norm(loc=0, scale=1.0)
        x = np.linspace(bal_i.min(), bal_i.max(), 1000)
        kde_values = torch.tensor([kernel.pdf(x - d).mean() for d in bal_i.numpy()])
        plt.clf()
        fig, ax = plt.subplots()
        ax.hist(kde_values, bins=20, color=colors[3], alpha=0.7)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

        plt.savefig(os.path.join(save_dir, 'KDE_CLS_' + str(i)+".png"))
        plt.close()


        plt.clf()
        fig, ax = plt.subplots()
        probplot(kde_values, dist='norm', plot=ax)
        ax.get_lines()[0].set_color(colors[3])
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.savefig(os.path.join(save_dir, 'KDE_QQPLOT_CLS_' + str(i)+".png"))
        plt.close()

        # BOXCOX
        transformed_data, lambda_value = stats.boxcox(bal_i.numpy())
        transformed_data = (transformed_data - min(transformed_data))/ (max(transformed_data) - min(transformed_data))
        fig, ax = plt.subplots()
        ax.hist(transformed_data, bins=20, color=colors[4], alpha=0.7)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

        plt.savefig(os.path.join(save_dir, 'BoxCox_CLS_'+ str(i)+".png"))
        plt.close()

        plt.clf()
        fig, ax = plt.subplots()
        probplot(transformed_data, dist='norm', plot=ax)
        ax.get_lines()[0].set_color(colors[4])
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.savefig(os.path.join(save_dir, 'BoxCox_QQPLOT_CLS_' + str(i)+".png"))
        plt.close()

    compare_class_specified_distrib()


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
