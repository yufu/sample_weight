import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.model as module_arch
import numpy as np
from parse_config import ConfigParser
import torch.nn.functional as F
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


def main():

    compare_distrib()



def compare_distrib():
    path = "./ImageNet_dist_balance"
    if not os.path.exists(path):
        os.mkdir(path)
    pth_dir = "./ImageNet_dists"
    for i in range(41):
        pth_path = os.path.join(pth_dir, str(i) + "_dist.pth")
        vit = torch.load(pth_path).cpu()
        plt.clf()
        plt.hist(vit, bins=20,color='blue', alpha=0.7)
        plt.savefig(os.path.join(path, str(i) + ".jpeg"))


def compare_class_specified_distrib():
    save_dir = "./"
    # compare vit and bal distrib
    for i in range(100):
        vit_i = vit[labels == i]
        bal_i = bal[labels == i]
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # 1行2列的子图
        ax1.hist(vit_i, bins=20, color='blue', alpha=0.7)
        ax2.hist(bal_i, bins=20, color='blue', alpha=0.7)
        ax1.set_title("vit")
        ax2.set_title("bal")
        plt.savefig(os.path.join(save_dir, str(i)+"pdf"))
        plt.close()

def balanced_class_specified_distrib(vit):
    save_dir = "./blc_vit_hist"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # compare vit and bal distrib
    for i in range(100):
        vit_i = vit[labels == i]
        plt.clf()
        plt.hist(vit_i, bins=20, color='blue', alpha=0.7)
        plt.title("vit distance distribution balanced cifar 100")
        plt.savefig(os.path.join(save_dir, str(i)+"pdf"))
        plt.close()
def verify_CLT(vit, bal):
    means_vit = np.empty((0))
    means_bal = np.empty((0))
    for i in range(100):
        vit_i = vit[labels == i]
        # bal_i = bal[labels == i]
        means_vit = np.concatenate((means_vit, vit_i.mean().unsqueeze(dim=0).numpy()))
        # means_bal = np.concatenate((means_bal, bal_i.mean().unsqueeze(dim=0).numpy()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # 1行2列的子图
    ax1.hist(means_vit, bins=20, color='blue', alpha=0.7)
    ax2.hist(means_vit, bins=20, color='blue', alpha=0.7)
    ax1.set_title("vit balance CLT")
    ax2.set_title("bal CLT")
    plt.savefig("verify_clt_blc.pdf")
    plt.close()
def compare_all_distrib():
    # compare all distrib
    save_dir = "./"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # 1行2列的子图
    ax1.hist(vit, bins=20, color='blue', alpha=0.7)
    ax2.hist(bal, bins=20, color='blue', alpha=0.7)
    ax1.set_title("vit")
    ax2.set_title("bal")
    plt.savefig(os.path.join(save_dir, "vit_bal_all_compare.pdf"))
    plt.close()

def plot_ax(ax, data, name):
    ax.hist(data, bins=20, color='blue', alpha=0.7)
    ax.set_title(name)

def pow_bal(bal, vit, labels):
    save_dir = "./bal_hist"

    # use pow for bal
    for i in range(100):
        bal_i = bal[labels == i]
        vit_i = vit[labels == i]
        plt.clf()
        fig, (l1, l2) = plt.subplots(2, 6, figsize=(20, 8))  # 1行2列的子图

        plot_ax(l1[0], vit_i, name='vit_i')
        for k in range(1, 6):
            plot_ax(l1[k], torch.pow(bal_i, 1/k), name=f"bal.pow(1/{k}")

        for m in range(0, 6):
            plot_ax(l2[m], torch.pow(bal_i, 1 / (m+6)), name=f"bal.pow(1/{m+6})")

        plt.savefig(os.path.join(save_dir, str(i)+".pdf"))
        plt.close()
def box_cox(bal, vit, labels):
    from scipy import stats
    import numpy as np
    save_dir = "./boxcox"
    for i in range(100):
        bal_i = bal[labels == i]
        vit_i = vit[labels == i]
        transformed_data, lambda_value = stats.boxcox(bal_i.numpy())
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 8))  # 1行2列的子图
        plot_ax(ax0, vit_i, name='vit')
        plot_ax(ax1, transformed_data, name='box-cox')
        plt.savefig(os.path.join(save_dir, str(i)+".pdf"))
        plt.close()

def box_cox_2(bal, vit, labels):
    from scipy import stats
    import numpy as np
    save_dir = "./"
    for i in range(100):
        bal_i = bal[labels == i]
        vit_i = vit[labels == i]
        transformed_data, lambda_value = stats.boxcox(bal_i.numpy())
        transformed_data = transformed_data / (max(transformed_data) - min(transformed_data))
        bal_i = bal_i / (max(bal_i) - min(bal_i))

        # get gap between two datasets
        gap = bal_i.min() - transformed_data.min()

        # get the final result
        transformed_data = transformed_data + gap.numpy()
        bal[labels == i] = torch.tensor(transformed_data)
    return bal, vit
    # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 8))  # 1行2列的子图
    # plot_ax(ax0, vit, name='vit')
    # plot_ax(ax1, bal, name='box-cox')
    # plt.savefig(os.path.join(save_dir, "box-cox-class-specific-all-norm.pdf"))
    # plt.close()

def box_cox_all(bal, vit):
    from scipy import stats
    import numpy as np
    save_dir = "./"
    transformed_data, lambda_value = stats.boxcox(bal.numpy())
    transformed_data = transformed_data / (max(transformed_data) - min(transformed_data))
    bal = bal / (max(bal) - min(bal))

    # get gap between two datasets
    gap = bal.min() - transformed_data.min()

    # get the final result
    transformed_data = transformed_data + gap.numpy()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 8))  # 1行2列的子图
    plot_ax(ax0, vit, name='vit')
    plot_ax(ax1, transformed_data, name='box-cox')
    plt.savefig(os.path.join(save_dir, "vit_bal_all_boxcox_translate_compare.pdf"))
    plt.close()


def e_transform(bal, vit, labels):
    import os
    save_dir = "./exp_hist"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i in range(100):
        bal_i = bal[labels == i]
        vit_i = vit[labels == i]
        transformed_data =  torch.exp(bal_i)
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 8))  # 1行2列的子图
        plot_ax(ax0, vit_i, name='vit')
        plot_ax(ax1, transformed_data, name='exp')
        plt.savefig(os.path.join(save_dir, str(i)+".pdf"))
        plt.close()
def e_transform2(bal, vit, labels):
    import os
    save_dir = "./pow(exp(-x)_-1)_hist"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i in range(100):
        bal_i = bal[labels == i]
        vit_i = vit[labels == i]
        transformed_data =  1/torch.exp(-bal_i)
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 8))  # 1行2列的子图
        plot_ax(ax0, vit_i, name='vit')
        plot_ax(ax1, transformed_data, name='1/exp(-x)')
        plt.savefig(os.path.join(save_dir, str(i)+".pdf"))
        plt.close()


def log_transform(bal, vit, labels):
    import os
    save_dir = "./log_hist"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i in range(100):
        bal_i = bal[labels == i]
        vit_i = vit[labels == i]
        transformed_data =  torch.log(bal_i)
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 8))  # 1行2列的子图
        plot_ax(ax0, vit_i, name='vit')
        plot_ax(ax1, transformed_data, name='log')
        plt.savefig(os.path.join(save_dir, str(i)+".pdf"))
        plt.close()


def kde(bal, vit, labels):
    import os
    save_dir = "./kde_hist"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    from scipy.stats import norm
    kernel = norm(loc=0, scale=1.0)
    for i in range(100):
        bal_i = bal[labels == i]
        vit_i = vit[labels == i]
        kernel = norm(loc=0, scale=1.0)
        x = np.linspace(bal_i.min(), bal_i.max(), 1000)
        kde_values = torch.tensor([kernel.pdf(x - d).mean() for d in bal_i.numpy()])
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 8))  # 1行2列的子图
        plot_ax(ax0, vit_i, name='vit')
        plot_ax(ax1, kde_values, name='kde_values')
        plt.savefig(os.path.join(save_dir, str(i) + ".pdf"))
        plt.close()
# e_transform2(bal, vit, labels)
def shapiro_wilk_test(data):
    from scipy.stats import shapiro
    import numpy as np

    # Perform Shapiro-Wilk test
    statistic, p_value = shapiro(data)

    # Print the results
    print(f"Shapiro-Wilk Statistic: {statistic}")
    print(f"P-value: {p_value}")

    # Interpret the results
    alpha = 0.05
    if p_value > alpha:
        print("Fail to reject the null hypothesis. The data looks normally distributed.")
    else:
        print("Reject the null hypothesis. The data does not look normally distributed.")

def skew_kurtosis(data):
    from scipy.stats import skew, kurtosis
    import numpy as np


    # Calculate skewness and kurtosis
    skewness = skew(data)
    kurt = kurtosis(data)

    # Print the results
    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurt}")

    # Interpret the results
    # Skewness close to 0 indicates a symmetric distribution.
    # Kurtosis close to 3 indicates a normal distribution.

def skew_kurtosis_box_cox(bal):
    from scipy.stats import skew, kurtosis
    import numpy as np
    from scipy import stats

    transformed_data, lambda_value = stats.boxcox(bal.numpy())
    transformed_data = transformed_data / (max(transformed_data) - min(transformed_data))
    bal = bal / (max(bal) - min(bal))

    # get gap between two datasets
    gap = bal.min() - transformed_data.min()

    # get the final result
    transformed_data = transformed_data + gap.numpy()
    # Calculate skewness and kurtosis
    skewness = skew(transformed_data)
    kurt = kurtosis(transformed_data)

    # Print the results
    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurt}")
def qq_draw(data, name, distrib='norm'):
    from scipy.stats import probplot

    # 绘制 Q-Q 图
    probplot(data, dist=distrib, plot=plt)


    plt.title('Q-Q Plot - Normal Distribution')
    plt.savefig('Comparisons/' + name + ".pdf")

def qq_draw_box_cox(bal):
    from scipy import stats
    from scipy.stats import probplot

    transformed_data, lambda_value = stats.boxcox(bal.numpy())
    transformed_data = transformed_data / (max(transformed_data) - min(transformed_data))
    bal = bal / (max(bal) - min(bal))

    # get gap between two datasets
    gap = bal.min() - transformed_data.min()

    # get the final result
    transformed_data = transformed_data + gap.numpy()
    probplot(transformed_data, dist='norm', plot=plt)
    plt.title('Q-Q Plot -Box-cox transformed data ')
    plt.savefig("Comparisons/qq_bal_cifar100_norm_p2_box_cox.pdf")


    # skew_kurtosis(vit)
    # skew_kurtosis(bal)
    # skew_kurtosis_box_cox(bal)
    # bal, vit = box_cox_2(bal, vit, labels)
    # qq_draw(bal, name='bal_max_min_all')







if __name__ == '__main__':

    main()
