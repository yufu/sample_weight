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

    # evaluate_dist_image()
    # evaluate(train_data_loader, device, model)
    compare_distrib()
    predict(train_data_loader, device, model)

# def compare_distrib_branch():
#     folder = 'hist_distrib_branch'
#     file = 'pth/total_dists_cifar_branch.pth'
#     dists = torch.load(file)
#     for expert in dists:


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
            total_dists[total_labels == i] = torch.cdist(class_centers[i][None, :], total_feat[total_labels == i], p=3)
        # calculate distances

        torch.save(total_feat, "total_feat_cifar100_p3_vit.pth")
        torch.save(class_centers, 'class_centers_cifar100_p3_vit.pth')
        torch.save(total_labels, 'total_labels_cifar100_p3_vit.pth')
        torch.save(total_dists, 'total_dists_cifar100_p3_vit.pth')

def predict(train_data_loader, device, model):
    class_number = 100

    total_feat = torch.empty((0)).cuda()
    total_labels = torch.empty((0)).cuda()
    class_centers = torch.empty((0)).cuda()
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(train_data_loader)):
            data, target = data.to(device), target.to(device)
            logits = model(data).logits
            predicted_class = torch.argmax(logits, dim=1).item()
            c = 1



def evaluate_dist_image():
    # class_centers = torch.load("class_centers.pth")
    total_labels = torch.load('total_labels_cifar.pth')
    total_dists = torch.load('total_dists_cifar100_balpoe.pth')
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
        
        dir = "./images_balpoe/" + str(k)
        if not os.path.exists(dir):
            os.mkdir(dir)
        for i in range(len(sort_k_dists_ind)):
            image = Image.fromarray(k_images[i])

            # 保存图像
            image.save(os.path.join(dir,f'{sort_k_dists_val[i].item():.3f}' + ".pdf"))  # 将图像保存为pdf文件


def compare_distrib():
    vit = torch.load("total_dists_vit_balance.pth").cpu()
    bal = torch.load("total_dists_cifar100_balpoe_p2.pth").cpu()
    labels = torch.load("total_labels_vit_balance.pth").cpu()
    import os
    import matplotlib.pyplot as plt




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
        plt.savefig(name + ".pdf")

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
        plt.savefig("qq_bal_cifar100_norm_p2_box_cox.pdf")


    # skew_kurtosis(vit)
    # skew_kurtosis(bal)
    # skew_kurtosis_box_cox(bal)
    # bal, vit = box_cox_2(bal, vit, labels)
    # qq_draw(bal, name='bal_max_min_all')
    verify_CLT(vit, bal)







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
