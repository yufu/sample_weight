import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 手动指定簇的中心点
centers = np.random.uniform(low=-7, high=7, size=(4, 2))
proto = centers.mean(axis=0)
dis = np.array([np.linalg.norm(centers[i]-proto) for i in range(len(centers))])
ind = dis.argsort()
dis.sort()
centers_new = centers[ind]
# 生成符合幂律分布的簇大小
num_clusters = 4
# cluster_sizes = np.array([1 / (i + 1) for i in range(num_clusters)])  # 幂律分布示例，你可以根据需要调整
# cluster_sizes /= np.sum(cluster_sizes)  # 归一化到总和为1
# total_samples = 200
# cluster_sizes *= total_samples  # 调整总样本数
cluster_sizes = [40, 20, 10, 5]
# 生成带有不同簇大小的样本数据
X = np.concatenate([np.random.normal(center, 1 , size=(int(size), 2)) for d, center, size in zip(dis, centers_new, cluster_sizes)])
proto = X.mean(axis=0)
# 使用 K-Means 聚类
kmeans = KMeans(n_clusters=num_clusters, init=centers, n_init=1, random_state=42)  # 使用手动指定的中心点进行初始化
kmeans.fit(X)

# 可视化聚类结果
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5)
plt.scatter(proto[0], proto[1], marker='*', c='red', s=200, label='Prototype')
plt.legend()
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('K-Means Clustering with Power Law Distributed Cluster Sizes')
plt.savefig('test_clus5.pdf')
