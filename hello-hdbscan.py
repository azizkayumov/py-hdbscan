import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load time-series data:
# data = np.genfromtxt('timeseries.csv', delimiter=',')
# data.reshape(-1, 1)

# Load 2D data:
# data = np.genfromtxt('single.csv', delimiter=',', names=True)
# data = np.array([data['x'], data['y']]).T

data = np.load('clusterable_data.npy')
print(data.shape)

clusterer = hdbscan.HDBSCAN(min_cluster_size=15).fit(data)
color_palette = sns.color_palette('deep', 8)
cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]
plt.scatter(*data.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
plt.savefig("clusters.png")
plt.show()

clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette('deep', 8))
plt.savefig("condensed_tree.png")
