# -*- coding = utf-8 -*-
# @File : visualization.py
# @Software : PyCharm
import numpy.random
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP


def visualization_by_umap(features, targets, title, axis_font_size=16, title_font_size=36):
    features = features.reshape((features.shape[0], -1))
    reducer = UMAP()
    embedding = reducer.fit_transform(features)
    plt.figure(figsize=(10, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=targets, cmap='Spectral', s=10)
    plt.title(title, fontsize=title_font_size)
    plt.xticks(fontsize=axis_font_size)
    plt.yticks(fontsize=axis_font_size)
    plt.savefig('images/' + "umap_" + title + '.png')
    plt.close()



def visualization_by_tsne(features, targets, title, axis_font_size=16, title_font_size=36):
    features = features.reshape((features.shape[0], -1))
    reducer = TSNE()
    embedding = reducer.fit_transform(features)
    plt.figure(figsize=(10, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=targets, cmap='Spectral', s=10)
    plt.title(title, fontsize=title_font_size)
    plt.xticks(fontsize=axis_font_size)
    plt.yticks(fontsize=axis_font_size)
    plt.savefig('images/' + "tsne_" + title + '.png')
    plt.close()



if __name__ == '__main__':
    features = numpy.random.rand(100, 32, 125, 5)
    targets = numpy.random.randn(100)

    visualization_by_umap(features, targets, "umap_test")
    visualization_by_tsne(features, targets, "tsne_test")
