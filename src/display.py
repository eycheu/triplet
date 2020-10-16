import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_image(data, target):
    fig, ax = plt.subplots(figsize=(6,3))
    iax = ax.imshow(data[0,:,:], interpolation="bicubic", cmap=plt.cm.RdBu_r);
    fig.colorbar(iax, ax=ax)
    ax.set_title("Class {}".format(target))

def plot_2d_circle(ax, radius=1):
    a = [a for a in np.arange(0, 360, 0.1)]
    x = np.cos(a) * radius
    y = np.sin(a) * radius
    ax.fill_between(x, y, alpha=0.1)

def plot_2d_embedding(embeddings, targets, ax, xlim=None, ylim=None, n_classes=2, kde=True, s=10):
    if kde:
        sns.kdeplot(embeddings[:,0], embeddings[:,1], ax=ax, cbar=False)
    
    for c in range(n_classes):
        inds = np.where(targets==c)[0]
        ax.scatter(embeddings[inds,0], embeddings[inds,1], s=s, alpha=0.8, label="Class {}".format(c))
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    ax.legend()
    ax.grid()