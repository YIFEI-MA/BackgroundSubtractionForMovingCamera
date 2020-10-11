import matplotlib.pyplot as plt
import numpy as np

from skimage.segmentation import slic, mark_boundaries
from skimage.data import astronaut


def segmentation(img, n_scales):
    n_segments = 10
    labels = []
    for i in range(1, n_scales+1):
        i = np.random.randint(i*n_segments, 100*i+1)
        labels.append([slic(img, n_segments*50*i, compactness=10.0, max_iter=20, sigma=1, enforce_connectivity=True, start_label=1), n_segments*i])

    return labels


if __name__ == '__main__':
    image = astronaut()
    labels_list = segmentation(image, 6)
    fig, ax = plt.subplots(3, 3, figsize=(30, 20), sharex=True, sharey=True)
    for i in range(len(labels_list)):
        title = labels_list[i][1]
        j = 0
        if i >= 3:
            j = 1
            i -= 3
        ax[j, i].imshow(mark_boundaries(image, labels_list[i][0]))
        ax[j, i].set_title((title, np.max(labels_list[i][0])))

    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()
