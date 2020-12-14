import itertools

import matplotlib.pyplot as plt
import numpy as np

def show_result(num_epoch, G_net):
    randn_in = np.random.randn(5*5, 100)

    test_images = G_net.predict(randn_in)

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k] * 0.5 + 0.5))

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig("results/epoch_" + str(num_epoch) + "_results.png")
    plt.close('all')  #避免内存泄漏