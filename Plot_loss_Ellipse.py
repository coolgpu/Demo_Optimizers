import torch
from torch.utils.data import DataLoader
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from Ellipse_Dataset import EllipseDataset
from torch import optim
import pandas as pd

def read_optimizer_results(filename):
    names = ['updateStr', 'update', 'EpochStr', 'Epoch', 'minibatchStr', 'minibatch', 'lossStr', 'loss', 'WaStr', 'Wa',
             'WbStr', 'Wb']
    df = pd.read_csv(filename, names=names, skiprows=3, skipfooter=2, engine='python', sep='[,=]')
    WaTraces = df['Wa'].values
    WbTraces = df['Wb'].values
    LossTraces = df['loss'].values
    EpochTraces = df['Epoch'].values
    minibatchTraces = df['minibatch'].values
    updateTraces = df['update'].values
    return WaTraces, WbTraces, LossTraces, EpochTraces, minibatchTraces, updateTraces

def main():
    device = torch.device('cpu')
    torch.manual_seed(9999)
    a = 1.261845
    b = 1.234378
    c = math.sqrt(a * a - b * b)
    nsamples = 512
    batch_size = 512

    # load previously generated results
    WaTraces_Adam, WbTraces_Adam, LossTraces_Adam, EpochTraces, minibatchTraces, updateTraces = read_optimizer_results(r'Adam_custom_implement_results.log')
    WaTraces_SGD, WbTraces_SGD, LossTraces_SGD, _, _, _ = read_optimizer_results(r'SGD_custom_implement_results.log')
    WaTraces_SGD_MOMENTUM, WbTraces_SGD_MOMENTUM, LossTraces_SGD_MOMENTUM, _, _, _ = read_optimizer_results(r'SGD_w_Momentum_custom_implement_results.log')
    WaTraces_RMSprop, WbTraces_RMSprop, LossTraces_RMSprop, _, _, _ = read_optimizer_results(r'RMSprop_custom_implement_results.log')

    nframes = len(WaTraces_Adam)

    xy_dataset = EllipseDataset(nsamples, a, b, noise_scale=0.1)
    xy_dataloader = DataLoader(xy_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Wa0 = torch.rand([], device=device, requires_grad=True)
    # Wb0 = torch.rand([], device=device, requires_grad=True)
    Wa0 = WaTraces_Adam[0]
    Wb0 = WbTraces_Adam[0]

    nWaGrids, nWbGrids = 200, 200
    WaGrid = np.linspace(0, 2.0, nWaGrids)
    WbGrid = np.linspace(0, 2.0, nWbGrids)

    Wa2d, Wb2d = np.meshgrid(WaGrid, WaGrid)
    loss = np.zeros(Wa2d.shape)

    for i_batch, sample_batched in enumerate(xy_dataloader):
        x, y = sample_batched['x'], sample_batched['y']

    for indexb, Wb in enumerate(WbGrid):
        for indexa, Wa in enumerate(WaGrid):
            y_pred_sqr = Wb ** 2 * (1.0 - (x + c) ** 2 / Wa ** 2)
            y_pred_sqr[y_pred_sqr < 0.00000001] = 0.00000001  # handle negative values caused by noise
            y_pred = torch.sqrt(y_pred_sqr)

            loss[indexb, indexa] = (y_pred - y).pow(2).sum()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.contour(WbGrid, WaGrid, loss, levels=15, linewidths=0.5, colors='gray')
    cntr1 = ax.contourf(WaGrid, WbGrid, loss, levels=100, cmap="RdBu_r")
    fig.colorbar(cntr1, ax=ax)
    ax.set(xlim=(0, 2.0), ylim=(0, 2.0))
    ax.set_title('Loss as a function of Wa and Wb')
    plt.xlabel("Wa")
    plt.ylabel("Wb")
    ax.set_aspect('equal', adjustable='box')
    ax.plot(a, b, 'yo', ms=3)
    ax.plot(Wa0, Wb0, 'ko', ms=3)

    plt.show(block=False)

    wblist_Adam = []
    walist_Adam = []
    point_Adam, = ax.plot([], [], 'go', lw=0.5, markersize=4)
    line_Adam, = ax.plot([], [], '-g', lw=2)

    wblist_RMSprop = []
    walist_RMSprop = []
    point_RMSprop, = ax.plot([], [], 'ko', lw=0.5, markersize=4)
    line_RMSprop, = ax.plot([], [], '-k', lw=2)

    wblist_SGD = []
    walist_SGD = []
    point_SGD, = ax.plot([], [], 'yo', lw=0.5, markersize=4)
    line_SGD, = ax.plot([], [], '-y', lw=2)

    wblist_SGD_MOMENTUM = []
    walist_SGD_MOMENTUM = []
    point_SGD_MOMENTUM, = ax.plot([], [], 'mo', lw=0.5, markersize=4)
    line_SGD_MOMENTUM, = ax.plot([], [], '-m', lw=2)

    text_update = ax.text(0.03, 0.03, '', transform=ax.transAxes, color="white", fontsize=14)

    # initialization function: plot the background of each frame
    def init():

        point_Adam.set_data([], [])
        line_Adam.set_data([], [])

        point_RMSprop.set_data([], [])
        line_RMSprop.set_data([], [])

        point_SGD.set_data([], [])
        line_SGD.set_data([], [])

        point_SGD_MOMENTUM.set_data([], [])
        line_SGD_MOMENTUM.set_data([], [])

        text_update.set_text('')

        return point_Adam, line_Adam, point_RMSprop, line_RMSprop, point_SGD, line_SGD, point_SGD_MOMENTUM, line_SGD_MOMENTUM, text_update

    # animation function.  This is called sequentially
    def animate(i):
        if i == 0:
            wblist_Adam[:] = []
            walist_Adam[:] = []

            wblist_RMSprop[:] = []
            walist_RMSprop[:] = []

            wblist_SGD[:] = []
            walist_SGD[:] = []

            wblist_SGD_MOMENTUM[:] = []
            walist_SGD_MOMENTUM[:] = []

        wa_Adam, wb_Adam  = WaTraces_Adam[i], WbTraces_Adam[i]
        wblist_Adam.append(wa_Adam)
        walist_Adam.append(wb_Adam)
        point_Adam.set_data(wa_Adam, wb_Adam)
        line_Adam.set_data(wblist_Adam, walist_Adam)

        wa_RMSprop, wb_RMSprop  = WaTraces_RMSprop[i], WbTraces_RMSprop[i]
        wblist_RMSprop.append(wa_RMSprop)
        walist_RMSprop.append(wb_RMSprop)
        point_RMSprop.set_data(wa_RMSprop, wb_RMSprop)
        line_RMSprop.set_data(wblist_RMSprop, walist_RMSprop)

        wa_SGD, wb_SGD  = WaTraces_SGD[i], WbTraces_SGD[i]
        wblist_SGD.append(wa_SGD)
        walist_SGD.append(wb_SGD)
        point_SGD.set_data(wa_SGD, wb_SGD)
        line_SGD.set_data(wblist_SGD, walist_SGD)

        wa_SGD_MOMENTUM, wb_SGD_MOMENTUM  = WaTraces_SGD_MOMENTUM[i], WbTraces_SGD_MOMENTUM[i]
        wblist_SGD_MOMENTUM.append(wa_SGD_MOMENTUM)
        walist_SGD_MOMENTUM.append(wb_SGD_MOMENTUM)
        point_SGD_MOMENTUM.set_data(wa_SGD_MOMENTUM, wb_SGD_MOMENTUM)
        line_SGD_MOMENTUM.set_data(wblist_SGD_MOMENTUM, walist_SGD_MOMENTUM)

        update, epoch, minibatch = updateTraces[i], EpochTraces[i]+1, minibatchTraces[i]+1
        text_update.set_text('Update = {:d}, epoch={:d}, minibatch={:d}'.format(update, epoch, minibatch))

        return point_Adam, line_Adam, point_RMSprop, line_RMSprop, point_SGD, line_SGD, point_SGD_MOMENTUM, line_SGD_MOMENTUM, text_update

    # call the animator.  blit=True means only re-draw the parts that have changed.
    intervalms = 10  # this means 10 ms per frame
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nframes, interval=intervalms, blit=True)




    print('Done!')

if __name__ == '__main__':
    main()

