import torch
from torch.utils.data import DataLoader
import math
from matplotlib import pyplot as plt
from Ellipse_Dataset import EllipseDataset
from torch import optim


def main():
    flag_manual_implement = False  # True: using our custom implementation; False: using Torch built-in
    flag_plot_final_result = True
    flag_log = True

    if flag_log:
        if flag_manual_implement:
            foutput = open("SGD_w_Momentum_custom_implement_results.log", "w")
            foutput.write('SDG_w_Momentum using custom implementation' + '\n')
        else:
            foutput = open("SGD_w_Momentum_pytorch_built-in_results.log", "w")
            foutput.write('SDG_w_Momentum using torch built-in' + '\n')

    device = torch.device('cpu')
    torch.manual_seed(9999)
    a = 1.261845
    b = 1.234378
    c = math.sqrt(a * a - b * b)
    nsamples = 512
    batch_size = 512
    epoch = 100
    learning_rate = 0.0005 # 1e-4
    momentum = 0.9
    dampening = 0.0

    xy_dataset = EllipseDataset(nsamples, a, b, noise_scale=0.1)
    xy_dataloader = DataLoader(xy_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    Wa = torch.rand([], device=device, requires_grad=True)
    Wb = torch.rand([], device=device, requires_grad=True)
    # Wa = torch.tensor(0.2866007089614868, device=device, requires_grad=True)
    # Wb = torch.tensor(0.7565606236457825, device=device, requires_grad=True)

    if flag_log:
        logstr = 'nsamples={}, batch_size={}, epoch={}, lr={}\ninitial Wa={}, Wb={}\n'\
            .format(nsamples, batch_size, epoch, learning_rate, Wa, Wb)
        foutput.write(logstr)
        print(logstr)

    VdWa = 0.0
    VdWb = 0.0

    if flag_manual_implement:
        optimizer = None
    else:
        optimizer = optim.SGD([Wa, Wb], lr=learning_rate, momentum=momentum, dampening=dampening)
    for t in range(epoch):
        for i_batch, sample_batched in enumerate(xy_dataloader):
            x, y = sample_batched['x'], sample_batched['y']

            # Forward pass
            y_pred_sqr = Wb ** 2 * (1.0 - (x + c) ** 2 / Wa ** 2)
            y_pred_sqr[y_pred_sqr < 0.00000001] = 0.00000001  # handle negative values caused by noise
            y_pred = torch.sqrt(y_pred_sqr)

            # Compute loss
            loss = (y_pred - y).pow(2).sum()

            if flag_log:
                logstr = 'Epoch={}, minibatch={} loss={:.5f}, Wa={:.4f}, Wb={:.4f},'.format(t, i_batch, loss.item(), Wa.data.numpy(), Wb.data.numpy())
                foutput.write(logstr)
                if t % 10 == 0 and i_batch == 0:
                    print(logstr)

            if flag_manual_implement:
                # fully-manual: perform back-propagation and calculate the gradients of loss w.r.t. Wa and Wb
                dWa_via_yi = 2.0 * (y_pred - y) * ((x+c) ** 2) * (Wb**2) / (Wa**3) / y_pred
                dWa = dWa_via_yi[y_pred_sqr > 0.00000001].sum()

                dWb_via_yi = (2.0 * (y_pred - y) * y_pred / Wb)
                dWb = dWb_via_yi[y_pred_sqr > 0.00000001].sum()

                logstr = ' dWa={:.4f}, dWb={:.4f}\n'.format(dWa.data.numpy(), dWb.data.numpy())
                foutput.write(logstr)

                # Update weights using Gradient Descent algorithm.
                with torch.no_grad():
                    VdWa = momentum * VdWa + (1.0 - dampening) * dWa
                    Wa -= learning_rate * VdWa

                    VdWb = momentum * VdWb + (1.0 - dampening) * dWb
                    Wb -= learning_rate * VdWb
            else:
                # Use Torch built-in autograd and optim to do the job
                optimizer.zero_grad()
                loss.backward()
                logstr = ' dWa={:.4f}, dWb={:.4f}\n'.format(Wa.grad.data.numpy(), Wb.grad.data.numpy())
                foutput.write(logstr)
                optimizer.step()

    # log the final results
    if flag_log:
        logstr = 'The ground truth is A={:.4f}, B={:.4f}\n'.format(a, b)
        if flag_manual_implement:
            logstr += 'Manually implemented gradient+optimizer result: Final estimated Wa={:.4f}, Wb={:.4f}\n'.format(Wa, Wb)
        else:
            logstr += 'PyTorch built-in AutoGradient+optimizer result: Final estimated Wa={:.4f}, Wb={:.4f}\n'.format(Wa, Wb)
        foutput.write(logstr)
        foutput.close()
        print(logstr)

    # plot the results obtained from the training
    if flag_plot_final_result:
        x = xy_dataset[:]['x']
        yfit = Wb * torch.sqrt(1.0-(x+c)**2/Wa**2)
        yfit[yfit!=yfit] = 0.0  # take care of the "Nan" at the end-points due to sqrt(negative_value_caused_by_noise)
        plt.plot(x, yfit.detach().numpy(), color="purple", linewidth=2.0)
        strEquation = r'$\frac{{{\left({x+' + '{:.3f}'.format(c) + r'}\right)}^2}}{{' + '{:.3f}'.format(Wa) + r'^2}}+\frac{y^2}{' + '{:.3f}'.format(Wb) + r'^2}=1$'
        x0, y0 = x.detach().numpy()[nsamples * 2 // 3], yfit.detach().numpy()[nsamples * 2 // 3]
        plt.annotate(strEquation, xy=(x0, y0), xycoords='data',
                     xytext=(+0.75, 1.75), textcoords='data', fontsize=16,
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        plt.show()

    print('Done!')


if __name__ == '__main__':
    main()
