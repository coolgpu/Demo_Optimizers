import torch
from torch.utils.data import DataLoader
import math
from matplotlib import pyplot as plt
from Ellipse_Dataset import EllipseDataset
from torch import optim


def main():
    flag_manual_implement = True # True: using our custom implementation; False: using Torch built-in
    flag_plot_final_result = True
    flag_log = True

    if flag_log:
        if flag_manual_implement:
            foutput = open("Adam_custom_implement_results.log", "w")
            foutput.write('Adam using custom implementation' + '\n')
        else:
            foutput = open("Adam_pytorch_built-in_results.log", "w")
            foutput.write('Adam using torch built-in' + '\n')

    device = torch.device('cpu')
    torch.manual_seed(9999)
    a = 1.261845
    b = 1.234378
    c = math.sqrt(a * a - b * b)
    nsamples = 512
    batch_size = 64
    epoch = 100
    learning_rate = 0.01
    beta1, beta2 = 0.9, 0.999
    eps = 1e-08

    xy_dataset = EllipseDataset(nsamples, a, b, noise_scale=0.1)
    xy_dataloader = DataLoader(xy_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    Wa = torch.rand([], device=device, requires_grad=True)
    Wb = torch.rand([], device=device, requires_grad=True)
    # Wa = torch.tensor(0.2866007089614868, device=device, requires_grad=True)
    # Wb = torch.tensor(0.7565606236457825, device=device, requires_grad=True)

    if flag_log:
        logstr = 'nsamples={}, batch_size={}, epoch={}, lr={}\ninitial Wa={}, Wb={}\n' \
            .format(nsamples, batch_size, epoch, learning_rate, Wa, Wb)
        foutput.write(logstr)
        print(logstr)

    VdWa = 0.0
    VdWb = 0.0
    SdWa = 0.0
    SdWb = 0.0
    beta1_to_pow_t = 1.0
    beta2_to_pow_t = 1.0

    if flag_manual_implement:
        optimizer = None
    else:
        optimizer = optim.Adam([Wa, Wb], lr=learning_rate, betas=(beta1, beta2), eps=eps)

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
                logstr = 'Epoch={}, minibatch={} loss={:.5f}, Wa={:.4f}, Wb={:.4f}\n'.format(t, i_batch, loss.item(), Wa.data.numpy(), Wb.data.numpy())
                foutput.write(logstr)
                if t % 10 == 0 and i_batch == 0:
                    print(logstr)

            if flag_manual_implement:
                # fully-manual: perform back-propagation and calculate the gradients of loss w.r.t. Wa and Wb
                dWa_via_yi = 2.0 * (y_pred - y) * ((x+c) ** 2) * (Wb**2) / (Wa**3) / y_pred
                dWa = dWa_via_yi[y_pred_sqr > 0.00000001].sum()

                dWb_via_yi = (2.0 * (y_pred - y) * y_pred / Wb)
                dWb = dWb_via_yi[y_pred_sqr > 0.00000001].sum()

                # Update weights using Adam algorithm.
                with torch.no_grad():
                    beta1_to_pow_t *= beta1
                    beta1_correction = 1.0 - beta1_to_pow_t
                    beta2_to_pow_t *= beta2
                    beta2_correction = 1.0 - beta2_to_pow_t

                    VdWa = beta1 * VdWa + (1.0 - beta1) * dWa
                    SdWa = beta2 * SdWa + (1.0 - beta2) * dWa * dWa
                    Wa -= learning_rate * (VdWa / beta1_correction) / (torch.sqrt(SdWa) / math.sqrt(beta2_correction) + eps)

                    VdWb = beta1 * VdWb + (1.0 - beta1) * dWb
                    SdWb = beta2 * SdWb + (1.0 - beta2) * dWb * dWb
                    Wb -= learning_rate * (VdWb / beta1_correction) / (torch.sqrt(SdWb) / math.sqrt(beta2_correction) + eps)
            else:
                # Use Torch built-in autograd and optim to do the job
                optimizer.zero_grad()
                loss.backward()
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
        yfit = Wb * torch.sqrt(1.0 - (x+c)**2 / Wa**2)
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

