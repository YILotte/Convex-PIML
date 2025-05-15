import sys
import matplotlib.pyplot as plt

filepath = 'D:\PhD study\simplines-main\simplines/'
sys.path.append(filepath)

import torch
from torch import autograd

import numpy as np

## Iteration of Successive Convex Approximation
maxiter =1000
mu = 100
c = 100
gamma = 1
epsilon = 0.1
for i in range(0, maxiter + 1):
    # Update theta2
    B_beta3 = torch.matmul(B_O, beta) * torch.matmul(B_O, beta) * torch.matmul(B_O, beta)
    F2 = torch.cat([torch.cat([torch.matmul(B_O, beta), torch.matmul(B_d_O, beta)], 1), B_beta3], 1)
    theta2_up1 = torch.inverse(torch.matmul(F2.T, F2))
    theta2_up2 = torch.matmul(F2.T, torch.matmul(B_2d_O, beta))
    theta2 = torch.matmul(theta2_up1, theta2_up2)

    # Compute the optimal solution of approximated convex function
    f2 = torch.matmul(B_2d_O, beta) - torch.matmul(F2, theta2)
    f = mu / 2 / w3 * torch.norm(torch.matmul(B_2d_0, beta) - torch.matmul(F2, theta2))
    f_grads = autograd.grad(outputs=f, inputs=beta)[0]

    Bbeta_up1 = torch.inverse(
        1 / w1 * torch.matmul(B.T, B) + 1 / w2 * torch.matmul(B_d.T, B_d) + c * np.eye(B.shape[1]))
    Bbeta_up2 = torch.matmul(1 / w1 * B.T, x1[1:120]) + torch.matmul(1 / w2 * B_d.T, x2[1:120]) + c * beta - f_grads
    Bbeta = torch.matmul(Bbeta_up1, Bbeta_up2)

    # Compute stepsize and update beta
    gamma = gamma * (1 - gamma * epsilon)
    beta = beta + gamma * (Bbeta - beta)

    # Print iteration and residual of each term
    fit_res = torch.norm(x1[1:120] - torch.matmul(B, beta)) +  torch.norm(
        x2[1:120] - torch.matmul(B_d, beta))
    ode_res = torch.norm(torch.matmul(B_2d, beta) - torch.matmul(F2, theta2))

    print('Iteration=%d, Fitting residual=%f, ODE residual=%f' % (i, fit_res, ode_res))

plt.plot(sol1[1:120])
y1 = torch.matmul(B, beta)
plt.plot(y1.detach().numpy())
plt.show()

plt.plot(sol2[1:120])
y2 = torch.matmul(B_d, beta)
plt.plot(y2.detach().numpy())
plt.show()