import sys
import matplotlib.pyplot as plt
import pandas as pd
import math

filepath = 'D:\PhD study\simplines-main\simplines/'
sys.path.append(filepath)
import bsplines as bs
import torch
from torch import autograd
import numpy as np
import random

## Read data from csv
## Loading and preparing the data
raw_data = pd.read_csv('D:\PhD study\simplines-main\dataBenchmark.csv')
U, y = raw_data['uVal'], raw_data['yVal']
sol = [y[i] for i in range(0, len(y), 5)]
plt.plot([i for i in range(1, len(y) + 1, 5)], sol, label='x2')
plt.plot(U, label='u')
plt.legend(loc="best", fontsize=10)
plt.show()

del_t = 0.00390625*5
T = 3.984375 + del_t
t = np.arange(0, T, del_t)
w2 = np.var(sol) / 100
w3 = np.var(sol)
w4 = 60
## Generate B-spline basis matrix
# The last entry of every b at location x represents the value of the span-th basis function at x
degree = 3
# del_tb = random.uniform(del_t / degree, del_t / (degree + 1))
del_tb = 0.02
knot = bs.make_knots(np.linspace(0, T - del_t, num=math.ceil(T / del_tb)), degree, False)
x = [del_t * i for i in range(0, len(sol))]
B = np.zeros((len(x), len(knot) - 1 - degree))
for i in range(0, len(x)):
    span = bs.find_span(knot, degree, x[i])
    b = bs.basis_funs(knot, degree, x[i], span)
    for j in range(0, degree + 1):
        B[i, span - degree + j] = b[j]

B_d = np.zeros((len(x), len(knot) - 1 - degree ))
for i in range(0, len(x)):
    span = bs.find_span(knot, degree, x[i])
    b = bs.basis_funs(knot, degree, x[i], span)
    for j in range(0, degree + 1):
        B_d[i, span - degree + j] = b[j]

B_2d = np.zeros((len(x), len(knot) - 1 - degree))
for i in range(0, len(x)):
    span = bs.find_span(knot, degree, x[i])
    b = bs.basis_funs_all_ders(knot, degree, x[i], span, 2)
    for j in range(0, degree + 1):
        B_2d[i, span - degree + j] = b[2, j]

                                        ## Generate new B for ODE
x_O = np.arange(0, T-del_t+del_t / 5, del_t / 5)
B_O = np.zeros((len(x_O), len(knot) - 1 - degree))
for i in range(0, len(x_O)):
    span = bs.find_span(knot, degree, x_O[i])
    b = bs.basis_funs(knot, degree, x_O[i], span)
    for j in range(0, degree + 1):
        B_O[i, span - degree + j] = b[j]

B_d_O = np.zeros((len(x_O), len(knot) - 1 - degree))
for i in range(0, len(x_O) ):
    span = bs.find_span(knot, degree, x_O[i])
    b = bs.basis_funs_1st_der(knot, degree, x_O[i], span)
    for j in range(0, degree + 1):
        B_d_O[i, span - degree + j] = b[j]

B_2d_O = np.zeros((len(x_O), len(knot) - 1 - degree))
for i in range(0, len(x_O) ):
    span = bs.find_span(knot, degree, x_O[i])
    b = bs.basis_funs_all_ders(knot, degree, x_O[i], span, 2)
    for j in range(0, degree + 1):
        B_2d_O[i , span - degree + j ] = b[2, j]

## Intialization

theta1 = torch.rand(1, 1)
theta1 = torch.tensor(theta1, dtype=torch.float64)
theta2 = torch.rand(1, 1)
theta2 = torch.tensor(theta2, dtype=torch.float64)
theta3 = torch.rand(1, 1)
theta3 = torch.tensor(theta3, dtype=torch.float64)
theta4 = torch.rand(1, 1)
theta4 = torch.tensor(theta4, dtype=torch.float64)

theta1.requires_grad_()
theta2.requires_grad_()
theta3.requires_grad_()
theta4.requires_grad_()
B = torch.from_numpy(B)
B_d = torch.from_numpy(B_d)
B_2d = torch.from_numpy(B_2d)
B_O = torch.from_numpy(B_O)
B_d_O = torch.from_numpy(B_d_O)
B_2d_O = torch.from_numpy(B_2d_O)
x2 = torch.rand(len(sol), 1)
for i in range(0, len(sol)):
    x2[i] = sol[i]
x2 = torch.tensor(x2, dtype=torch.float64)
u = torch.rand(1021, 1)
for i in range(1, 1022):
    u[i-1] = U[i]
u = torch.tensor(u, dtype=torch.float64)
beta=torch.matmul(torch.matmul(torch.inverse(torch.matmul(B.T,B)),B.T),x2)
beta = torch.tensor(beta, dtype=torch.float64)
beta.requires_grad_()
## Iteration of Successive Convex Approximation
maxiter = 2000
mu = 1
c = 10000
gamma = 1
epsilon = 0.8
fit_res = torch.rand(maxiter + 1, 1)
ode_res = torch.rand(maxiter + 1, 1)
res = torch.rand(maxiter + 1, 1)
for i in range(0, maxiter + 1):

    ## Compute the optimal solution of approximated convex function
    # Approximate beta

    F1 = torch.matmul(B_O, beta) * torch.matmul(B_O, beta) * torch.matmul(B_O, beta) * torch.matmul(B_O,
                                                                                              beta) + 112 * torch.matmul(
        B_O, beta) * torch.matmul(B_O, beta) * torch.matmul(B_O, beta) + 1120 * torch.matmul(B_O, beta) * torch.matmul(B_O,
                                                                                                               beta) + 1792 * torch.matmul(
        B_O, beta) + 1024
    F2 = theta3 * (theta1 * theta2 * (theta4 * u - torch.matmul(B_O, beta)) - torch.matmul(B_d_O, beta))
    F3 = 16 * torch.matmul(B_O, beta) * torch.matmul(B_O, beta) * torch.matmul(B_O, beta) + 448 * torch.matmul(B_O,
                                                                                                         beta) * torch.matmul(
        B_O, beta) + 1792 * torch.matmul(B_O, beta) + 1024
    f = mu / 2 / w3 *torch.norm(F1 - F2 * F3)
    f_grads = autograd.grad(outputs=f, inputs=beta)[0]
    Bbeta_up1 = torch.inverse(1 / w2 * torch.matmul(B.T, B) + c * np.eye(B.shape[1]))
    Bbeta_up2 = torch.matmul(1 / w2 * B.T, x2) + c * beta - f_grads
    Bbeta = torch.matmul(Bbeta_up1, Bbeta_up2)
    gamma = gamma * (1 - gamma * epsilon)
    beta = beta + gamma * (Bbeta - beta)

    x2_sq = F1 / F3
    # Update theta1
    theta1_up1 = theta4 * u - torch.matmul(B_O, beta)
    theta1_up2 = torch.inverse(torch.matmul(theta1_up1.T, theta1_up1))
    theta1_up3 = torch.matmul(theta1_up2, theta1_up1.T) / theta2
    theta1 = torch.matmul(theta1_up3, x2_sq / theta3 + torch.matmul(B_d_O, beta))

    # Update theta2
    theta2_up1 = theta4 * u - torch.matmul(B_O, beta)
    theta2_up2 = torch.inverse(torch.matmul(theta2_up1.T, theta2_up1))
    theta2_up3 = torch.matmul(theta2_up2, theta2_up1.T) / theta1
    theta2 = torch.matmul(theta2_up3, x2_sq / theta3 + torch.matmul(B_d_O, beta))

    # Update theta3
    theta3_up1 = theta1 * theta2 * (theta4 * u - torch.matmul(B_O, beta)) - torch.matmul(B_d_O, beta)
    theta3_up2 = torch.matmul(torch.inverse(torch.matmul(theta3_up1.T, theta3_up1)), theta3_up1.T)
    theta3 = torch.matmul(theta3_up2, x2_sq)

    # Update theta4
    theta4_up1 = torch.matmul(torch.inverse(torch.matmul(u.T, u)), u.T)
    theta4 = torch.matmul(theta4_up1, 1 / theta1 / theta2 * (
            x2_sq / theta3 + torch.matmul(B_d_O, beta)) + torch.matmul(B_O,
                                                                       beta))

    # Print iteration and residual of each term
    F1 = torch.matmul(B_O, beta) * torch.matmul(B_O, beta) * torch.matmul(B_O, beta) * torch.matmul(B_O,
                                                                                              beta) + 112 * torch.matmul(
        B_O, beta) * torch.matmul(B_O, beta) * torch.matmul(B_O, beta) + 1120 * torch.matmul(B_O, beta) * torch.matmul(B_O,
                                                                                                               beta) + 1792 * torch.matmul(
        B_O, beta) + 1024
    F2 = theta3 * (theta1 * theta2 * (theta4 * u - torch.matmul(B_O, beta)) - torch.matmul(B_d_O, beta))
    F3 = 16 * torch.matmul(B_O, beta) * torch.matmul(B_O, beta) * torch.matmul(B_O, beta) + 448 * torch.matmul(B_O,
                                                                                                         beta) * torch.matmul(
        B_O, beta) + 1792 * torch.matmul(B_O, beta) + 1024
    fit_res[i] = torch.norm(x2 - torch.matmul(B, beta))
    ode_res[i] = torch.norm(F1 - F2 * F3)
    res[i] = fit_res[i] + ode_res[i]
    if i % 100 == 0:
        print('Iteration=%d, Fitting residual=%f, ODE residual=%f, Total residual=%f' % (
            i, fit_res[i], ode_res[i], res[i]))

plt.scatter(x, x2[1:len(x2) - 1], label='Ture x2', c='g')
# plt.plot(x, sol2[1:len(sol1) - 1], label='Ture x2', c='g')
y2 = torch.matmul(B_O, beta)
plt.plot(x_O, y2.detach().numpy(), label='Fitting x2')
plt.legend(loc="best", fontsize=10)
plt.show()

plt.plot(fit_res.detach().numpy(), label='Fitting residual')
plt.plot(ode_res.detach().numpy(), label='ODE residual')
plt.plot(res.detach().numpy(), label='Total residual')
plt.legend(loc="best", fontsize=10)
plt.show()

ODE = F1 - F2 * F3
plt.plot(ODE.detach().numpy(), label='ODE residual')
plt.legend(loc="best", fontsize=10)
plt.show()