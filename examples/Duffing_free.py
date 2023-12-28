import sys
import matplotlib.pyplot as plt

filepath = 'D:\PhD study\simplines-main\simplines/'
sys.path.append(filepath)
import bsplines as bs
import torch
from torch import autograd
from scipy.integrate import odeint
import numpy as np
import random
import math

## Generate data by solving ODE
def pfun(x, t):
    x1, x2 = x
    return np.array([x2, -10 * x1 - 0.5 * x2 - 30 * x1 ** 3])

del_t = 0.1
T = 8 + del_t
t = np.arange(0, T, del_t)
sol = odeint(pfun, [1, 1], t)
# Add noise into solution
sol1 = sol[:, 0]
sol2 = sol[:, 1]
sol3 = -10 * sol1 - 0.5 * sol2 - 30 * sol1 ** 3
w1 = np.var(sol1)
w2 = np.var(sol2)
w3 = np.var(-10 * sol1 - 0.5 * sol2 - 30 * sol1 ** 3)/500
m = 0
sigma = 0.1
for i in range(len(sol)):
    sol1[i] += random.gauss(m, sigma)
    sol2[i] += random.gauss(m, sigma)
    sol3[i] += random.gauss(m, sigma)
plt.plot(sol1)
plt.plot(sol2)
plt.plot(sol3)
plt.show()

## Generate B-spline basis matrix
# The last entry of every b at location x represents the value of the span-th basis function at x
degree = 3
del_tb = random.uniform(del_t / degree, del_t / (degree + 1))
knot = bs.make_knots(np.linspace(0, T - del_t, num=math.ceil(T / del_tb)), degree, False)

# x = [del_t * i for i in range(1, len(sol1) - 1)]
# B = np.zeros((len(x), len(knot) - 1 - degree - 2 * degree))

x = [del_t * i for i in range(1, len(sol1) - 1)]
B = np.zeros((len(x), len(knot) - 1 - degree - 2 * degree))
for i in range(0, len(x)):
    span = bs.find_span(knot, degree, x[i])
    b = bs.basis_funs(knot, degree, x[i], span)
    for j in range(0, degree + 1):
        B[i, span - 2 * degree + j] = b[j]

B_d = np.zeros((len(x), len(knot) - 1 - degree - 2 * degree))
for i in range(0, len(x)):
    span = bs.find_span(knot, degree, x[i])
    b = bs.basis_funs(knot, degree, x[i], span)
    for j in range(0, degree + 1):
        B_d[i, span - 2 * degree + j] = b[j]

B_2d = np.zeros((len(x), len(knot) - 1 - degree - 2 * degree))
for i in range(0, len(x)):
    span = bs.find_span(knot, degree, x[i])
    b = bs.basis_funs_all_ders(knot, degree, x[i], span, 2)
    for j in range(0, degree + 1):
        B_2d[i, span - 2 * degree + j] = b[2, j]

## Generate new B for ODE
x_O = np.arange(del_t, T - 2 * del_t + del_t / 10, del_t / 10)
B_O = np.zeros((len(x_O), len(knot) - 1 - degree - 2 * degree))
for i in range(0, len(x_O)):
    span = bs.find_span(knot, degree, x_O[i])
    b = bs.basis_funs(knot, degree, x_O[i], span)
    for j in range(0, degree + 1):
        B_O[i, span - 2 * degree + j] = b[j]

B_d_O = np.zeros((len(x_O), len(knot) - 1 - degree - 2 * degree))
for i in range(0, len(x_O) ):
    span = bs.find_span(knot, degree, x_O[i])
    b = bs.basis_funs_1st_der(knot, degree, x_O[i], span)
    for j in range(0, degree + 1):
        B_d_O[i - 1, span - degree + j - degree] = b[j]

B_2d_O = np.zeros((len(x_O), len(knot) - 1 - degree - 2 * degree))
for i in range(0, len(x_O)):
    span = bs.find_span(knot, degree, x_O[i])
    b = bs.basis_funs_all_ders(knot, degree, x_O[i], span, 2)
    for j in range(0, degree + 1):
        B_2d_O[i - 1, span - degree + j - degree] = b[2, j]

## Intialization
beta = torch.rand(B.shape[1], 1)
beta = torch.tensor(beta, dtype=torch.float64)
theta2 = torch.rand(3, 1)
theta2 = torch.tensor(theta2, dtype=torch.float64)
beta.requires_grad_()
theta2.requires_grad_()
B = torch.from_numpy(B)
B_d = torch.from_numpy(B_d)
B_2d = torch.from_numpy(B_2d)
B_O = torch.from_numpy(B_O)
B_d_O = torch.from_numpy(B_d_O)
B_2d_O = torch.from_numpy(B_2d_O)
x1 = torch.rand(len(sol1), 1)
x2 = torch.rand(len(sol2), 1)
x3 = torch.rand(len(sol3), 1)
for i in range(0, len(sol)):
    x1[i] = sol1[i]
    x2[i] = sol2[i]
    x3[i] = sol3[i]
x1 = torch.tensor(x1, dtype=torch.float64)
x2 = torch.tensor(x2, dtype=torch.float64)
x3 = torch.tensor(x3, dtype=torch.float64)
## Iteration of Successive Convex Approximation
maxiter = 2000
mu = 1000
c = 20000
gamma = 1
epsilon = 0.5
a1 = 0
a2 = 0
a3 = 10000
fit_res = torch.rand(maxiter + 1, 1)
ode_res = torch.rand(maxiter + 1, 1)
res = torch.rand(maxiter + 1, 1)
for i in range(0, maxiter + 1):

    # Compute the optimal solution of approximated convex function
    B_beta3 = torch.matmul(B_O, beta) * torch.matmul(B_O, beta) * torch.matmul(B_O, beta)
    F2 = torch.cat([torch.cat([torch.matmul(B_O, beta), torch.matmul(B_d_O, beta)], 1), B_beta3], 1)
    f = mu / 2 * torch.norm(torch.matmul(B_2d_O, beta) - torch.matmul(F2, theta2))
    f_grads = autograd.grad(outputs=f, inputs=beta, retain_graph=1)[0]

    Bbeta_up1 = torch.inverse(
        a1 * 1 / w1 * torch.matmul(B.T, B) + a2 * 1 / w2 * torch.matmul(B_d.T, B_d)+a3 * 1 / w3 * torch.matmul(B_2d.T, B_2d) + c * np.eye(B.shape[1]))
    Bbeta_up2 = a1 * torch.matmul(1 / w1 * B.T, x1[1:len(x1)-1]) + a2 * torch.matmul(1 / w2 * B_d.T,
                                                                                             x2[1:len(x1)-1]) +a3 * torch.matmul(1 / w3 * B_2d.T, x3[1:len(x3)-1])+ c * beta - f_grads
    Bbeta = torch.matmul(Bbeta_up1, Bbeta_up2)

    # Compute step size and update beta
    gamma = gamma * (1 - gamma * epsilon)
    beta = beta + gamma * (Bbeta - beta)

    # Update theta2
    B_beta3 = torch.matmul(B_O, beta) * torch.matmul(B_O, beta) * torch.matmul(B_O, beta)
    F2 = torch.cat([torch.cat([torch.matmul(B_O, beta), torch.matmul(B_d_O, beta)], 1), B_beta3], 1)
    theta2_up1 = torch.inverse(torch.matmul(F2.T, F2))
    theta2_up2 = torch.matmul(F2.T, torch.matmul(B_2d_O, beta))
    theta2 = torch.matmul(theta2_up1, theta2_up2)

    # Print iteration and residual of each term
    fit_res[i] = a1*torch.norm(x1[1:len(x1)-1] - torch.matmul(B, beta)) + a2*torch.norm(
        x2[1:len(x1)-1] - torch.matmul(B_d, beta))+a3*torch.norm(
        x3[1:len(x1)-1] - torch.matmul(B_2d, beta))
    ode_res[i] = torch.norm(torch.matmul(B_2d_O, beta) - torch.matmul(F2, theta2))
    res[i] = fit_res[i] + ode_res[i]
    if i % 100 == 0:
        print('Iteration=%d, Fitting residual=%f, ODE residual=%f, Total residual=%f, Parameter estimation=' % (
            i, fit_res[i], ode_res[i], res[i]))
        print(theta2.detach().numpy().T)

# Plot figures to check results
plt.scatter(x, sol1[1:len(x1)-1], label='Ture x1', c='g')
plt.plot(x, sol1[1:len(x1)-1], label='Ture x1', c='g')
y1 = torch.matmul(B_O, beta)
plt.plot(x_O, y1.detach().numpy(), label='Fitting x1')
plt.legend(loc="best", fontsize=10)
plt.show()

plt.scatter(x, sol2[1:len(x1)-1], label='Ture x2', c='g')
plt.plot(x, sol2[1:len(x1)-1], label='Ture x2', c='g')
y2 = torch.matmul(B_d_O, beta)
plt.plot(x_O, y2.detach().numpy(), label='Fitting x2')
plt.legend(loc="best", fontsize=10)
plt.show()

plt.scatter(x, sol3[1:len(x3)-1], label='Ture x3', c='g')
plt.plot(x, sol3[1:len(x1)-1], label='Ture x3', c='g')
y3 = torch.matmul(B_2d_O, beta)
plt.plot(x_O, y3.detach().numpy(), label='Fitting x3')
plt.legend(loc="best", fontsize=10)
plt.show()

plt.plot(fit_res.detach().numpy(), label='Fitting residual')
plt.plot(ode_res.detach().numpy(), label='ODE residual')
plt.plot(res.detach().numpy(), label='Total residual')
plt.legend(loc="best", fontsize=10)
plt.show()

ODE = torch.matmul(B_2d_O, beta) - torch.matmul(F2, theta2)
plt.plot(ODE.detach().numpy(), label='ODE residual')
plt.legend(loc="best", fontsize=10)
plt.show()


## Define the exact line search algorithm
# def Adam(Bsol, Bbeta, beta, B_O, B_d_O, B_2d_O, theta1, theta2, mu, tt, flag, w1, x1, i):
#     gamma = torch.tensor(0.1, dtype=torch.float64)
#     gamma.requires_grad_()
#     beta1 = 0.9
#     beta2 = 0.999
#     epsilon = 10e-8
#     alpha = 0.001
#     m_j = 0
#     v_j = 0
#     if flag == 1:
#         gbeta = beta + gamma * (Bsol - beta)
#         gtheta3 = theta3
#         threshold = 1e-10
#     else:
#         gbeta = beta
#         gtheta3 = theta3 + gamma * (Bsol - theta3)
#         if i == 0:
#             threshold = 1e-3
#         else:
#             threshold = 1e-4
#     F2 = torch.cat([torch.cat([B_O @ gbeta, B_d_O @ gbeta], 1), (B_O @ gbeta) * (B_O @ gbeta) * (B_O @ gbeta)], 1)
#     F1 = torch.cos(tt * gtheta3)
#     f = mu / 2 * torch.norm(B_2d_O @ gbeta - F2 @ theta2 - theta1 * F1) + gamma * 1 / w1 * (
#             torch.norm(x1[1:len(x1) - 1] - B @ Bbeta) - torch.norm(x1[1:len(x1) - 1] - B @ beta))
#     gamma_grads = autograd.grad(outputs=f, inputs=gamma)[0]
#     m_j = beta1 * m_j + (1 - beta1) * gamma_grads
#     v_j = beta2 * v_j + (1 - beta2) * gamma_grads ** 2
#     alpha_j = alpha * np.sqrt(1 - beta2) / (1 - beta1)
#     gamma_new = gamma - alpha_j * m_j / (np.sqrt(v_j) + epsilon * np.sqrt(1 - beta2))
#     j = 2
#     while (1):
#         if torch.abs(gamma_new - gamma) > threshold:
#             gamma = gamma_new
#             if flag == 1:
#                 gbeta = beta + gamma * (Bsol - beta)
#                 gtheta3 = theta3
#             else:
#                 gbeta = beta
#                 gtheta3 = theta3 + gamma * (Bsol - theta3)
#             F2 = torch.cat([torch.cat([B_O @ gbeta, B_d_O @ gbeta], 1), (B_O @ gbeta) * (B_O @ gbeta) * (B_O @ gbeta)],
#                            1)
#             F1 = torch.cos(tt * gtheta3)
#             f = mu / 2 * torch.norm(B_2d_O @ gbeta - F2 @ theta2 - theta1 * F1) + gamma * 1 / w1 * (
#                     torch.norm(x1[1:len(x1) - 1] - B @ Bbeta) - torch.norm(x1[1:len(x1) - 1] - B @ beta))
#             gamma_grads = autograd.grad(outputs=f, inputs=gamma)[0]
#             m_j = beta1 * m_j + (1 - beta1) * gamma_grads
#             v_j = beta2 * v_j + (1 - beta2) * gamma_grads ** 2
#             alpha_j = alpha * np.sqrt(1 - beta2 ** j) / (1 - beta1 ** j)
#             gamma_new = gamma - alpha_j * m_j / (np.sqrt(v_j) + epsilon * np.sqrt(1 - beta2 ** j))
#             j += 1
#             # print(gamma_grads,gamma_new)
#             if gamma_new > 1:
#                 gamma_new = 1
#                 break
#             elif gamma_new < 0:
#                 gamma_new = 0
#                 break
#         else:
#             break
#
#     return gamma_new