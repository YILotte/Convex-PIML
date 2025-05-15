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
    return np.array([x2, -1.7 * (x1 ** 2 - 1) * x2 - x1])

del_t = 0.1
T = 5 + del_t
t = np.arange(0, T, del_t)
sol = odeint(pfun, [1, 1], t)
# Add noise into solution
sol1 = sol[:, 0]
sol2 = sol[:, 1]
w1 = np.var(sol1) / 500
w2 = np.var(sol2)
w3 = np.var(-1.7 * (sol1 ** 2 - 1) * sol2 - sol1)
m = 0
sigma = 0.2
for i in range(len(sol)):
    sol1[i] += random.gauss(m, sigma)
    sol2[i] += random.gauss(m, sigma)
plt.plot(sol1)
plt.plot(sol2)
plt.show()
## Generate B-spline basis matrix
# The last entry of every b at location x represents the value of the span-th basis function at x
degree = 3
del_tb = random.uniform(del_t / degree, del_t / (degree + 1))
knot = bs.make_knots(np.linspace(0, T, num=math.ceil(T / del_tb)), degree, False)

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
x_O = np.arange(del_t, T - 2*del_t + del_t / 10, del_t / 10)
B_O = np.zeros((len(x_O), len(knot) - 1 - degree - 2 * degree))
for i in range(0, len(x_O)):
    span = bs.find_span(knot, degree, x_O[i])
    b = bs.basis_funs(knot, degree, x_O[i], span)
    for j in range(0, degree + 1):
        B_O[i, span - 2 * degree + j] = b[j]

B_d_O = np.zeros((len(x_O), len(knot) - 1 - degree - 2 * degree))
for i in range(1, len(x_O) - 1):
    span = bs.find_span(knot, degree, x_O[i])
    b = bs.basis_funs_1st_der(knot, degree, x_O[i], span)
    for j in range(0, degree + 1):
        B_d_O[i - 1, span - degree + j - degree] = b[j]

B_2d_O = np.zeros((len(x_O), len(knot) - 1 - degree - 2 * degree))
for i in range(1, len(x_O) - 1):
    span = bs.find_span(knot, degree, x_O[i])
    b = bs.basis_funs_all_ders(knot, degree, x_O[i], span, 2)
    for j in range(0, degree + 1):
        B_2d_O[i - 1, span - degree + j - degree] = b[2, j]

## Intialization
beta = torch.rand(B.shape[1], 1)
beta = torch.tensor(beta, dtype=torch.float64)
theta2 = -200*torch.rand(1, 1)
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
for i in range(0, len(sol)):
    x1[i] = sol1[i]
    x2[i] = sol2[i]
x1 = torch.tensor(x1, dtype=torch.float64)
x2 = torch.tensor(x2, dtype=torch.float64)

## Iteration of Successive Convex Approximation
maxiter = 6000
mu = 80
c = 200
gamma = 1
epsilon = 0.5
crtl1 = 1
crtl2 = 1
fit_res = torch.rand(maxiter + 1, 1)
ode_res = torch.rand(maxiter + 1, 1)
res = torch.rand(maxiter + 1, 1)
for i in range(0, maxiter + 1):

    # Compute the optimal solution of approximated convex function
    F2 = (torch.matmul(B_O, beta) * torch.matmul(B_O, beta) - 1) * torch.matmul(B_d_O, beta)
    f = mu / 2 / w3 * torch.norm(torch.matmul(B_2d_O, beta) + theta2 * F2 + torch.matmul(B_O,
                                                                                         beta))  # Only this block need to be modified when governing equation change

    f_grads = autograd.grad(outputs=f, inputs=beta, retain_graph=1)[0]

    Bbeta_up1 = torch.inverse(
        crtl1 * 1 / w1 * torch.matmul(B.T, B) + crtl2 * 1 / w2 * torch.matmul(B_d.T, B_d) + c * np.eye(B.shape[1]))
    Bbeta_up2 = crtl1 * torch.matmul(1 / w1 * B.T, x1[1:len(x1)-1]) + crtl2 * torch.matmul(1 / w2 * B_d.T,
                                                                              x2[1:len(x2)-1]) + c * beta - f_grads
    Bbeta = torch.matmul(Bbeta_up1, Bbeta_up2)
    # Compute step size and update beta
    gamma = gamma * (1 - gamma * epsilon)
    beta = beta + gamma * (Bbeta - beta)

    # Update theta2 with analytic solution
    F2 = (torch.matmul(B_O, beta) * torch.matmul(B_O, beta) - 1) * torch.matmul(B_d_O, beta)
    theta2_up1 = torch.inverse(torch.matmul(F2.T, F2))
    theta2_up2 = torch.matmul(F2.T, -torch.matmul(B_2d_O, beta)-torch.matmul(B_O, beta))
    theta2 = torch.matmul(theta2_up1, theta2_up2)

    # Print iteration and residual of each term
    fit_res[i] = 1 / 2 / w1 * torch.norm(x1[1:len(x1)-1] - torch.matmul(B, beta)) + 1 / 2 / w2 * torch.norm(
        x2[1:len(x2)-1] - torch.matmul(B_d, beta))
    ode_res[i] = 1 / 2 / w3 * mu * torch.norm(torch.matmul(B_2d_O, beta) + torch.matmul(F2, theta2))
    res[i] = fit_res[i] + ode_res[i]
    if i % 100 == 0:
        print('Iteration=%d, Fitting residual=%f, ODE residual=%f, Total residual=%f' % (
            i, fit_res[i], ode_res[i], res[i]))

# Plot figures to check results
plt.scatter(x, sol1[1:len(sol1) - 1], label='Ture x1', c='g')
#plt.plot(x, sol1[1:len(sol1) - 1], label='Ture x1', c='g')
y1 = torch.matmul(B_O, beta)
plt.plot(x_O, y1.detach().numpy(), label='Fitting x1')
plt.legend(loc="best", fontsize=10)
plt.show()

plt.scatter(x, sol2[1:len(sol1) - 1], label='Ture x2', c='g')
#plt.plot(x, sol2[1:len(sol1) - 1], label='Ture x2', c='g')
y2 = torch.matmul(B_d_O, beta)
plt.plot(x_O, y2.detach().numpy(), label='Fitting x2')
plt.legend(loc="best", fontsize=10)
plt.show()

plt.plot(fit_res.detach().numpy(), label='Fitting residual')
plt.plot(ode_res.detach().numpy(), label='ODE residual')
plt.plot(res.detach().numpy(), label='Total residual')
plt.legend(loc="best", fontsize=10)
plt.show()

ODE = torch.matmul(B_2d_O, beta) + torch.matmul(F2, theta2)
plt.plot(ODE.detach().numpy(), label='ODE residual')
plt.legend(loc="best", fontsize=10)
plt.show()