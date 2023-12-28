import sys
import matplotlib.pyplot as plt
from torch import autograd
import torch
import numpy as np
import math
import pandas as pd
import cmath
from mpl_toolkits.mplot3d import Axes3D

filepath = 'D:\PhD study\simplines-main\simplines/'
sys.path.append(filepath)
import bsplines as bs
import plotly.graph_objects as go
import random
import plotly

## Read data
raw_data = pd.read_csv('D:\PhD study\simplines-main/burgers.csv')
U = raw_data['u']
w1 = np.var(U)

## Generate Bt for fitting
del_t = 0.04  # The interval of original data
T = 1 + del_t
t = np.arange(0, T, del_t)  # If the sampling rate is not uniform, t is the true sampling rate
degree = 2
# del_tb = random.uniform(del_t / degree, del_t / (degree + 1))
del_tb = del_t / (degree + 0.1)
knot = bs.make_knots(np.linspace(0, T - del_t, num=math.ceil(T / del_tb)), degree, False)

# t1 = t[1:len(t) - 1]  # Give up the first and the last data point of t to avoid clamped effect
t1 = raw_data['t']
t1 = np.unique(t1)
Bt = np.zeros((len(t1), len(knot) - 1 - degree - 2 * degree))
Bt_d = np.zeros((len(t1), len(knot) - 1 - degree - 2 * degree))
Bt_2d = np.zeros((len(t1), len(knot) - 1 - degree - 2 * degree))
for i in range(0, len(t1)):
    span = bs.find_span(knot, degree, t1[i])
    b = bs.basis_funs_all_ders(knot, degree, t1[i], span, 2)
    for j in range(0, degree + 1):
        Bt[i, span - 2 * degree + j] = b[0, j]
        Bt_d[i, span - 2 * degree + j] = b[1, j]
        Bt_2d[i, span - 2 * degree + j] = b[2, j]

## Generate Bt_P for PDE
Nx = 10
Nt = 10
t1_P = np.linspace(t1[0], t1[-1],
                   num=Nt * len(t1))

Bt_P = np.zeros((len(t1_P), len(knot) - 1 - degree - 2 * degree))
Bt_d_P = np.zeros((len(t1_P), len(knot) - 1 - degree - 2 * degree))
for i in range(0, len(t1_P)):
    span = bs.find_span(knot, degree, t1_P[i])
    b = bs.basis_funs_all_ders(knot, degree, t1_P[i], span, 2)
    for j in range(0, degree + 1):
        Bt_P[i, span - 2 * degree + j] = b[0, j]
        Bt_d_P[i, span - 2 * degree + j] = b[1, j]

## Genrate Bx for fitting
del_x = 0.1  # The distance between sensors
X = 1 + del_x  # The distance between the first and the last sensor
x = np.arange(-1, X, del_x)
degree = 3
# del_xb = random.uniform(del_x / degree, del_x / (degree + 1))
del_xb = del_x / (degree + 0.1)
knot = bs.make_knots(np.linspace(-1, X - del_x, num=math.ceil((X + 1) / del_xb)), degree, False)

# x1 = x[1:len(x) - 1]  # If the sensors are arranged unevenly, x1 is the true distance
x1 = raw_data['x']
x1 = np.unique(x1)
Bx = np.zeros((len(x1), len(knot) - 1 - degree - 2 * degree))
for i in range(0, len(x1)):
    span = bs.find_span(knot, degree, x1[i])
    b = bs.basis_funs_all_ders(knot, degree, x1[i], span, 2)
    for j in range(0, degree + 1):
        Bx[i, span - 2 * degree + j] = b[0, j]

## Genrate Bx for PDE
x1_P = np.linspace(x1[0], x1[-1], num=Nx * len(x1))  # If the sensors are arranged unevenly, x1 is the true distance
Bx_P = np.zeros((len(x1_P) * len(t1_P), len(knot) - 1 - degree - 2 * degree))

## Generate two-dimensional basis function matrix B for fitting
B = np.zeros((len(x1) * len(t1), Bt.shape[1] * Bx.shape[1]))
for i in range(0, len(x1)):
    span = bs.find_span(knot, degree, x1[i])
    b = bs.basis_funs_all_ders(knot, degree, x1[i], span, 2)
    for j in range(span - 2 * degree, span + 1 - degree):
        B[len(t1) * i:len(t1) * (i + 1), Bt.shape[1] * j:Bt.shape[1] * (j + 1)] = Bt * b[0, j - span + 2 * degree]

## Generate two-dimensional basis function matrix B_P for PDE
B_P = np.zeros((len(x1_P) * len(t1_P), Bt_P.shape[1] * Bx_P.shape[1]))
B_t_P = np.zeros((len(x1_P) * len(t1_P), Bt_P.shape[1] * Bx_P.shape[1]))
B_x_P = np.zeros((len(x1_P) * len(t1_P), Bt_P.shape[1] * Bx_P.shape[1]))
B_2x_P = np.zeros((len(x1_P) * len(t1_P), Bt_P.shape[1] * Bx_P.shape[1]))
for i in range(0, len(x1_P)):
    span = bs.find_span(knot, degree, x1_P[i])
    b = bs.basis_funs_all_ders(knot, degree, x1_P[i], span, 2)
    for j in range(span - 2 * degree, span + 1 - degree):
        B_P[len(t1_P) * i:len(t1_P) * (i + 1), Bt_P.shape[1] * j:Bt_P.shape[1] * (j + 1)] = Bt_P * b[
            0, j - span + 2 * degree]
        B_t_P[len(t1_P) * i:len(t1_P) * (i + 1), Bt_P.shape[1] * j:Bt_P.shape[1] * (j + 1)] = Bt_d_P * b[
            0, j - span + 2 * degree]
        B_x_P[len(t1_P) * i:len(t1_P) * (i + 1), Bt_P.shape[1] * j:Bt_P.shape[1] * (j + 1)] = Bt_P * b[
            1, j - span + 2 * degree]
        B_2x_P[len(t1_P) * i:len(t1_P) * (i + 1), Bt_P.shape[1] * j:Bt_P.shape[1] * (j + 1)] = Bt_P * b[
            2, j - span + 2 * degree]

## Initialization
beta = torch.rand(B.shape[1], 1)
beta = torch.tensor(beta, dtype=torch.float64)

theta = torch.rand(1, 1)
theta = torch.tensor(theta, dtype=torch.float64)

beta.requires_grad_()
B = torch.from_numpy(B)
B_P = torch.from_numpy(B_P)
B_t_P = torch.from_numpy(B_t_P)
B_x_P = torch.from_numpy(B_x_P)
B_2x_P = torch.from_numpy(B_2x_P)

u = torch.rand(len(U), 1)
for i in range(0, len(U)):
    u[i] = U[i]
u = torch.tensor(u, dtype=torch.float64)

## Iteration of Successive Convex Approximation
maxiter = 300
mu = 500
a1 = 4000
c = 1000


# gamma = 1
# epsilon = 0.02
# fit_res = torch.rand(maxiter + 1, 1)
# pde_res = torch.rand(maxiter + 1, 1)
# res = torch.rand(maxiter + 1, 1)
def get_cuberoot(x):
    if x < 0:
        x = abs(x)
        cube_root = x ** (1 / 3) * (-1)
    else:
        cube_root = x ** (1 / 3)
    return cube_root


def exact_line_search(B, B_P, B_x_P, B_t_P, B_2x_P, u, Bbeta, beta, theta, mu, w1):
    nabla = (Bbeta.detach().numpy() - beta.detach().numpy())
    B1 = (B_P @ nabla) * (B_x_P @ nabla)
    B2 = B_t_P @ nabla + (B_P @ beta.detach().numpy()) * (B_x_P @ nabla) + (B_P @ nabla) * (
            B_x_P @ beta.detach().numpy()) - theta * B_2x_P @ nabla
    B3 = B_t_P @ beta.detach().numpy() + (B_P @ beta.detach().numpy()) * (
            B_x_P @ beta.detach().numpy()) - theta * B_2x_P @ beta.detach().numpy()
    g = 1 / 2 / w1 * (
            torch.norm(u - B @ Bbeta.detach().numpy()) * torch.norm(u - B @ Bbeta.detach().numpy()) - torch.norm(
        u - B @ beta.detach().numpy()) * torch.norm(u - B @ beta.detach().numpy()))
    mu = 0.5 * mu
    a = 4 * mu * B1.T @ B1
    b = 3 * mu * (B1.T @ B2 + B2.T @ B1)
    c = 2 * mu * (B1.T @ B3 + B2.T @ B2 + B3.T @ B1)
    d = mu * (B2.T @ B3 + B3.T @ B2) + g

    p = (3 * a * c - b ** 2) / (3 * a ** 2)
    q = (27 * a ** 2 * d - 9 * a * b * c + 2 * b ** 3) / (27 * a ** 3)

    D = (p / 3) ** 3 + (q / 2) ** 2

    if D > 0:
        flag = 1
        x = get_cuberoot(-q / 2 + np.sqrt(D)) + get_cuberoot(-q / 2 - np.sqrt(D))
        if x >= b / 3 / a and x <= 1 + b / 3 / a:
            gamma = x - b / 3 / a
        else:
            if q >= 0:
                gamma = 0
            else:
                gamma = 1
    elif D == 0:
        flag = 2
        x = np.zeros((1, 2))
        x[0, 0] = 2 * get_cuberoot(-q / 2)
        x[0, 1] = complex(-1, np.sqrt(3)) * get_cuberoot(-q / 2) + complex(-1, np.sqrt(3)) ** 2 * get_cuberoot(-q / 2)
        x = x[x >= b / 3 / a and x <= 1 + b / 3 / a]
        y = a / 4 * (x - b / 3 / a) ** 4 + b / 3 * (x - b / 3 / a) ** 3 + c / 2 * (x - b / 3 / a) ** 2 + d * (
                x - b / 3 / a)
        min_index = min(enumerate(y))
        gamma = x[0, min_index[0]] - b / 3 / a
    elif D < 0:
        flag = 3
        x = np.zeros((1, 3))
        x[0, 0] = get_cuberoot(-q / 2 + cmath.sqrt(D)) + get_cuberoot(-q / 2 - cmath.sqrt(D))
        x[0, 1] = complex(-1, np.sqrt(3)) * get_cuberoot(-q / 2 + cmath.sqrt(D)) + complex(-1, np.sqrt(
            3)) ** 2 * get_cuberoot(
            -q / 2 - cmath.sqrt(D))
        x[0, 2] = complex(-1, np.sqrt(3)) ** 2 * get_cuberoot(-q / 2 + cmath.sqrt(D)) + complex(-1, np.sqrt(
            3)) * get_cuberoot(
            -q / 2 - cmath.sqrt(D))
        x = x[x >= b / 3 / a and x <= 1 + b / 3 / a]
        y = a / 4 * (x - b / 3 / a) ** 4 + b / 3 * (x - b / 3 / a) ** 3 + c / 2 * (x - b / 3 / a) ** 2 + d * (
                x - b / 3 / a)
        min_index = min(enumerate(y))
        gamma = x[0, min_index[0]] - b / 3 / a

    print('Stepsize=%f, Real solution=%d' % (
        gamma, flag))
    return gamma


for i in range(0, maxiter + 1):

    if i == 0:
        # Compute the approximated solution of beta
        F = B_2x_P @ beta
        f = mu / 2 * torch.norm(B_t_P @ beta + (B_P @ beta) * (B_x_P @ beta) - theta * F)
        f_grads = autograd.grad(outputs=f, inputs=beta)[0]

        Bbeta = torch.inverse(a1 / w1 * B.T @ B + c * np.eye(B.shape[1])) @ (a1 / w1 * B.T @ u + c * beta - f_grads)
        gamma = exact_line_search(B, B_P, B_x_P, B_t_P, B_2x_P, u, Bbeta, beta, theta, mu, w1)
        # gamma = gamma * (1 - gamma * epsilon)
        beta = beta + gamma * (Bbeta - beta)

        ## Update theta
        with torch.no_grad():
            theta.copy_(torch.inverse(F.T @ F) @ F.T @ (B_t_P @ beta + (B_P @ beta) * (B_x_P @ beta)))

        # Print iteration and residual of each term
        fit_res = torch.norm(u - B @ beta)
        # pde_res = torch.norm(B_2x_P @ beta - theta * F)
        pde_res = torch.norm(B_t_P @ beta + (B_P @ beta) * (B_x_P @ beta) - theta * F)
        res = fit_res + pde_res
        print('Iteration=%d, Fitting residual=%f, PDE residual=%f, Total residual=%f' % (
            i, fit_res, pde_res, res))
        print(
            'Theta=' + str(theta.detach().numpy().T))
    else:
        # Compute the approximated solution of beta
        with torch.no_grad():
            F.copy_(B_2x_P @ beta)

        f = mu / 2 * torch.norm(B_t_P @ beta + (B_P @ beta) * (B_x_P @ beta) - theta * F)

        with torch.no_grad():
            f_grads.copy_(autograd.grad(outputs=f, inputs=beta)[0])
            Bbeta.copy_(
                torch.inverse(a1 / w1 * B.T @ B + c * np.eye(B.shape[1])) @ (a1 / w1 * B.T @ u + c * beta - f_grads))
        # gamma = gamma * (1 - gamma * epsilon)
        gamma = exact_line_search(B, B_P, B_x_P, B_t_P, B_2x_P, u, Bbeta, beta, theta, mu, w1)
        with torch.no_grad():
            beta += gamma * (Bbeta - beta)
            ## Update theta
            theta.copy_(torch.inverse(F.T @ F) @ F.T @ (B_t_P @ beta + (B_P @ beta) * (B_x_P @ beta)))
        # Print iteration and residual of each term
        fit_res = torch.norm(u - B @ beta)
        # pde_res = torch.norm(B_2x_P @ beta - theta * F)
        pde_res = torch.norm(B_t_P @ beta + (B_P @ beta) * (B_x_P @ beta) - theta * F)
        res = fit_res + pde_res
        if i % 1 == 0:
            print('Iteration=%d, Fitting residual=%f, PDE residual=%f, Total residual=%f' % (
                i, fit_res, pde_res, res))
            print(
                'Theta=' + str(theta.detach().numpy().T) + '\n')

# Plot figures to check results
# Plot fitting surface using B
X = x1.repeat(len(t1))
X = X.reshape(456,)
Y = np.tile(t1, (1, len(x1)))
Y=Y.reshape(456,)
Z = U.to_numpy()
Z = Z.reshape(456,)
fig = go.Figure(data=[go.Scatter3d(x=X,
                                   y=Y.T,
                                   z=Z,
                                   mode='markers')])
fig.show()
plotly.offline.plot(fig)

fig = plt.figure()
ax = plt.subplot(projection='3d')
ax.set_title('Fitting surface and data points')
ax.scatter(x1.repeat(len(t1)), np.tile(t1, (1, len(x1))), u, c='r')

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')

X, Y = np.meshgrid(x1, t1)
u1 = (B @ beta).detach().numpy()
U1 = u1.reshape(len(x1), len(t1))
ax.plot_surface(X, Y, U1.T, alpha=0.9, cstride=1, rstride=1, cmap='rainbow')
plt.show()

# Plot fitting surface using B_P
fig1 = plt.figure()
ax = plt.subplot(projection='3d')
ax.set_title('PDE surface and data points')
ax.scatter(x1.repeat(len(t1)), np.tile(t1, (len(x1), 1)), u, c='r')

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')

XP, YP = np.meshgrid(x1_P, t1_P)
u1P = (B_P @ beta).detach().numpy()
U1P = u1P.reshape(len(x1_P), len(t1_P))
ax.plot_surface(XP, YP, U1P.T, alpha=0.9, cstride=1, rstride=1, cmap='rainbow')
plt.show()

# Plot PDE residual surface
fig2 = plt.figure()
ax = plt.subplot(projection='3d')
ax.set_title('PDE surface and data points')
ax.scatter(x1.repeat(len(t1)), np.tile(t1, (len(x1), 1)), u, c='r')

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
XP, YP = np.meshgrid(x1_P, t1_P)
pdeRes = (F).detach().numpy()
PDEres = pdeRes.reshape(len(x1_P), len(t1_P))
ax.plot_surface(XP, YP, PDEres.T, alpha=0.9, cstride=1, rstride=1, cmap='rainbow')
plt.show()

# Plot PDE residual surface
fig3 = plt.figure()
ax = plt.subplot(projection='3d')
ax.set_title('PDE surface and data points')

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
XP, YP = np.meshgrid(x1_P, t1_P)
pdeRes = (B_t_P @ beta).detach().numpy()
PDEres = pdeRes.reshape(len(x1_P), len(t1_P))
ax.plot_surface(XP, YP, PDEres.T, alpha=0.9, cstride=1, rstride=1, cmap='rainbow')
plt.show()

# Plot PDE residual surface
fig4 = plt.figure()
ax = plt.subplot(projection='3d')
ax.set_title('PDE surface and data points')

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
XP, YP = np.meshgrid(x1_P, t1_P)
pdeRes = ((B_P @ beta) * (B_x_P @ beta)).detach().numpy()
PDEres = pdeRes.reshape(len(x1_P), len(t1_P))
ax.plot_surface(XP, YP, PDEres.T, alpha=0.9, cstride=1, rstride=1, cmap='rainbow')
plt.show()
