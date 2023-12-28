import sys
import matplotlib.pyplot as plt

filepath = 'D:\PhD study\simplines-main\simplines/'
sys.path.append(filepath)
import bsplines as bs
import torch
import numpy as np
import random
import math
import pandas as pd

## Generate Bt for fitting
del_t = 0.05  # The interval of original data
T = 5 + del_t
t = np.arange(0, T, del_t)  # If the sampling rate is not uniform, t is the true sampling rate
degree = 3
del_tb = random.uniform(del_t / degree, del_t / (degree + 1))
knot = bs.make_knots(np.linspace(0, T - del_t, num=math.ceil(T / del_tb)), degree, False)

t1 = t[1:len(t) - 1]  # Give up the first and the last data point of t to aovid clamped effect
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
t1_P = np.linspace(t1[0], t1[-1],
                   num=2 * len(t1))  # Give up the first and the last data point of t to aovid clamped effect
Bt_P = np.zeros((len(t1_P), len(knot) - 1 - degree - 2 * degree))
Bt_d_P = np.zeros((len(t1_P), len(knot) - 1 - degree - 2 * degree))
Bt_2d_P = np.zeros((len(t1_P), len(knot) - 1 - degree - 2 * degree))
for i in range(0, len(t1_P)):
    span = bs.find_span(knot, degree, t1_P[i])
    b = bs.basis_funs_all_ders(knot, degree, t1_P[i], span, 2)
    for j in range(0, degree + 1):
        Bt_P[i, span - 2 * degree + j] = b[0, j]
        Bt_d_P[i, span - 2 * degree + j] = b[1, j]
        Bt_2d_P[i, span - 2 * degree + j] = b[2, j]

## Genrate Bx for fitting
del_x = 0.1  # The distance between sensors
X = 1 + del_x  # The distance between the first and the last sensor
x = np.arange(0, X, del_x)
degree = 5
del_xb = random.uniform(del_x / degree, del_x / (degree + 1))
knot = bs.make_knots(np.linspace(0, X - del_x, num=math.ceil(X / del_xb)), degree, False)

x1 = x[1:len(x) - 1]  # If the sensors are arranged unevenly, x1 is the true distance
Bx = np.zeros((len(x1), len(knot) - 1 - degree - 2 * degree))

## Genrate Bx for PDE
x1_P = np.linspace(x1[0], x1[-1], num=2 * len(x1))  # If the sensors are arranged unevenly, x1 is the true distance
Bx_P = np.zeros((len(x1_P) * len(t1_P), len(knot) - 1 - degree - 2 * degree))
Bx_4d_P = np.zeros((len(x1_P) * len(t1_P), len(knot) - 1 - degree - 2 * degree))
for i in range(0, len(x1_P)):
    span = bs.find_span(knot, degree, x1_P[i])
    b = bs.basis_funs_all_ders(knot, degree, x1_P[i], span, 4)
    for j in range(0, degree + 1):
        Bx_P[i, span - 2 * degree + j] = b[0, j]
        Bx_4d_P[i, span - 2 * degree + j] = b[4, j]

## Generate two-dimensional basis function matrix B for fitting
B = np.zeros((len(x1) * len(t1), Bt.shape[1] * Bx.shape[1]))
for i in range(0, len(x1)):
    span = bs.find_span(knot, degree, x1[i]) - degree
    b = bs.basis_funs_all_ders(knot, degree, x1[i], span, 4)
    for j in range(span - degree, span + 1):
        B[len(t1) * i:len(t1) * (i + 1), Bt.shape[1] * j:Bt.shape[1] * (j + 1)] = Bt * b[0, j - span + degree]

## Generate two-dimensional basis function matrix B_P for PDE
B_4x_P = np.zeros((len(x1_P) * len(t1_P), Bt_P.shape[1] * Bx_P.shape[1]))
B_t_P = np.zeros((len(x1_P) * len(t1_P), Bt_P.shape[1] * Bx_P.shape[1]))
B_2t_P = np.zeros((len(x1_P) * len(t1_P), Bt_P.shape[1] * Bx_P.shape[1]))
for i in range(0, len(x1_P)):
    span = bs.find_span(knot, degree, x1_P[i]) - degree
    b = bs.basis_funs_all_ders(knot, degree, x1_P[i], span, 4)
    for j in range(span - degree, span + 1):
        B_4x_P[len(t1_P) * i:len(t1_P) * (i + 1), Bt_P.shape[1] * j:Bt_P.shape[1] * (j + 1)] = Bt_P * b[
            4, j - span + degree]
        B_t_P[len(t1_P) * i:len(t1_P) * (i + 1), Bt_P.shape[1] * j:Bt_P.shape[1] * (j + 1)] = Bt_d_P * b[
            0, j - span + degree]
        B_2t_P[len(t1_P) * i:len(t1_P) * (i + 1), Bt_P.shape[1] * j:Bt_P.shape[1] * (j + 1)] = Bt_2d_P * b[
            0, j - span + degree]

## Read data and initialization
raw_data = pd.read_csv('D:\PhD study\simplines-main\E-Bbeam.csv')
U = raw_data['u']
w1 = np.var(U)
beta = torch.rand(B.shape[1], 1)
beta = torch.tensor(beta, dtype=torch.float64)
theta1 = torch.rand(1, 1)
theta1 = torch.tensor(theta1, dtype=torch.float64)
theta2 = torch.rand(1, 1)
theta2 = torch.tensor(theta2, dtype=torch.float64)
theta3 = torch.rand(1, 1)
theta3 = torch.tensor(theta3, dtype=torch.float64)

B = torch.from_numpy(B)
B_4x_P = torch.from_numpy(B_4x_P)
B_2t_P = torch.from_numpy(B_2t_P)
B_t_P = torch.from_numpy(B_t_P)

u = torch.rand(len(U), 1)
for i in range(0, len(U)):
    u[i] = U[i]
u = torch.tensor(u, dtype=torch.float64)

## Iteration of Successive Convex Approximation
maxiter = 10000
mu = 1
a1 = 200
c = 1000
gamma = 1e-10
epsilon = 0.1
fit_res = torch.rand(maxiter + 1, 1)
ode_res = torch.rand(maxiter + 1, 1)
res = torch.rand(maxiter + 1, 1)

## Update beta
# F = theta1 * B_4x_P-B_2t_P*theta2-B_t_P*theta3
# beta = torch.inverse(1 / w1 * B.T @ B + mu * F.T @ F) @ (1 / w1 * B.T @ u)

for i in range(0, maxiter + 1):
    if i==0:
        f_grads = mu * (theta1 * B_4x_P - B_2t_P * theta2 - B_t_P * theta3).T @ (
                theta1 * B_4x_P - B_2t_P * theta2 - B_t_P * theta3) @ beta
        Bbeta = torch.inverse(1 / w1 * B.T @ B + c * np.eye(B.shape[1])) @ (1 / w1 * B.T @ u + c * beta - f_grads)
        gamma = gamma * (1 - gamma * epsilon)
        beta += gamma * (Bbeta - beta)
        ## Update theta1
        theta1 = torch.inverse((B_4x_P @ beta).T @ (B_4x_P @ beta)) @ (B_4x_P @ beta).T @ (
                B_2t_P @ beta * theta2 + B_t_P @ beta * theta3)

        ##Updata theta2
        theta2 = torch.inverse((B_2t_P @ beta).T @ (B_2t_P @ beta)) @ (B_2t_P @ beta).T @ (
                theta1 * B_4x_P @ beta - theta3 * B_t_P @ beta)

        ##Updata theta3
        theta3 = torch.inverse((B_t_P @ beta).T @ (B_t_P @ beta)) @ (B_t_P @ beta).T @ (
                theta1 * B_4x_P @ beta - theta2 * B_2t_P @ beta)
        fit_res[i] = torch.norm(u - B @ beta)
        ode_res[i] = torch.norm(theta1 * B_4x_P @ beta - theta2*B_2t_P@beta-theta3*B_t_P@beta)
        res[i] = fit_res[i] + ode_res[i]
        print('Iteration=%d, Fitting residual=%f, ODE residual=%f, Total residual=%f' % (
            i, fit_res[i], ode_res[i], res[i]))
        print(
            'Theta1=' + str(theta1.detach().numpy().T) + ',Theta2=' + str(
                theta2.detach().numpy().T) + ',Theta3=' + str(
                theta3.detach().numpy().T))
    else:
        ## Update beta
        # F.copy_(theta1 * B_4x_P - B_2t_P * theta2 - B_t_P * theta3)
        # beta.copy_(torch.inverse(1 / w1 * B.T @ B + mu * F.T @ F) @ (1 / w1 * B.T @ u))
        f_grads = mu * (theta1 * B_4x_P - B_2t_P * theta2 - B_t_P * theta3).T @ (
                 theta1 * B_4x_P - B_2t_P * theta2 - B_t_P * theta3) @ beta
        Bbeta = torch.inverse(1 / w1 * B.T @ B + c * np.eye(B.shape[1])) @ (1 / w1 * B.T @ u + c * beta - f_grads)
        gamma = gamma * (1 - gamma * epsilon)
        beta += gamma * (Bbeta - beta)

        ## Update theta1
        theta1.copy_(torch.inverse((B_4x_P @ beta).T @ (B_4x_P @ beta)) @ (B_4x_P @ beta).T @ (
            B_2t_P @ beta * theta2 + B_t_P @ beta * theta3))

        ##Updata theta2
        theta2.copy_(torch.inverse((B_2t_P @ beta).T @ (B_2t_P @ beta)) @ (B_2t_P @ beta).T @ (
            theta1 * B_4x_P @ beta - theta3 * B_t_P @ beta))

        ##Updata theta3
        theta3.copy_(torch.inverse((B_t_P @ beta).T @ (B_t_P @ beta)) @ (B_t_P @ beta).T @ (
            theta1 * B_4x_P @ beta - theta2 * B_2t_P @ beta))

        # Print iteration and residual of each term
        fit_res[i] = torch.norm(u - B @ beta)
        ode_res[i] = torch.norm(theta1 * B_4x_P @ beta - theta2*B_2t_P@beta-theta3*B_t_P@beta)
        res[i] = fit_res[i] + ode_res[i]
        if i % 1 == 0:
            print('Iteration=%d, Fitting residual=%f, ODE residual=%f, Total residual=%f' % (
            i, fit_res[i], ode_res[i], res[i]))
            print(
            'Theta1=' + str(theta1.detach().numpy().T) + ',Theta2=' + str(
                theta2.detach().numpy().T)+ ',Theta3=' + str(
                theta3.detach().numpy().T))