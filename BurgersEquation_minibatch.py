import sys
from torch import autograd
import torch
import numpy as np
import math
import pandas as pd
import cmath

filepath = 'D:\PhD study\simplines-main\simplines/'
sys.path.append(filepath)
import bsplines as bs
import plotly.graph_objects as go
import random
import plotly

## Load data
raw_data = pd.read_csv('D:\PhD study\simplines-main/burgers.csv')
U = raw_data['u']
w1 = np.var(U)

t0 = raw_data['t']
t1 = np.unique(t0)

x0 = raw_data['x']
x1 = np.unique(x0)

# Data visualization
fig = go.Figure(data=[go.Scatter3d(x=x1.repeat(len(t1)).reshape(len(U), ),
                                   y=np.tile(t1, (1, len(x1))).reshape(len(U), ),
                                   z=U.to_numpy().reshape(len(U), ),
                                   mode='markers')])
fig.show()
plotly.offline.plot(fig)

# Generate Bt for fitting
del_t = 0.01  # The interval of original data
T = 0.90 + del_t
degree_t = 2
del_tb = random.uniform(del_t / degree_t, del_t / (degree_t + 1))
knot_t = bs.make_knots(np.linspace(0, T - del_t, num=math.ceil(T / del_tb)), degree_t, False)
Bt = np.zeros((len(t1), len(knot_t) - 1 - degree_t - 2 * degree_t))

f_batch_size = 2 ** 12  # The number of points put into algorithm in each batch, so the size of B_P is fixed

Bt_P = np.zeros((f_batch_size, len(knot_t) - 1 - degree_t - 2 * degree_t))
for i in range(0, len(t1)):
    span = bs.find_span(knot_t, degree_t, t1[i])
    b = bs.basis_funs_all_ders(knot_t, degree_t, t1[i], span, 2)
    for j in range(0, degree_t + 1):
        Bt[i, span - 2 * degree_t + j] = b[0, j]

del_x = 0.05  # The distance between sensors
X = 1 + del_x  # The distance between the first and the last sensor
degree_x = 3
del_xb = random.uniform(del_x / degree_x, del_x / (degree_x + 1))
knot_x = bs.make_knots(np.linspace(-1, X - del_x, num=math.ceil((X + 1) / del_xb)), degree_x, False)

Bx = np.zeros((len(x1), len(knot_x) - 1 - degree_x - 2 * degree_x))
for i in range(0, len(x1)):
    span = bs.find_span(knot_x, degree_x, x1[i])
    b = bs.basis_funs_all_ders(knot_x, degree_x, x1[i], span, 2)
    for j in range(0, degree_x + 1):
        Bx[i, span - 2 * degree_x + j] = b[0, j]
Bx_P = np.zeros((f_batch_size, len(knot_x) - 1 - degree_x - 2 * degree_x))

B = np.zeros((len(x1) * len(t1), Bt.shape[1] * Bx.shape[1]))
for k in range(0, len(x0)):
    span_x = bs.find_span(knot_x, degree_x, x0[k])
    span_t = bs.find_span(knot_t, degree_t, t0[k])
    b_x = bs.basis_funs_all_ders(knot_x, degree_x, x0[k], span_x, 2)
    b_t = bs.basis_funs_all_ders(knot_t, degree_t, t0[k], span_t, 1)
    for m in range(0, degree_x + 1):
        for n in range(0, degree_t + 1):
            B[k, (span_x - 2 * degree_x + m) * Bt.shape[1] + span_t - 2 * degree_t + n] = b_x[0, m] * b_t[0, n]

# Generate point randomly for PDE constraints
x_min, x_max = -0.95, 0.95
t_min, t_max = 0.04, 0.86

num_points = 2 ** 20

x_coords = [random.uniform(x_min, x_max) for _ in range(num_points - 4 * int(np.sqrt(num_points)))]
t_coords = [random.uniform(t_min, t_max) for _ in range(num_points - 4 * int(np.sqrt(num_points)))]

x_min_bound = [x_min for i in range(0, int(np.sqrt(num_points)))]
x_max_bound = [x_max for i in range(0, int(np.sqrt(num_points)))]
t_min_bound = [random.uniform(t_min, t_max) for _ in range(int(np.sqrt(num_points)))]
t_max_bound = [random.uniform(t_min, t_max) for _ in range(int(np.sqrt(num_points)))]
x_coords.extend(x_min_bound)
x_coords.extend(x_max_bound)
t_coords.extend(t_min_bound)
t_coords.extend(t_max_bound)

x_min_bound = [random.uniform(x_min, x_max) for _ in range(int(np.sqrt(num_points)))]
x_max_bound = [random.uniform(x_min, x_max) for _ in range(int(np.sqrt(num_points)))]
t_min_bound = [t_min for i in range(0, int(np.sqrt(num_points)))]
t_max_bound = [t_max for i in range(0, int(np.sqrt(num_points)))]
x_coords.extend(x_min_bound)
x_coords.extend(x_max_bound)
t_coords.extend(t_min_bound)
t_coords.extend(t_max_bound)

random.shuffle(x_coords)
random.shuffle(t_coords)


# Define exact line search
def get_cuberoot(x):
    if x < 0:
        x = abs(x)
        cube_root = x ** (1 / 3) * (-1)
    else:
        cube_root = x ** (1 / 3)
    return cube_root


def exact_line_search(B, B_P, B_x_P, B_t_P, B_2x_P, u, Bbeta, beta, theta, mu, w1, a1):
    nabla = (Bbeta.detach().numpy() - beta.detach().numpy())
    B1 = (B_P @ nabla) * (B_x_P @ nabla)
    B2 = B_t_P @ nabla + (B_P @ beta.detach().numpy()) * (B_x_P @ nabla) + (B_P @ nabla) * (
            B_x_P @ beta.detach().numpy()) - theta * B_2x_P @ nabla
    B3 = B_t_P @ beta.detach().numpy() + (B_P @ beta.detach().numpy()) * (
            B_x_P @ beta.detach().numpy()) - theta * B_2x_P @ beta.detach().numpy()
    g = a1 / 2 / w1 * (
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
                gamma = torch.from_numpy(np.array(0.0))
            else:
                gamma = torch.from_numpy(np.array(1.0))
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

    return gamma, flag


# Operate SCA
n_epoch = 50  # The number of epoch
ind = [i for i in range(0, len(x_coords))]
beta = torch.rand(B.shape[1], 1)
beta = torch.tensor(beta, dtype=torch.float64)

theta = torch.rand(1, 1)
theta = torch.tensor(theta, dtype=torch.float64)

beta.requires_grad_()
B = torch.from_numpy(B)
u = torch.rand(len(U), 1)
mu = 200
a1 = 2000
c = 1000
for i in range(0, len(U)):
    u[i] = U[i]
u = torch.tensor(u, dtype=torch.float64)
# Begin
for i in range(0, n_epoch):
    random.shuffle(ind)
    for j in range(0, int(num_points / f_batch_size)):
        B_t_P = np.zeros((f_batch_size, Bt.shape[1] * Bx.shape[1]))
        B_x_P = np.zeros((f_batch_size, Bt.shape[1] * Bx.shape[1]))
        B_2x_P = np.zeros((f_batch_size, Bt.shape[1] * Bx.shape[1]))
        B_P = np.zeros((f_batch_size, Bt.shape[1] * Bx.shape[1]))

        for k in range(0 + f_batch_size * j, f_batch_size * (j + 1)):
            span_x = bs.find_span(knot_x, degree_x, x_coords[ind[k]])
            span_t = bs.find_span(knot_t, degree_t, t_coords[ind[k]])
            b_x = bs.basis_funs_all_ders(knot_x, degree_x, x_coords[ind[k]], span_x, 2)
            b_t = bs.basis_funs_all_ders(knot_t, degree_t, t_coords[ind[k]], span_t, 1)
            for m in range(0, degree_x + 1):
                for n in range(0, degree_t + 1):
                    B_P[k - f_batch_size * j, (span_x - 2 * degree_x + m) * Bt.shape[1] + span_t - 2 * degree_t + n] = \
                        b_x[0, m] * b_t[0, n]
                    B_t_P[k - f_batch_size * j, (span_x - 2 * degree_x + m) * Bt.shape[1] + span_t - 2 * degree_t + n] = \
                        b_x[0, m] * b_t[1, n]
                    B_x_P[k - f_batch_size * j, (span_x - 2 * degree_x + m) * Bt.shape[1] + span_t - 2 * degree_t + n] = \
                        b_x[1, m] * b_t[0, n]
                    B_2x_P[
                        k - f_batch_size * j, (span_x - 2 * degree_x + m) * Bt.shape[1] + span_t - 2 * degree_t + n] = \
                        b_x[2, m] * b_t[0, n]

        B_P = torch.from_numpy(B_P)
        B_t_P = torch.from_numpy(B_t_P)
        B_x_P = torch.from_numpy(B_x_P)
        B_2x_P = torch.from_numpy(B_2x_P)
        if j == 0:
            F = B_2x_P @ beta
            f = mu / 2 * torch.norm(B_t_P @ beta + (B_P @ beta) * (B_x_P @ beta) - theta * F)
            f_grads = autograd.grad(outputs=f, inputs=beta)[0]

            Bbeta = torch.inverse(a1 / w1 * B.T @ B + c * np.eye(B.shape[1])) @ (a1 / w1 * B.T @ u + c * beta - f_grads)
            gamma, flag = exact_line_search(B, B_P, B_x_P, B_t_P, B_2x_P, u, Bbeta, beta, theta, mu, w1, a1)
            beta = beta + gamma * (Bbeta - beta)

            ## Update theta
            with torch.no_grad():
                theta.copy_(torch.inverse(F.T @ F) @ F.T @ (B_t_P @ beta + (B_P @ beta) * (B_x_P @ beta)))
            # Print iteration and residual of each term
            fit_res = torch.norm(u - B @ beta)
            # pde_res = torch.norm(B_2x_P @ beta - theta * F)
            pde_res = torch.norm(B_t_P @ beta + (B_P @ beta) * (B_x_P @ beta) - theta * F)
            res = fit_res + pde_res

            print('\n' + 'Epoch | Batch | Stepsize | Flag | Fitting residual | PDE residual | Total residual | Theta')
            print(i, ' |', j, ' |', gamma.detach().numpy(), '|', flag, '| ', fit_res.detach().numpy(), ' | ',
                  pde_res.detach().numpy(), ' | ',
                  res.detach().numpy(), ' |', theta.detach().numpy())
        else:
            with torch.no_grad():
                F.copy_(B_2x_P @ beta)

            f = mu / 2 * torch.norm(B_t_P @ beta + (B_P @ beta) * (B_x_P @ beta) - theta * F)

            with torch.no_grad():
                f_grads.copy_(autograd.grad(outputs=f, inputs=beta)[0])
                Bbeta.copy_(
                    torch.inverse(a1 / w1 * B.T @ B + c * np.eye(B.shape[1])) @ (
                            a1 / w1 * B.T @ u + c * beta - f_grads))
            # gamma = gamma * (1 - gamma * epsilon)
            gamma, flag = exact_line_search(B, B_P, B_x_P, B_t_P, B_2x_P, u, Bbeta, beta, theta, mu, w1, a1)
            with torch.no_grad():
                beta += gamma * (Bbeta - beta)
                ## Update theta
                theta.copy_(torch.inverse(F.T @ F) @ F.T @ (B_t_P @ beta + (B_P @ beta) * (B_x_P @ beta)))
            # Print iteration and residual of each term
            fit_res = torch.norm(u - B @ beta)
            pde_res = torch.norm(B_t_P @ beta + (B_P @ beta) * (B_x_P @ beta) - theta * F)
            res = fit_res + pde_res
            if j % 1 == 0:
                print(i, ' |', j, ' |', gamma.detach().numpy(), '|', flag, '| ', fit_res.detach().numpy(),
                      ' | ',
                      pde_res.detach().numpy(), ' | ',
                      res.detach().numpy(), ' |', theta.detach().numpy())

# Preparation for visualization
Nx = 10
Nt = 10
t1_P = list(np.linspace(t1[0], t1[-1],
                        num=Nt * len(t1)))
x1_P = np.linspace(x1[0], x1[-1],
                   num=Nx * len(x1))

x_coords_d = [val for val in x1_P for i in range(len(t1_P))]
t_coords_d = t1_P * len(x1_P)
B_t_Pd = np.zeros((len(x_coords_d), Bt.shape[1] * Bx.shape[1]))
B_x_Pd = np.zeros((len(x_coords_d), Bt.shape[1] * Bx.shape[1]))
B_2x_Pd = np.zeros((len(x_coords_d), Bt.shape[1] * Bx.shape[1]))
B_Pd = np.zeros((len(x_coords_d), Bt.shape[1] * Bx.shape[1]))
for k in range(0, len(x_coords_d)):
    span_x = bs.find_span(knot_x, degree_x, x_coords_d[k])
    span_t = bs.find_span(knot_t, degree_t, t_coords_d[k])
    b_x = bs.basis_funs_all_ders(knot_x, degree_x, x_coords_d[k], span_x, 2)
    b_t = bs.basis_funs_all_ders(knot_t, degree_t, t_coords_d[k], span_t, 1)
    for m in range(0, degree_x + 1):
        for n in range(0, degree_t + 1):
            B_Pd[k, (span_x - 2 * degree_x + m) * Bt.shape[1] + span_t - 2 * degree_t + n] = \
                b_x[0, m] * b_t[0, n]
            B_t_Pd[k, (span_x - 2 * degree_x + m) * Bt.shape[1] + span_t - 2 * degree_t + n] = \
                b_x[0, m] * b_t[1, n]
            B_x_Pd[k, (span_x - 2 * degree_x + m) * Bt.shape[1] + span_t - 2 * degree_t + n] = \
                b_x[1, m] * b_t[0, n]
            B_2x_Pd[
                k, (span_x - 2 * degree_x + m) * Bt.shape[1] + span_t - 2 * degree_t + n] = \
                b_x[2, m] * b_t[0, n]

B_Pd = torch.from_numpy(B_Pd)
B_t_Pd = torch.from_numpy(B_t_Pd)
B_x_Pd = torch.from_numpy(B_x_Pd)
B_2x_Pd = torch.from_numpy(B_2x_Pd)

zz = (B_Pd @ beta).detach().numpy()
ZZ = np.zeros((len(t1_P), len(x1_P)))
for i in range(0, len(x1_P)):
    ZZ[:, i] = zz[i * len(t1_P):(i + 1) * len(t1_P), 0]

# Result visualization
xx, tt = np.meshgrid(x1_P, t1_P)
fig = go.Figure(data=[go.Surface(
    x=xx,
    y=tt,
    z=ZZ)])

fig.update_layout(
    scene=dict(
        xaxis_title='x',
        yaxis_title='t',
        zaxis_title='u',
        aspectratio=dict(x=0.5, y=1, z=0.5)
    ),
    width=800, height=800,
)

fig.add_scatter3d(x=x1.repeat(len(t1)).reshape(len(U), ), y=np.tile(t1, (1, len(x1))).reshape(len(U), ),
                  z=U.to_numpy().reshape(len(U), ), mode='markers',
                  marker=dict(size=2,
                              color="black",
                              # colorscale='b'
                              ))

fig.show()
plotly.offline.plot(fig)
## Plot B2x beta
zz1 = (B_2x_Pd @ beta).detach().numpy()
ZZ1 = np.zeros((len(t1_P), len(x1_P)))
for i in range(0, len(x1_P)):
    ZZ1[:, i] = zz1[i * len(t1_P):(i + 1) * len(t1_P), 0]
fig = go.Figure(data=[go.Surface(
    x=xx,
    y=tt,
    z=ZZ1)])

fig.update_layout(
    scene=dict(
        xaxis_title='x',
        yaxis_title='t',
        zaxis_title='u',
        aspectratio=dict(x=0.5, y=1, z=0.5)
    ),
    width=800, height=800,
)
fig.show()
plotly.offline.plot(fig)
## Plot Bt beta
zz2 = (B_t_Pd @ beta).detach().numpy()
ZZ2 = np.zeros((len(t1_P), len(x1_P)))
for i in range(0, len(x1_P)):
    ZZ2[:, i] = zz2[i * len(t1_P):(i + 1) * len(t1_P), 0]
fig = go.Figure(data=[go.Surface(
    x=xx,
    y=tt,
    z=ZZ2)])

fig.update_layout(
    scene=dict(
        xaxis_title='x',
        yaxis_title='t',
        zaxis_title='u',
        aspectratio=dict(x=0.5, y=1, z=0.5)
    ),
    width=800, height=800,
)

fig.show()
plotly.offline.plot(fig)
## Plot B beta*Bt beta
zz3 = (B_Pd @ beta * B_t_Pd @ beta).detach().numpy()
ZZ3 = np.zeros((len(t1_P), len(x1_P)))
for i in range(0, len(x1_P)):
    ZZ3[:, i] = zz3[i * len(t1_P):(i + 1) * len(t1_P), 0]
fig = go.Figure(data=[go.Surface(
    x=xx,
    y=tt,
    z=ZZ3)])

fig.update_layout(
    scene=dict(
        xaxis_title='x',
        yaxis_title='t',
        zaxis_title='u',
        aspectratio=dict(x=0.5, y=1, z=0.5)
    ),
    width=800, height=800,
)

fig.show()
plotly.offline.plot(fig)
