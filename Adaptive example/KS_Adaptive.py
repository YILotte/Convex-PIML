import AdapParaEst as ape
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
def plot_contour(X, T, U_pred, ax, title,m1,m2,cmap):
    norm = mpl.colors.Normalize(vmin=m1, vmax=m2)
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap=cmap,
                  extent=[T.min(), T.max(), X.min(), X.max()],
                  origin='lower', aspect='auto',norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar=plt.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=20)
    ax.tick_params(axis='x', length=3, width=1.5, colors='k', labelsize=20)
    ax.tick_params(axis='y', length=3, width=1.5, colors='k', labelsize=20)
    ax.set_xlabel('$t$',fontsize=30)
    ax.set_ylabel('$x$',fontsize=30)
    ax.set_title(title, fontsize=30)
un = pd.read_csv('../data/KS/noise_free_KS_in_col.csv')
thre = 0
raw_data_in = pd.read_csv('../data/KS/noise_free_KS_in_col.csv')
raw_data_ic = pd.read_csv('../data/KS/KS_ic.csv')
raw_data_bc = pd.read_csv('../data/KS/KS_bc.csv')
raw_data = pd.concat([raw_data_in,raw_data_ic, raw_data_bc],ignore_index=True)
Un = pd.concat([un,raw_data_ic, raw_data_bc],ignore_index=True)
U = Un['u']
u = np.random.rand(len(U), 1)
for i in range(0, len(U)):
    u[i] = U[i]
u = torch.tensor(u, dtype=torch.float64)
u=u.to('cuda')
x0 = np.array(raw_data['x']).reshape([len(u),1])
t0 = np.array(raw_data['t']).reshape([len(u),1])
x = np.unique(raw_data['x'])
t = np.unique(raw_data['t'])
X = np.hstack((x0, t0))
X_star = X
X_grid, T_grid = np.meshgrid(x, t)
degree = [20, 16]
T0 = np.min(raw_data_ic['t'])
T1 = np.max(raw_data['t'])
X0 = np.min(raw_data_bc['x'])
X1 = np.max(raw_data_bc['x'])

node_init = ape.knots_generate([[X0, X1], [T0, T1]], [50, 25], degree, 0,2)
# ——————————————————Optimize initial knots based on data fitting error distribution——————————————————#
B = ape.collocation_matrix_2d(node_init, degree, [0, 0],raw_data, 0)
device = torch.device("cuda")
B = torch.from_numpy(B).to(device)
R_fit,beta=ape.fit_data_KS(raw_data, B)

## Plot results
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
u_pred = (B @ beta).cpu().detach().numpy()
U_pred = griddata(X_star, u_pred.flatten(), (X_grid, T_grid), method='cubic')
fig, axs = plt.subplots(1, 1, figsize=(7, 6.3))
for i in range(degree[0], len(node_init['node_x'])):
    axs.axhline(y=node_init['node_x'][i], ls='-', xmin=-1, xmax=1, color='#808080', linewidth=1.2)
for i in range(degree[1], len(node_init['node_t'])):
    axs.axvline(x=node_init['node_t'][i], ls='-', ymin=0, ymax=1, color='#808080', linewidth=1.2)
plot_contour(X_grid, T_grid, U_pred, axs, title=r'$\hat{u}$',m1=min(u_pred), m2=max(u_pred),cmap='YlGnBu')
plt.show()

error = R_fit['residual']
Error = griddata(X_star, error.flatten(), (X_grid, T_grid), method='cubic')
fig, axs = plt.subplots(1, 1, figsize=(7, 6.3))
for i in range(degree[0], len(node_init['node_x'])):
    axs.axhline(y=node_init['node_x'][i], ls='-', xmin=-1, xmax=1, color='#808080', linewidth=1.2)
for i in range(degree[1], len(node_init['node_t'])):
    axs.axvline(x=node_init['node_t'][i], ls='-', ymin=0, ymax=1, color='#808080', linewidth=1.2)
plot_contour(X_grid, T_grid, Error, axs, title=r'data fitting error distribution', m1=min(error), m2=max(error),cmap='jet')
plt.show()

if np.mean(R_fit['residual']) > thre:
    node_new = ape.knot_refinement(R_fit, node_init, [2, 1], degree, 0)
flag=0

maxiter = 10
while flag < maxiter:
    flag += 1
    print('Iteration = %d' % (flag))
    B = ape.collocation_matrix_2d(node_new, degree, [0, 0], raw_data, 0)
    B = torch.from_numpy(B).to(device)
    R_fit,beta=ape.fit_data_KS(raw_data, B)

    ##画估计结果图
    u_pred = (B @ beta).cpu().detach().numpy()
    U_pred = griddata(X_star, u_pred.flatten(), (X_grid, T_grid), method='cubic')
    fig, axs = plt.subplots(1, 1, figsize=(7, 6.3))
    for i in range(degree[0], len(node_new['node_x'])):
        axs.axhline(y=node_new['node_x'][i], ls='-', xmin=-1, xmax=1, color='#808080', linewidth=1.2)
    for i in range(degree[1], len(node_new['node_t'])):
        axs.axvline(x=node_new['node_t'][i], ls='-', ymin=0, ymax=1, color='#808080', linewidth=1.2)
    plot_contour(X_grid, T_grid, U_pred, axs, title=r'$\hat{u}$',m1=min(u_pred), m2=max(u_pred),cmap='YlGnBu')
    plt.show()

    error = R_fit['residual']
    Error = griddata(X_star,error.flatten(), (X_grid, T_grid), method='cubic')
    fig, axs = plt.subplots(1, 1, figsize=(7, 6.3))
    for i in range(degree[0], len(node_new['node_x'])):
        axs.axhline(y=node_new['node_x'][i], ls='-', xmin=-1, xmax=1, color='#808080', linewidth=1.2)
    for i in range(degree[1], len(node_new['node_t'])):
        axs.axvline(x=node_new['node_t'][i], ls='-', ymin=0, ymax=1, color='#808080', linewidth=1.2)
    plot_contour(X_grid, T_grid, Error, axs, title=r'data fitting error distribution',m1=min(error), m2=max(error),cmap='jet')
    plt.show()

    if flag >= maxiter:
        break
    else:
        if np.mean(R_fit['residual']) > thre:
            node_new = ape.knot_refinement(R_fit, node_new, [2, 1], degree, 0)
node_init=node_new
# ——————————————————Further optimize knots based on data fitting and physics error distributions——————————————————#
B = ape.collocation_matrix_2d(node_init, degree, [0, 0], raw_data, 0)
B_t = ape.collocation_matrix_2d(node_init, degree, [0, 1], raw_data, 0)
B_2t = ape.collocation_matrix_2d(node_init, degree, [0, 2], raw_data, 0)
B_x = ape.collocation_matrix_2d(node_init, degree, [1, 0], raw_data, 0)
B_2x = ape.collocation_matrix_2d(node_init, degree, [2, 0], raw_data, 0)
B_4x = ape.collocation_matrix_2d(node_init, degree, [4, 0], raw_data, 0)
B_ic = ape.collocation_matrix_2d(node_init, degree, [0, 0],raw_data_ic, 0)
B_bc = ape.collocation_matrix_2d(node_init, degree, [0, 0],raw_data_bc, 0)

B = torch.from_numpy(B)
B_t = torch.from_numpy(B_t)
B_2t = torch.from_numpy(B_2t)
B_x = torch.from_numpy(B_x)
B_2x = torch.from_numpy(B_2x)
B_4x = torch.from_numpy(B_4x)
B_ic = torch.from_numpy(B_ic)
B_bc = torch.from_numpy(B_bc)

device = torch.device("cuda")
B,B_t,B_2t,B_x,B_2x,B_4x,B_ic, B_bc=B.to(device),B_t.to(device),B_2t.to(device),B_x.to(device),B_2x.to(device),B_4x.to(device),B_ic.to(device),B_bc.to(device)
flag = 0
a1 = torch.tensor(3, dtype=torch.float64)
a2 = torch.tensor(5, dtype=torch.float64)
a3 = torch.tensor(5, dtype=torch.float64)
beta, theta, R_init, R_fit_init, R_phy_init, res, fit_res, pde_res, Residual_initial,Para = ape.sca_KS(raw_data,raw_data_ic,raw_data_bc, B, B_t, B_x, B_2x, B_4x,B_ic,B_bc,a1,a2,a3)

error = R_fit_init['residual']
Error = griddata(X_star, error.flatten(), (X_grid, T_grid), method='cubic')
fig, axs = plt.subplots(1, 1, figsize=(8.2, 7.5))
for i in range(degree[0], len(node_init['node_x'])):
    axs.axhline(y=node_init['node_x'][i], ls='-', xmin=-1, xmax=1, color='#808080', linewidth=1.2)
for i in range(degree[1], len(node_init['node_t'])):
    axs.axvline(x=node_init['node_t'][i], ls='-', ymin=0, ymax=1, color='#808080', linewidth=1.2)
plot_contour(X_grid, T_grid, Error, axs, title=r'data fitting error distribution', m1=min(error), m2=max(error),
             cmap='jet')
plt.show()

error = R_phy_init['residual']
Error = griddata(X_star, error.flatten(), (X_grid, T_grid), method='cubic')
fig, axs = plt.subplots(1, 1, figsize=(8.2, 7.5))
for i in range(degree[0], len(node_init['node_x'])):
    axs.axhline(y=node_init['node_x'][i], ls='-', xmin=-1, xmax=1, color='#808080', linewidth=1.2)
for i in range(degree[1], len(node_init['node_t'])):
    axs.axvline(x=node_init['node_t'][i], ls='-', ymin=0, ymax=1, color='#808080', linewidth=1.2)
plot_contour(X_grid, T_grid, Error, axs, title=r'physics error distribution', m1=min(error), m2=max(error),
             cmap='jet')
plt.show()


curv = pd.DataFrame({'x': np.reshape(x0,[len(u), ]), 't': np.reshape(t0,[len(u), ]), 'curvature_x': abs(np.reshape((B_2x @ beta).detach().cpu().numpy(), [len(x0), ])),
                     'curvature_t': abs(np.reshape((B_2t @ beta).detach().cpu().numpy(), [len(t0), ]))})
Res = np.array([res.detach().cpu().numpy(), fit_res.detach().cpu().numpy(), pde_res.detach().cpu().numpy()]).reshape([1, 3])

# knot refinement

if np.mean(R_fit_init['residual']) > thre:
    node_new = ape.knot_refinement(R_fit_init, node_init, [1, 1], degree, 0)
node_new = ape.knot_refinement(R_phy_init, node_new, [3, 1], degree, 0)
# Plot the results

maxiter = 3
loss = Residual_initial
while flag < maxiter:
    # Position optimization
    node_po = ape.position_optimization(node_new, curv, [1e-5, 1e-5], degree, 0)
    flag += 1

    B = ape.collocation_matrix_2d(node_po, degree, [0, 0], raw_data, 0)
    B_t = ape.collocation_matrix_2d(node_po, degree, [0, 1], raw_data, 0)
    B_2t = ape.collocation_matrix_2d(node_po, degree, [0, 2], raw_data, 0)
    B_x = ape.collocation_matrix_2d(node_po, degree, [1, 0], raw_data, 0)
    B_2x = ape.collocation_matrix_2d(node_po, degree, [2, 0], raw_data, 0)
    B_4x = ape.collocation_matrix_2d(node_po, degree, [4, 0], raw_data, 0)
    B_ic = ape.collocation_matrix_2d(node_po, degree, [0, 0], raw_data_ic, 0)
    B_bc = ape.collocation_matrix_2d(node_po, degree, [0, 0], raw_data_bc, 0)

    B = torch.from_numpy(B)
    B_t = torch.from_numpy(B_t)
    B_2t = torch.from_numpy(B_2t)
    B_x = torch.from_numpy(B_x)
    B_2x = torch.from_numpy(B_2x)
    B_4x = torch.from_numpy(B_4x)
    B_ic = torch.from_numpy(B_ic)
    B_bc = torch.from_numpy(B_bc)

    device = torch.device("cuda")
    B, B_t, B_2t, B_x, B_2x, B_4x, B_ic, B_bc = B.to(device), B_t.to(device), B_2t.to(device), B_x.to(device), B_2x.to(
        device), B_4x.to(device), B_ic.to(device), B_bc.to(device)

    print('Iteration = %d' % (flag))
    beta, theta, R, R_fit, R_phy, res, fit_res, pde_res, Residual,para = ape.sca_KS(raw_data,raw_data_ic,raw_data_bc, B, B_t, B_x, B_2x, B_4x,B_ic,B_bc,a1,a2,a3)
    curv = pd.DataFrame({'x': np.reshape(x0, [len(u), ]), 't': np.reshape(t0, [len(u), ]),
                         'curvature_x': abs(np.reshape((B_2x @ beta).detach().cpu().numpy(), [len(x0), ])),
                         'curvature_t': abs(np.reshape((B_2t @ beta).detach().cpu().numpy(), [len(t0), ]))})
    a = np.array([res.detach().cpu().numpy(), fit_res.detach().cpu().numpy(), pde_res.detach().cpu().numpy()]).reshape([1, 3])
    Res = np.concatenate((Res, a), axis=0)

    loss = np.concatenate((loss, Residual), axis=1)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    axes.plot(np.log10(a1*Residual[:,0]+a2*Residual[:,1]+a3*Residual[:,2]+Residual[:,3]), c='k', linewidth=2, zorder=2,label=r'Loss curve for $5^{th}$ FAS iteration',)
    axes.set_xlabel(r'iteration of BSCA')
    axes.set_ylabel(r'$log_{10}$loss')
    axes.xaxis.label.set_size(40)
    axes.yaxis.label.set_size(40)
    axes.tick_params(axis='x', direction='in', length=3, width=1.5, colors='k', labelsize=30)
    axes.tick_params(axis='y', direction='in', length=3, width=1.5, colors='k', labelsize=30)
    # axes.set_ylim(np.max(Residual) - 2, np.max(Residual) + 5)
    axes.set_xlim(-4, len(Residual) + 2)
    axes.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    plt.rcParams.update({'font.size': 25})
    axes.legend(loc="best", fontsize=40)
    plt.show()

    error = R_fit['residual']
    Error = griddata(X_star, error.flatten(), (X_grid, T_grid), method='cubic')
    fig, axs = plt.subplots(1, 1, figsize=(8.2, 7.5))
    for i in range(degree[0], len(node_new['node_x'])):
        axs.axhline(y=node_new['node_x'][i], ls='-', xmin=-1, xmax=1, color='#808080', linewidth=1.2)
    for i in range(degree[1], len(node_new['node_t'])):
        axs.axvline(x=node_new['node_t'][i], ls='-', ymin=0, ymax=1, color='#808080', linewidth=1.2)
    plot_contour(X_grid, T_grid, Error, axs, title=r'data fitting error distribution', m1=min(error), m2=max(error),
                 cmap='jet')
    plt.show()

    error = R_phy['residual']
    Error = griddata(X_star, error.flatten(), (X_grid, T_grid), method='cubic')
    fig, axs = plt.subplots(1, 1, figsize=(8.2, 7.5))
    for i in range(degree[0], len(node_new['node_x'])):
        axs.axhline(y=node_new['node_x'][i], ls='-', xmin=-1, xmax=1, color='#808080', linewidth=1.2)
    for i in range(degree[1], len(node_init['node_t'])):
        axs.axvline(x=node_new['node_t'][i], ls='-', ymin=0, ymax=1, color='#808080', linewidth=1.2)
    plot_contour(X_grid, T_grid, Error, axs, title=r'physics error distribution', m1=min(error), m2=max(error),
                 cmap='jet')
    plt.show()

    if flag > maxiter:
        break
    else:
        # Knot refinement
        if np.mean(R_fit['residual']) > thre:
            node_new = ape.knot_refinement(R_fit, node_po, [1, 1], degree, 0)
        node_new = ape.knot_refinement(R_phy, node_new, [3, 1], degree, 0)