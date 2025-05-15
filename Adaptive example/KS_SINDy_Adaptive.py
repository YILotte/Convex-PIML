import AdapParaEst as ape
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

thre = 0
raw_data_in = pd.read_csv('../data/KS/20%noise_KS_in_col.csv')
raw_data_ic = pd.read_csv('../data/KS/KS_ic.csv')
raw_data_bc = pd.read_csv('../data/KS/KS_bc.csv')
raw_data = pd.concat([raw_data_in,raw_data_ic, raw_data_bc],ignore_index=True)
U = raw_data['u']
u = np.random.rand(len(U), 1)
for i in range(0, len(U)):
    u[i] = U[i]
u = torch.tensor(u, dtype=torch.float64)
x0 = raw_data['x']
t0 = raw_data['t']
x = np.unique(raw_data['x'])
t = np.unique(raw_data['t'])
xx, tt = raw_data['x'],raw_data['t']
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

##Plot results
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, axes = plt.subplots(1, 1, figsize=(12, 10))
zz5 = (B @ beta).detach().cpu().numpy()
cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=np.min(zz5), vmax=np.max(zz5))
im = axes.tricontourf(xx, tt, zz5.reshape([len(zz5),]), cmap=cmap, norm=norm)
cbar = fig.colorbar(im, ax=axes)
cbar.ax.tick_params(labelsize=30)
for i in range(degree[1], len(node_init['node_t'])):
    axes.axhline(y=node_init['node_t'][i], ls='-', xmin=-1, xmax=1, color='#808080', linewidth=1.2)
for i in range(degree[0], len(node_init['node_x'])):
    axes.axvline(x=node_init['node_x'][i], ls='-', ymin=0, ymax=1, color='#808080', linewidth=1.2)
axes.set_xlabel(r'\textit{x}')
axes.set_ylabel(r'\textit{t}')
axes.xaxis.label.set_size(35)
axes.yaxis.label.set_size(35)
axes.tick_params(axis='x', direction='in', length=3, width=1.5, colors='k', labelsize=30)
axes.tick_params(axis='y', direction='in', length=3, width=1.5, colors='k', labelsize=30)
axes.set_xlim(np.min(raw_data['x']), np.max(raw_data['x']))
axes.set_ylim(np.min(raw_data['t']), np.max(raw_data['t']))
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
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, axes = plt.subplots(1, 1, figsize=(12, 10))
    zz5 = (B @ beta).detach().cpu().numpy()
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=np.min(zz5), vmax=np.max(zz5))
    im = axes.tricontourf(xx, tt, zz5.reshape([len(zz5), ]), cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, ax=axes)
    cbar.ax.tick_params(labelsize=30)
    for i in range(degree[1], len(node_new['node_t'])):
        axes.axhline(y=node_new['node_t'][i], ls='-', xmin=-1, xmax=1, color='#808080', linewidth=1.2)
    for i in range(degree[0], len(node_init['node_x'])):
        axes.axvline(x=node_new['node_x'][i], ls='-', ymin=0, ymax=1, color='#808080', linewidth=1.2)
    axes.set_xlabel(r'\textit{x}')
    axes.set_ylabel(r'\textit{t}')
    axes.xaxis.label.set_size(35)
    axes.yaxis.label.set_size(35)
    axes.tick_params(axis='x', direction='in', length=3, width=1.5, colors='k', labelsize=30)
    axes.tick_params(axis='y', direction='in', length=3, width=1.5, colors='k', labelsize=30)
    axes.set_xlim(np.min(raw_data['x']), np.max(raw_data['x']))
    axes.set_ylim(np.min(raw_data['t']), np.max(raw_data['t']))
    plt.show()
    if flag >= maxiter:
        break
    else:
        if np.mean(R_fit['residual']) > thre:
            node_new = ape.knot_refinement(R_fit, node_new, [2, 1], degree, 0)

# ——————————————————Discover equation with SINDy——————————————————#

B = ape.collocation_matrix_2d(node_new, degree, [0, 0], raw_data, 0)
B_t = ape.collocation_matrix_2d(node_new, degree, [0, 1], raw_data, 0)
B_2t = ape.collocation_matrix_2d(node_new, degree, [0, 2], raw_data, 0)
B_x = ape.collocation_matrix_2d(node_new, degree, [1, 0], raw_data, 0)
B_2x = ape.collocation_matrix_2d(node_new, degree, [2, 0], raw_data, 0)
B_3x = ape.collocation_matrix_2d(node_new, degree, [3, 0], raw_data, 0)
B_4x = ape.collocation_matrix_2d(node_new, degree, [4, 0], raw_data, 0)
B_5x = ape.collocation_matrix_2d(node_new, degree, [5, 0], raw_data, 0)
B_ic = ape.collocation_matrix_2d(node_new, degree, [0, 0],raw_data_ic, 0)
B_bc = ape.collocation_matrix_2d(node_new, degree, [0, 0],raw_data_bc, 0)

B = torch.from_numpy(B)
B_t = torch.from_numpy(B_t)
B_2t = torch.from_numpy(B_2t)
B_x = torch.from_numpy(B_x)
B_2x = torch.from_numpy(B_2x)
B_3x = torch.from_numpy(B_3x)
B_4x = torch.from_numpy(B_4x)
B_5x = torch.from_numpy(B_5x)
B_ic = torch.from_numpy(B_ic)
B_bc = torch.from_numpy(B_bc)
B,B_t,B_2t,B_x,B_2x,B_3x,B_4x,B_5x,B_ic, B_bc=B.to(device),B_t.to(device),B_2t.to(device),B_x.to(device),B_2x.to(device),B_3x.to(device),B_4x.to(device),B_5x.to(device),B_ic.to(device),B_bc.to(device)

a1 = torch.tensor(3, dtype=torch.float64)
a2 = torch.tensor(5, dtype=torch.float64)
a3 = torch.tensor(5, dtype=torch.float64)

mu = torch.tensor(5, dtype=torch.float64)
threshold = 0.2
maxiter_bsca = 100
epsilon = 0.6

beta, Lambda, R_fit, R_phy, fit_res, pde_res, Residual, Phi, index, Para = ape.SCA_KS_SINDy(raw_data,raw_data_ic,raw_data_bc,B , B_t, B_x, B_2x, B_3x, B_4x,B_5x,B_ic,B_bc,a1,a2,a3, mu, threshold,maxiter_bsca,epsilon)

loss=Residual
Parameter=Para

##Plot results
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, axes = plt.subplots(1, 1, figsize=(12, 8))
# axes.xaxis.grid(True)
# axes.yaxis.grid(True)
axes.plot(np.log10(a1*Residual[:,0]+a2*Residual[:,1]+a3*Residual[:,2]+Residual[:,3]), c='k', linewidth=2, zorder=2, label='Loss curve before using FAS')
axes.set_xlabel(r'iteration of BSCA')
axes.set_ylabel(r'$log_{10}$loss')
axes.xaxis.label.set_size(40)
axes.yaxis.label.set_size(40)
axes.tick_params(axis='x', direction='in', length=3, width=1.5, colors='k', labelsize=25)
axes.tick_params(axis='y', direction='in', length=3, width=1.5, colors='k', labelsize=25)
# axes.set_ylim(np.max(Residual) - 2, np.max(Residual) + 5)
axes.set_xlim(-4, len(Residual) + 2)
axes.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
axes.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
plt.rcParams.update({'font.size': 25})
axes.legend(loc="best", fontsize=40)
axes.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
plt.show()

# ——————————————————Further optimize knots based on data fitting and physics error distributions——————————————————#

if np.mean(R_fit['residual']) > thre:
    node_new = ape.knot_refinement(R_fit, node_new, [1, 1], degree, 0)
node_new = ape.knot_refinement(R_phy, node_new, [3, 1], degree, 0)
curv = pd.DataFrame(
    {'x': x0, 't': t0, 'curvature_x': abs(np.reshape((B_2x @ beta).cpu().detach().numpy(), [len(x0), ])),
     'curvature_t': abs(np.reshape((B_2t @ beta).cpu().detach().numpy(), [len(t0), ]))})
flag=0
maxiter = 3
while flag < maxiter:
    print('Iteration = %d' % (flag))
    flag += 1
    node_po = ape.position_optimization(node_new, curv, [1e-5, 1e-5], degree, 0)
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

    beta, theta, R, R_fit, R_phy, res, fit_res, pde_res, Res,para = ape.sca_KS(raw_data, raw_data_ic, raw_data_bc, B,
                                                                               B_t, B_x, B_2x, B_4x, B_ic, B_bc, a1, a2,
                                                                               a3)
    zero_column = np.zeros((len(para), 32))
    para = np.hstack((para,zero_column))
    Para = np.concatenate((Para, para), axis=0)

    total_res =( a1*Res[:,0] +a2*Res[:,1] +a3*Res[:,2] + Res[:,3]).reshape([len(Res),1])
    Res = np.hstack((Res,total_res))
    Residual=np.concatenate((Residual, Res), axis=0)
    curv = pd.DataFrame(
        {'x': x0, 't': t0, 'curvature_x': abs(np.reshape((B_2x @ beta).detach().cpu().numpy(), [len(x0), ])),
         'curvature_t': abs(np.reshape((B_2t @ beta).detach().cpu().numpy(), [len(t0), ]))})
    if flag >= maxiter:
        break
    else:
        # Knot refinement
        if np.mean(R_fit['residual']) > thre:
            node_new = ape.knot_refinement(R_fit, node_po, [1, 1], degree, 0)
        node_new = ape.knot_refinement(R_phy, node_new, [3, 1], degree, 0)