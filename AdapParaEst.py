from sklearn import preprocessing
import fast_proximal as fp
import numpy as np
import bsplines as bs
import torch
from torch import autograd
import math

__all__ = ['knots_generate',
           'collocation_matrix',
           'collocation_matrix_2d',
           'collocation_matrix_3d_M',
           'collocation_matrix_3d_S',
           'position_optimization',
           'knot_refinement',
           'exact_line_search_KS',
           'solve',
           'findF',
           'findG',
           'findH',
           'sca_KS',
           'SCA_KS_SINDy',
           'library_KS',
           'STLASSO_KS',
           'exact_line_search_KS_SINDy',
           'sca_2D_NS',
           'sca_2D_NS_SINDy',
           'exact_line_search_NS_SINDy',
           'library_NS',
           'fit_data_NS',
           'fit_data_KS',
           'fit_data_LNN']


# ==============================================================================
def knots_generate(D, N, degree, form, dimension):
    """
        Generating knots.

    Parameters
    ----------
    D: list.
        Domain range.
    N : list[int]
        The number of knots.
    degree: list[int]
        The degree of B-splines.
    form: int.
        0 for opened B-spline
        1 for clamped B-spline
    Returns
    -------
    node: Dictionary.
        Knot vector.
    """
    if dimension == 1:
        if form == 0:
            T0 = D[0]
            T1 = D[1]
            node = np.linspace(T0, T1, num=N)
            node = np.insert(node, 0, [T0 - (T1 - T0) / N * (degree + 1 - i) for i in range(1, degree + 1)])
            node = np.append(node, [T1 + (T1 - T0) / N * i for i in range(1, degree + 1)])
            node = {'node_t': node}
        else:
            T0 = D[0]
            T1 = D[1]
            node = {'node_t': np.linspace(T0, T1, num=N)}
    elif dimension == 2:
        if form == 0:
            T0 = D[1][0]
            T1 = D[1][1]
            X0 = D[0][0]
            X1 = D[0][1]
            node_x = np.linspace(X0, X1, num=N[0])
            node_x = np.insert(node_x, 0,
                               [X0 - (X1 - X0) / N[0] * (degree[0] + 1 - i) for i in range(1, degree[0] + 1)])
            node_x = np.append(node_x, [X1 + (X1 - X0) / N[0] * i for i in range(1, degree[0] + 1)])
            node_t = np.linspace(T0, T1, num=N[1])
            node_t = np.insert(node_t, 0,
                               [T0 - (T1 - T0) / N[1] * (degree[1] + 1 - i) for i in range(1, degree[1] + 1)])
            node_t = np.append(node_t, [T1 + (T1 - T0) / N[1] * i for i in range(1, degree[1] + 1)])
            node = {'node_x': node_x,
                    'node_t': node_t}
        else:
            T0 = D[1][0]
            T1 = D[1][1]
            X0 = D[0][0]
            X1 = D[0][1]
            node = {'node_x': np.linspace(X0, X1, num=N[0]),
                    'node_t': np.linspace(T0, T1, num=N[1])}

    elif dimension == 3:
        if form == 0:
            X0 = D[0][0]
            X1 = D[0][1]
            Z0 = D[1][0]
            Z1 = D[1][1]
            T0 = D[2][0]
            T1 = D[2][1]
            node_x = np.linspace(X0, X1, num=N[0])
            node_x = np.insert(node_x, 0,
                               [X0 - (X1 - X0) / N[0] * (degree[0] + 1 - i) for i in range(1, degree[0] + 1)])
            node_x = np.append(node_x, [X1 + (X1 - X0) / N[0] * i for i in range(1, degree[0] + 1)])

            node_z = np.linspace(Z0, Z1, num=N[1])
            node_z = np.insert(node_z, 0,
                               [Z0 - (Z1 - Z0) / N[1] * (degree[1] + 1 - i) for i in range(1, degree[1] + 1)])
            node_z = np.append(node_z, [Z1 + (Z1 - Z0) / N[1] * i for i in range(1, degree[1] + 1)])

            node_t = np.linspace(T0, T1, num=N[2])
            node_t = np.insert(node_t, 0,
                               [T0 - (T1 - T0) / N[2] * (degree[2] + 1 - i) for i in range(1, degree[2] + 1)])
            node_t = np.append(node_t, [T1 + (T1 - T0) / N[2] * i for i in range(1, degree[2] + 1)])
            node = {'node_x': node_x,
                    'node_z': node_z,
                    'node_t': node_t}
        else:
            X0 = D[0][0]
            X1 = D[0][1]
            Z0 = D[1][0]
            Z1 = D[1][1]
            T0 = D[2][0]
            T1 = D[2][1]
            node = {'node_x': np.linspace(X0, X1, num=N[0]),
                    'node_z': np.linspace(Z0, Z1, num=N[1]),
                    'node_t': np.linspace(T0, T1, num=N[2])}
    return node


def collocation_matrix(node, degree, n, t, form):
    """
    Generating collocation matrix according to node, data and degree and the order of basis function for 1-D problem.

    Parameters
    ----------
    node : float.
          The coordinate of knot.
    degree : int
        The degree of B-spline basis function.
    n: int
        The order of derivative

    t: float
        The coordinates of data points

    form: int.
          0 for opened B-spline
          1 for clamped B-spline
    Returns
    -------
    B: ndarray.
          collocation matrix.
    """
    knot = bs.make_knots(node, degree, False)
    B = np.zeros((len(t), len(knot) - degree - 1))
    for i in range(0, len(t)):
        span = bs.find_span(knot, degree, t[i])
        b = bs.basis_funs_all_ders(knot, degree, t[i], span, n)
        for j in range(0, degree + 1):
            B[i, span - degree + j] = b[n, j]
    if form == 0:
        # B = np.delete(B, range(0, degree), 1)
        # B = np.delete(B, range(-degree, 0), 1)
        B = np.zeros((len(t), len(knot) - 3 * degree - 1))
        for i in range(0, len(t)):
            span = bs.find_span(knot, degree, t[i])
            b = bs.basis_funs_all_ders(knot, degree, t[i], span, n)
            for j in range(0, degree + 1):
                if span - 2 * degree + j < B.shape[1]:
                    B[i, span - 2 * degree + j] = b[n, j]
    return B


def collocation_matrix_2d(node, degree, N, raw_data, form):
    """
    Generating collocation matrix in 2-D case according to node, data and degree and the order of basis function for 2-D problem.

    Parameters
    ----------
    node : float.
          The coordinate of knot.
    degree : list[int]
        The degree of B-spline basis function.
    N: list[int]
        The order of derivative.
    raw_data: DataFrame
        Raw data.
    form: int.
          0 for opened B-spline.
          1 for clamped B-spline.
    Returns
    -------
    B: ndarray.
          collocation matrix.
    """
    node_x = node['node_x']
    node_t = node['node_t']
    degree_x = degree[0]
    degree_t = degree[1]
    knot_x = bs.make_knots(node_x, degree_x, False)
    knot_t = bs.make_knots(node_t, degree_t, False)
    x0 = raw_data['x']
    t0 = raw_data['t']
    if form != 0:
        B = np.zeros((len(x0), (len(knot_t) - 1 - degree_t) * (len(knot_x) - 1 - degree_x)))
        for k in range(0, len(x0)):
            span_x = bs.find_span(knot_x, degree_x, x0[k])
            span_t = bs.find_span(knot_t, degree_t, t0[k])
            b_x = bs.basis_funs_all_ders(knot_x, degree_x, x0[k], span_x, N[0])
            b_t = bs.basis_funs_all_ders(knot_t, degree_t, t0[k], span_t, N[1])
            for m in range(0, degree_x + 1):
                for n in range(0, degree_t + 1):
                    B[k, (span_x - degree_x + m) * (len(knot_t) - 1 - degree_t) + span_t - degree_t + n] = \
                        b_x[N[0], m] * b_t[N[1], n]

        # Nx = (len(knot_x) - 1 - degree_x)
        # Nt = (len(knot_t) - 1 - degree_t)
        # ind_del = [i for i in range(0, degree[0] * Nt)]
        # for j in range(0, degree[1]):
        #     ind_del = np.append(ind_del, [degree[0] * Nt + i * Nt + j for i in range(0, Nx - 2 * degree[0])])
        # for j in range(0, degree[1]):
        #     ind_del = np.append(ind_del, [degree[0] * Nt + i * Nt + Nt - 1 - j for i in range(0, Nx - 2 * degree[0])])
        # ind_del = np.append(ind_del, [i for i in range(Nt * Nx - degree[0] * Nt, Nt * Nx)])
        # B = np.delete(B, ind_del, 1)

    else:
        B = np.zeros((len(x0), (len(knot_t) - 3 * degree_t - 1) * (len(knot_x) - 3 * degree_x - 1)))
        for k in range(0, len(x0)):
            span_x = bs.find_span(knot_x, degree_x, x0[k])
            span_t = bs.find_span(knot_t, degree_t, t0[k])
            b_x = bs.basis_funs_all_ders(knot_x, degree_x, x0[k], span_x, N[0])
            b_t = bs.basis_funs_all_ders(knot_t, degree_t, t0[k], span_t, N[1])
            # for m in range(0, degree_x + 1):
            #     if span_x - 2 * degree_x + m < (len(knot_x) - 3 * degree_x - 1):
            #         for n in range(0, degree_t + 1):
            #             if span_t - 2 * degree_t + n < (len(knot_t) - 3 * degree_t - 1):
            #                 B[k, (span_x - 2 * degree_x + m) * (
            #                         len(knot_t) - 3 * degree_t - 1) + span_t - 2 * degree_t + n] = \
            #                     b_x[N[0], m] * b_t[N[1], n]
            for n in range(0, degree_t + 1):
                if span_t - 2 * degree_t + n < (len(knot_t) - 3 * degree_t - 1):
                    for m in range(0, degree_x + 1):
                        if span_x - 2 * degree_x + m < (len(knot_x) - 3 * degree_x - 1):
                            B[k, (span_t - 2 * degree_t + n) * (
                                    len(knot_x) - 3 * degree_x - 1) + span_x - 2 * degree_x + m] = \
                                b_x[N[0], m] * b_t[N[1], n]
    return B

def collocation_matrix_3d_M(node, degree, N, raw_data, form):
    """
    Generating collocation matrix case according to node, data and degree and the order of basis function for 3-D problem. It is used for multiprocessing.

    Parameters
    ----------
    node : float.
          The coordinate of knot.
    degree : list[int]
        The degree of B-spline basis function.
    N: list[int]
        The order of derivative.
    raw_data: DataFrame
        Raw data.
    form: int.
          0 for opened B-spline.
          1 for clamped B-spline.
    Returns
    -------
    B: ndarray.
          collocation matrix.
    """
    node_x = node['node_x']
    node_z = node['node_z']
    node_t = node['node_t']
    degree_x = degree[0]
    degree_z = degree[1]
    degree_t = degree[2]
    knot_x = bs.make_knots(node_x, degree_x, False)
    knot_z = bs.make_knots(node_z, degree_z, False)
    knot_t = bs.make_knots(node_t, degree_t, False)
    x0 = raw_data['x']
    z0 = raw_data['y']
    t0 = raw_data['t']

    # A=np.array([])
    if form == 0:
        B = np.zeros(
            (len(x0),
             (len(knot_x) - 3 * degree_x - 1) * (len(knot_z) - 3 * degree_z - 1) * (len(knot_t) - 3 * degree_t - 1)))
        for k in range(0, len(x0)):
            span_x = bs.find_span(knot_x, degree_x, x0[k])
            span_z = bs.find_span(knot_z, degree_z, z0[k])
            span_t = bs.find_span(knot_t, degree_t, t0[k])
            b_x = bs.basis_funs_all_ders(knot_x, degree_x, x0[k], span_x, N[0])
            b_z = bs.basis_funs_all_ders(knot_z, degree_z, z0[k], span_z, N[1])
            b_t = bs.basis_funs_all_ders(knot_t, degree_t, t0[k], span_t, N[2])
            for m in range(0, degree_x + 1):
                if span_x - 2 * degree_x + m < (len(knot_x) - 3 * degree_x - 1):
                    for n in range(0, degree_z + 1):
                        if span_z - 2 * degree_z + n < (len(knot_z) - 3 * degree_z - 1):
                            for i in range(0, degree_t + 1):
                                if span_t - 2 * degree_t + i < (len(knot_t) - 3 * degree_t - 1):
                                    B[k, (span_x - 2 * degree_x + m) * (len(knot_z) - 3 * degree_z - 1) * (
                                            len(knot_t) - 3 * degree_t - 1) + (span_z - 2 * degree_z + n) * (
                                                 len(knot_t) - 3 * degree_t - 1) + span_t - 2 * degree_t + i] = \
                                        b_x[N[0], m] * b_z[N[1], n] * b_t[N[2], i]
    else:
        B = np.zeros(
            (len(x0), (len(knot_t) - 1 - degree_t) * (len(knot_x) - 1 - degree_x) * (len(knot_z) - 1 - degree_z)))
        for k in range(0, len(x0)):
            span_x = bs.find_span(knot_x, degree_x, x0[k])
            span_z = bs.find_span(knot_z, degree_z, z0[k])
            span_t = bs.find_span(knot_t, degree_t, t0[k])
            b_x = bs.basis_funs_all_ders(knot_x, degree_x, x0[k], span_x, N[0])
            b_z = bs.basis_funs_all_ders(knot_t, degree_t, t0[k], span_t, N[1])
            b_t = bs.basis_funs_all_ders(knot_t, degree_t, t0[k], span_t, N[2])
            for m in range(0, degree_x + 1):
                for n in range(0, degree_z + 1):
                    for i in range(0, degree_t + 1):
                        B[k, (span_x - degree_x + m) * (span_z - degree_z + n) * (
                                len(knot_t) - 1 - degree_t) + span_t - degree_t + i] = \
                            b_x[N[0], m] * b_z[N[1], n] * b_t[N[2], i]

    return B

def collocation_matrix_3d_S(node, degree, N, raw_data, r, num_splits, form):
    """
    Generating collocation matrix according to node, data and degree and the order of basis function, for 3-D problem.

    Parameters
    ----------
    node : float.
          The coordinate of knot.
    degree : list[int]
        The degree of B-spline basis function.
    N: list[int]
        The order of derivative.
    raw_data: DataFrame
        Raw data.
    r: int
        The number of the batch.
    num_splits: int
        The number of splits.
    form: int.
          0 for opened B-spline.
          1 for clamped B-spline.
    Returns
    -------
    B: ndarray.
          collocation matrix.
    """
    node_x = node['node_x']
    node_z = node['node_z']
    node_t = node['node_t']
    degree_x = degree[0]
    degree_z = degree[1]
    degree_t = degree[2]
    knot_x = bs.make_knots(node_x, degree_x, False)
    knot_z = bs.make_knots(node_z, degree_z, False)
    knot_t = bs.make_knots(node_t, degree_t, False)
    arr=np.array(range(0,raw_data.shape[0]))
    splits=np.array_split(arr, num_splits)
    raw_data=raw_data.loc[splits[r]].reset_index(drop=True)

    x0 = raw_data['x']
    z0 = raw_data['y']
    t0 = raw_data['t']

    if form == 0:
        B = np.zeros(
            (len(x0),
             (len(knot_x) - 3 * degree_x - 1) * (len(knot_z) - 3 * degree_z - 1) * (len(knot_t) - 3 * degree_t - 1)))
        for k in range(0, len(x0)):
            span_x = bs.find_span(knot_x, degree_x, x0[k])
            span_z = bs.find_span(knot_z, degree_z, z0[k])
            span_t = bs.find_span(knot_t, degree_t, t0[k])
            b_x = bs.basis_funs_all_ders(knot_x, degree_x, x0[k], span_x, N[0])
            b_z = bs.basis_funs_all_ders(knot_z, degree_z, z0[k], span_z, N[1])
            b_t = bs.basis_funs_all_ders(knot_t, degree_t, t0[k], span_t, N[2])
            for m in range(0, degree_x + 1):
                if span_x - 2 * degree_x + m < (len(knot_x) - 3 * degree_x - 1):
                    for n in range(0, degree_z + 1):
                        if span_z - 2 * degree_z + n < (len(knot_z) - 3 * degree_z - 1):
                            for i in range(0, degree_t + 1):
                                if span_t - 2 * degree_t + i < (len(knot_t) - 3 * degree_t - 1):
                                    B[k, (span_x - 2 * degree_x + m) * (len(knot_z) - 3 * degree_z - 1) * (
                                            len(knot_t) - 3 * degree_t - 1) + (span_z - 2 * degree_z + n) * (
                                                 len(knot_t) - 3 * degree_t - 1) + span_t - 2 * degree_t + i] = \
                                        b_x[N[0], m] * b_z[N[1], n] * b_t[N[2], i]
    else:
        B = np.zeros(
            (len(x0), (len(knot_t) - 1 - degree_t) * (len(knot_x) - 1 - degree_x) * (len(knot_z) - 1 - degree_z)))
        for k in range(0, len(x0)):
            span_x = bs.find_span(knot_x, degree_x, x0[k])
            span_z = bs.find_span(knot_z, degree_z, z0[k])
            span_t = bs.find_span(knot_t, degree_t, t0[k])
            b_x = bs.basis_funs_all_ders(knot_x, degree_x, x0[k], span_x, N[0])
            b_z = bs.basis_funs_all_ders(knot_t, degree_t, t0[k], span_t, N[1])
            b_t = bs.basis_funs_all_ders(knot_t, degree_t, t0[k], span_t, N[2])
            for m in range(0, degree_x + 1):
                for n in range(0, degree_z + 1):
                    for i in range(0, degree_t + 1):
                        B[k, (span_x - degree_x + m) * (span_z - degree_z + n) * (
                                len(knot_t) - 1 - degree_t) + span_t - degree_t + i] = \
                            b_x[N[0], m] * b_z[N[1], n] * b_t[N[2], i]

    return B

def position_optimization(node_new, curv, lr, degree, form):
    """
    Adjusting the position of knots based on current knot vectors and data feature distribution.

    Parameters
    ----------
    node_new : Dictionary.
        Current knot vector.
    curv : float
        The data feature of each interval or block.
    lr: float
        Learning rate
    degree : list[int]
        The degree of B-spline basis function.
    form: int.
        0 for opened B-spline
        1 for clamped B-spline
    Returns
    -------
    node: Dictionary.
         Knot ector of optimized knots.
    """
    dimension = len(node_new)
    if dimension == 1:
        t = curv['t']
        if form == 0:
            node = [i for i in node_new['node_t']]
            node0 = [node[i] for i in range(0, degree)]
            node1 = [node[i] for i in range(- degree, 0)]
            node = np.delete(node, range(0, degree), 0)
            node = np.delete(node, range(- degree, 0), 0)
        else:
            node = [i for i in node_new['node_t']]
        knot_mid = [node[i] for i in range(0, len(node))]
        U = knot_mid[1:len(knot_mid) - 1]  # The knot on boundary can not be optimized
        k = np.zeros((len(knot_mid) - 1,))  # The data feature distribution in each interval
        for i in range(0, len(k)):
            k[i] = np.sum([curv['curvature'][j] for j in range(0, len(curv['curvature'])) if
                           curv['t'][j] >= knot_mid[i] and curv['t'][j] <= knot_mid[i + 1]])
        E_pre = np.var(k)

        while 1:
            grad = 2 * np.diff(k)
            U = U + lr * grad
            knot_mid[1:len(knot_mid) - 1] = U
            for i in range(0, len(k)):
                k[i] = np.sum([curv['curvature'][j] for j in range(0, len(curv['curvature'])) if
                               curv['t'][j] >= knot_mid[i] and curv['t'][j] <= knot_mid[i + 1]])

            E = np.var(k)
            a = np.sum((np.sort(U) - U) * (np.sort(U) - U))

            if ((a == 0 and U[0] > node[0]
                 and U[-1] < node[-1])
                    and E <= E_pre):
                node_mmm = [i for i in knot_mid]
                if form == 0:
                    node_mmm = np.insert(node_mmm, 0, node0, 0)
                    node_mmm = np.append(node_mmm, node1)
                B = collocation_matrix(node_mmm, degree, 0, t, form)
                if ([sum(B[:, i]) for i in range(0, B.shape[
                    1])] != 0):
                    # Make sure that all knots are in order and don't break boundary after optimization
                    node[1:len(node) - 1] = U
                    E_pre = np.var(k)
            else:
                break

        if form == 0:
            node = np.insert(node, 0, node0, 0)
            node = np.append(node, node1)
        node_po = {'node_t': node}
    elif dimension == 2:
        x = np.unique(curv['x'])
        t = np.unique(curv['t'])
        curv1 = curv['curvature_x']
        curv_mat_x = np.zeros((len(x), len(t)))
        for i in range(0, len(x)):
            curv_mat_x[i, :] = curv1[i * len(t):(i + 1) * len(t)]
        curv1 = curv['curvature_t']
        curv_mat_t = np.zeros((len(x), len(t)))
        for i in range(0, len(x)):
            curv_mat_t[i, :] = curv1[i * len(t):(i + 1) * len(t)]
        if form == 0:
            node_x = [i for i in node_new['node_x']]
            node0_x = [node_x[i] for i in range(0, degree[0])]
            node1_x = [node_x[i] for i in range(- degree[0], 0)]
            node_x = np.delete(node_x, range(0, degree[0]), 0)
            node_x = np.delete(node_x, range(- degree[0], 0), 0)

            node_t = [i for i in node_new['node_t']]
            node0_t = [node_t[i] for i in range(0, degree[1])]
            node1_t = [node_t[i] for i in range(- degree[1], 0)]
            node_t = np.delete(node_t, range(0, degree[1]), 0)
            node_t = np.delete(node_t, range(- degree[1], 0), 0)
        else:
            node_x = [i for i in node_new['node_x']]
            node_t = [i for i in node_new['node_t']]
        knot_mid_x = [i for i in node_x]
        knot_mid_t = [i for i in node_t]
        k = np.zeros((len(knot_mid_x) - 1, len(knot_mid_t) - 1))  # The data feature distribution in each block
        m_l = np.zeros((k.shape[0]))
        n_l = np.zeros((k.shape[1]))
        for i in range(1, k.shape[0]):
            m_l[i] = ((x - knot_mid_x[i]) >= 0).argmax()
        for j in range(1, k.shape[1]):
            n_l[j] = ((t - knot_mid_t[j]) >= 0).argmax()

        m_r = m_l[1:len(m_l)] - 1
        n_r = n_l[1:len(n_l)] - 1
        m_r = np.append(m_r, len(x) - 1)
        n_r = np.append(n_r, len(t) - 1)

        # x knot optimization
        curv_mat = curv_mat_x
        U_x = knot_mid_x[1:len(knot_mid_x) - 1]  # The knot on boundary can not be optimized
        for i in range(0, k.shape[0]):
            for j in range(0, k.shape[1]):
                k[i, j] = np.sum(curv_mat[int(m_l[i]):int(m_r[i]), int(n_l[j]):int(n_r[j])])
        E_pre = np.var(k)
        while 1:
            grad = np.zeros((len(U_x),))
            for i in range(0, len(U_x)):
                diff_array = np.absolute(x - U_x[i])
                aa = curv_mat[diff_array.argmin(), :]
                a = np.zeros((1, k.shape[1]))
                for j in range(0, k.shape[1]):
                    a[0, j] = np.sum(
                        [aa[m] for m in range(0, len(t)) if t[m] >= knot_mid_t[j] and t[m] <= knot_mid_t[j + 1]])
                grad[i] = 2 * np.sum((k[i + 1, :] - k[i, :]) * a)
            U_x = U_x + lr[0] * grad
            knot_mid_x[1:len(knot_mid_x) - 1] = U_x

            m_l = np.zeros((k.shape[0]))
            n_l = np.zeros((k.shape[1]))
            for i in range(1, k.shape[0]):
                m_l[i] = ((x - knot_mid_x[i]) >= 0).argmax()
            for j in range(1, k.shape[1]):
                n_l[j] = ((t - knot_mid_t[j]) >= 0).argmax()

            m_r = m_l[1:len(m_l)] - 1
            n_r = n_l[1:len(n_l)] - 1
            m_r = np.append(m_r, len(x) - 1)
            n_r = np.append(n_r, len(t) - 1)
            for i in range(0, k.shape[0]):
                for j in range(0, k.shape[1]):
                    k[i, j] = np.sum(curv_mat[int(m_l[i]):int(m_r[i]), int(n_l[j]):int(n_r[j])])

            E = np.var(k)

            if ((((U_x + lr[0] * grad == np.sort(U_x + lr[0] * grad)).all()
                  and (U_x + lr[0] * grad)[0] > node_x[0]
                  and (U_x + lr[0] * grad)[-1] < node_x[-1])
                 and E <= E_pre)):
                node_mmm_x = [i for i in knot_mid_x]
                if form == 0:
                    node_mmm_x = np.insert(node_mmm_x, 0, node0_x, 0)
                    node_mmm_x = np.append(node_mmm_x, node1_x)
                B = collocation_matrix(node_mmm_x, degree[0], 0, x, form)
                if ([sum(B[:, i]) for i in range(0, B.shape[
                    1])] != 0):
                    E_pre = E.copy()
                    node_x[1:len(node_x) - 1] = U_x
                else:
                    break
            else:
                break

            # t knot optimization
            curv_mat = curv_mat_t
            U_t = knot_mid_t[1:len(knot_mid_t) - 1]  # The knot on boundary can not be optimized
            for i in range(0, k.shape[0]):
                for j in range(0, k.shape[1]):
                    k[i, j] = np.sum(curv_mat[int(m_l[i]):int(m_r[i]), int(n_l[j]):int(n_r[j])])
            E_pre = np.var(k)

            while 1:
                grad = np.zeros((len(U_t),))
                for i in range(0, len(U_t)):
                    diff_array = np.absolute(t - U_t[i])
                    aa = curv_mat[:, diff_array.argmin()]
                    a = np.zeros((k.shape[0],))
                    for j in range(0, k.shape[0]):
                        a[j] = np.sum(
                            [aa[m] for m in range(0, len(x)) if x[m] >= knot_mid_x[j] and x[m] <= knot_mid_x[j + 1]])
                    grad[i] = 2 * np.sum((k[:, i + 1] - k[:, i]) * a)
                U_t = U_t + lr[1] * grad
                knot_mid_t[1:len(knot_mid_t) - 1] = U_t

                m_l = np.zeros((k.shape[0]))
                n_l = np.zeros((k.shape[1]))
                for i in range(1, k.shape[0]):
                    m_l[i] = ((x - knot_mid_x[i]) >= 0).argmax()
                for j in range(1, k.shape[1]):
                    n_l[j] = ((t - knot_mid_t[j]) >= 0).argmax()

                m_r = m_l[1:len(m_l)] - 1
                n_r = n_l[1:len(n_l)] - 1
                m_r = np.append(m_r, len(x) - 1)
                n_r = np.append(n_r, len(t) - 1)
                for i in range(0, k.shape[0]):
                    for j in range(0, k.shape[1]):
                        k[i, j] = np.sum(curv_mat[int(m_l[i]):int(m_r[i]), int(n_l[j]):int(n_r[j])])

                E = np.var(k)

                if (((U_t + lr[1] * grad == np.sort(U_t + lr[1] * grad)).all()
                     and (U_t + lr[1] * grad)[0] > node_t[0]
                     and (U_t + lr[1] * grad)[-1] < node_t[-1])
                        and E <= E_pre):
                    # and flag == len(knot_mid_t) - 1:
                    node_mmm_t = [i for i in knot_mid_t]
                    if form == 0:
                        node_mmm_t = np.insert(node_mmm_t, 0, node0_t, 0)
                        node_mmm_t = np.append(node_mmm_t, node1_t)
                    B = collocation_matrix(node_mmm_t, degree[1], 0, t, form)
                    if ([sum(B[:, i]) for i in range(0, B.shape[
                        1])] != 0):
                        node_t[1:len(node_t) - 1] = U_t
                        E_pre = np.var(k)
                    else:
                        break
                else:
                    break
        if form == 0:
            node_x = np.insert(node_x, 0, node0_x, 0)
            node_x = np.append(node_x, node1_x)

            node_t = np.insert(node_t, 0, node0_t, 0)
            node_t = np.append(node_t, node1_t)
        node_po = {'node_x': node_x, 'node_t': node_t}
    return node_po


def knot_refinement(R, node, M, degree, form):
    """
    Add new knots on the place with larger error

    Parameters
    ----------
    R: DataFrame
       The distribution of total residual
    node : Dictionary
        Current knot vector.
    M : list[int]
        The knots needed to be added each time on each direction.
    Degree : list[int]
        The degree of basis function on different directions.
    form: int.
        0 for opened B-spline.
        1 for clamped B-spline.
    Returns
    -------
    node_new: Dictionary.
          Coodinate vector of optimized knots.
    """
    global node_new
    global node_new_x
    global node_new_t
    dimension = len(node)
    min_max_scaler = preprocessing.MinMaxScaler()
    res = R['residual']
    if dimension == 1:
        # Calculate the residual distribution on each interval or block
        m = M
        t = R['t']
        if form == 0:
            node = node['node_t']
            node0 = [node[i] for i in range(0, degree)]
            node1 = [node[i] for i in range(- degree, 0)]
            node = np.delete(node, range(0, degree), 0)
            node_mid = np.delete(node, range(- degree, 0), 0)
        else:
            node_mid = node['node_t']
        r = np.zeros((len(node_mid) - 1,))  # The residual distribution in each interval
        for i in range(0, len(r)):
            r[i] = np.sum([res[j] for j in range(0, len(res)) if
                            t[j] >= node_mid[i] and t[j] < node_mid[i + 1]])
        while M > 0:
            ind = np.argmax(r)
            new_knot = (node_mid[ind] + node_mid[ind + 1]) / 2
            node_mid = np.insert(node_mid, ind + 1, new_knot)

            r = np.zeros((len(node_mid) - 1,))  # The residual distribution in each interval
            for i in range(0, len(r)):
                r[i] = np.sum([res[j] for j in range(0, len(res)) if
                                t[j] >= node_mid[i] and t[j] < node_mid[i + 1]])
            node_mmm = [i for i in node_mid]
            if form == 0:
                node_mmm = np.insert(node_mmm, 0, node0, 0)
                node_mmm = np.append(node_mmm, node1)
            B = collocation_matrix(node_mmm, degree, 0, t, form)
            if ([sum(B[:, i]) for i in range(0, B.shape[
                1])] != 0):  # and flag == len(
                # node_mid) - 1:
                node_new = [i for i in node_mid]
                M = M - 1
            else:
                break
        if M == m:
            node_n = node
            print('No new knot is added!')
        else:
            if form == 0:
                node_new = np.insert(node_new, 0, node0, 0)
                node_new = np.append(node_new, node1)
            node_n = {'node_t': node_new}

    elif dimension == 2:
        M_x = M[0]
        M_t = M[1]
        m_x = M_x
        m_t = M_t
        x = R['x']
        t = R['t']
        if form == 0:
            node_mid_x = node['node_x']
            node_mid_t = node['node_t']
            node0_x = [node_mid_x[i] for i in range(0, degree[0])]
            node1_x = [node_mid_x[i] for i in range(-degree[0], 0)]
            node_mid_x = np.delete(node_mid_x, range(0, degree[0]), 0)
            node_mid_x = np.delete(node_mid_x, range(-degree[0], 0), 0)

            node0_t = [node_mid_t[i] for i in range(0, degree[1])]
            node1_t = [node_mid_t[i] for i in range(- degree[1], 0)]
            node_mid_t = np.delete(node_mid_t, range(0, degree[1]), 0)
            node_mid_t = np.delete(node_mid_t, range(-degree[1], 0), 0)
        else:
            node_mid_x = node['node_x']
            node_mid_t = node['node_t']

        # Add knots in x direction
        r_x1 = np.zeros((len(node_mid_x) - 1,))  # The residual distribution in each interval
        for i in range(0, len(r_x1)):
            rrr = [res[j] for j in range(0, len(res)) if
                   x[j] > node_mid_x[i] and x[j] < node_mid_x[i + 1]]
            if not rrr:
                rrr = 0
            r_x1[i] = np.sum(rrr)
        r_x = r_x1

        while M_x > 0:
            node_copy_x = [i for i in node_mid_x]
            ind = np.argmax(r_x)
            new_knot = (node_mid_x[ind] + node_mid_x[ind + 1]) / 2
            node_mid_x = np.insert(node_mid_x, ind + 1, new_knot)

            r_x1 = np.zeros((len(node_mid_x) - 1,))  # The residual distribution in each interval
            for i in range(0, len(r_x1)):
                rrr = [res[j] for j in range(0, len(res)) if
                       x[j] > node_mid_x[i] and x[j] < node_mid_x[i + 1]]
                if not rrr:
                    rrr = 0
                r_x1[i] = np.sum(rrr)

            r_x = r_x1
            node_mmm_x = [i for i in node_mid_x]
            if form == 0:
                node_mmm_x = np.insert(node_mmm_x, 0, node0_x, 0)
                node_mmm_x = np.append(node_mmm_x, node1_x)
            B = collocation_matrix(node_mmm_x, degree[0], 0, x, form)
            if ([sum(B[:, i]) for i in range(0, B.shape[
                1])] != 0):
                # and flag == len(
                # node_mid_x) - 1:
                node_new_x = [i for i in node_mid_x]
                M_x = M_x - 1
            else:
                break

        # Add knots in t direction
        r_t1 = np.zeros((len(node_mid_t) - 1,))  # The residual distribution in each interval
        for i in range(0, len(r_t1)):
            rrr = [res[j] for j in range(0, len(res)) if
                   t[j] > node_mid_t[i] and t[j] < node_mid_t[i + 1]]
            if not rrr:
                rrr = 0
            r_t1[i] = np.sum(rrr)
        r_t = r_t1

        while M_t > 0:
            ind = np.argmax(r_t)
            new_knot = (node_mid_t[ind] + node_mid_t[ind + 1]) / 2
            node_mid_t = np.insert(node_mid_t, ind + 1, new_knot)

            r_t1 = np.zeros((len(node_mid_t) - 1,))  # The residual distribution in each interval
            for i in range(0, len(r_t1)):
                rrr = [res[j] for j in range(0, len(res)) if
                       t[j] > node_mid_t[i] and t[j] < node_mid_t[i + 1]]
                if not rrr:
                    rrr = 0
                r_t1[i] = np.sum(rrr)
            r_t = r_t1

            node_mmm_t = [i for i in node_mid_t]
            if form == 0:
                node_mmm_t = np.insert(node_mmm_t, 0, node0_t, 0)
                node_mmm_t = np.append(node_mmm_t, node1_t)
            B = collocation_matrix(node_mmm_t, degree[1], 0, t, form)
            if ([sum(B[:, i]) for i in range(0, B.shape[
                1])] != 0):
                # and flag == len(
                # node_mid_t) - 1:
                node_new_t = [i for i in node_mid_t]
                M_t = M_t - 1
            else:
                break

        if M_x == m_x:
            node_new_x = node_mid_x
        if M_t == m_t:
            node_new_t = node_mid_t

        if M_x == m_x and M_t == m_t:
            print('No new knot is added!')
        if form == 0:
            node_new_x = np.insert(node_new_x, 0, node0_x, 0)
            node_new_x = np.append(node_new_x, node1_x)

            node_new_t = np.insert(node_new_t, 0, node0_t, 0)
            node_new_t = np.append(node_new_t, node1_t)
        node_n = {'node_x': node_new_x, 'node_t': node_new_t}

    elif dimension == 3:
        M_x = M[0]
        M_z = M[1]
        M_t = M[2]
        m_x = M_x
        m_z = M_z
        m_t = M_t
        x = R['x']
        z = R['z']
        t = R['t']
        if form == 0:
            node_mid_x = node['node_x']
            node_mid_z = node['node_z']
            node_mid_t = node['node_t']

            node0_x = [node_mid_x[i] for i in range(0, degree[0])]
            node1_x = [node_mid_x[i] for i in range(-degree[0], 0)]
            node_mid_x = np.delete(node_mid_x, range(0, degree[0]), 0)
            node_mid_x = np.delete(node_mid_x, range(-degree[0], 0), 0)

            node0_z = [node_mid_z[i] for i in range(0, degree[1])]
            node1_z = [node_mid_z[i] for i in range(-degree[1], 0)]
            node_mid_z = np.delete(node_mid_z, range(0, degree[1]), 0)
            node_mid_z = np.delete(node_mid_z, range(-degree[1], 0), 0)

            node0_t = [node_mid_t[i] for i in range(0, degree[2])]
            node1_t = [node_mid_t[i] for i in range(- degree[2], 0)]
            node_mid_t = np.delete(node_mid_t, range(0, degree[2]), 0)
            node_mid_t = np.delete(node_mid_t, range(-degree[2], 0), 0)
        else:
            node_mid_x = node['node_x']
            node_mid_z = node['node_z']
            node_mid_t = node['node_t']

        # Add knots in x direction
        r_x1 = np.zeros((len(node_mid_x) - 1,))  # The residual distribution in each interval
        for i in range(0, len(r_x1)):
            rrr = [res[j] for j in range(0, len(res)) if
                   x[j] > node_mid_x[i] and x[j] < node_mid_x[i + 1]]
            if not rrr:
                rrr = 0
            r_x1[i] = np.sum(rrr)
        r_x = r_x1
        while M_x > 0:
            ind = np.argmax(r_x)
            new_knot = (node_mid_x[ind] + node_mid_x[ind + 1]) / 2
            node_mid_x = np.insert(node_mid_x, ind + 1, new_knot)

            r_x1 = np.zeros((len(node_mid_x) - 1,))  # The residual distribution in each interval
            for i in range(0, len(r_x1)):
                rrr = [res[j] for j in range(0, len(res)) if
                       x[j] > node_mid_x[i] and x[j] < node_mid_x[i + 1]]
                if not rrr:
                    rrr = 0
                r_x1[i] = np.sum(rrr)
            r_x = r_x1
            node_mmm_x = [i for i in node_mid_x]
            if form == 0:
                node_mmm_x = np.insert(node_mmm_x, 0, node0_x, 0)
                node_mmm_x = np.append(node_mmm_x, node1_x)
            B = collocation_matrix(node_mmm_x, degree[0], 0, x, form)
            if ([sum(B[:, i]) for i in range(0, B.shape[
                1])] != 0):

                node_new_x = [i for i in node_mid_x]
                M_x = M_x - 1
            else:
                break

        # Add knots in z direction
        r_z1 = np.zeros((len(node_mid_z) - 1,))  # The residual distribution in each interval
        for i in range(0, len(r_z1)):
            rrr = [res[j] for j in range(0, len(res)) if
                   z[j] > node_mid_z[i] and z[j] < node_mid_z[i + 1]]
            if not rrr:
                rrr = 0
            r_z1[i] = np.sum(rrr)
        r_z = r_z1

        while M_z > 0:
            ind = np.argmax(r_z)
            new_knot = (node_mid_z[ind] + node_mid_z[ind + 1]) / 2
            node_mid_z = np.insert(node_mid_z, ind + 1, new_knot)

            r_z1 = np.zeros((len(node_mid_z) - 1,))  # The residual distribution in each interval
            for i in range(0, len(r_z1)):
                rrr = [res[j] for j in range(0, len(res)) if
                       z[j] > node_mid_z[i] and z[j] < node_mid_z[i + 1]]
                if not rrr:
                    rrr = 0
                r_z1[i] = np.sum(rrr)
            r_z = r_z1

            node_mmm_z = [i for i in node_mid_z]
            if form == 0:
                node_mmm_z = np.insert(node_mmm_z, 0, node0_z, 0)
                node_mmm_z = np.append(node_mmm_z, node1_z)
            B = collocation_matrix(node_mmm_z, degree[0], 0, x, form)
            if ([sum(B[:, i]) for i in range(0, B.shape[
                1])] != 0):
                node_new_z = [i for i in node_mid_z]
                M_z = M_z - 1
            else:
                break
        # Add knots in t direction
        r_t1 = np.zeros((len(node_mid_t) - 1,))  # The residual distribution in each interval
        for i in range(0, len(r_t1)):
            rrr = [res[j] for j in range(0, len(res)) if
                   t[j] > node_mid_t[i] and t[j] < node_mid_t[i + 1]]
            if not rrr:
                rrr = 0
            r_t1[i] = np.sum(rrr)
        r_t = r_t1
        while M_t > 0:
            ind = np.argmax(r_t)
            new_knot = (node_mid_t[ind] + node_mid_t[ind + 1]) / 2
            node_mid_t = np.insert(node_mid_t, ind + 1, new_knot)

            r_t1 = np.zeros((len(node_mid_t) - 1,))  # The residual distribution in each interval
            for i in range(0, len(r_t1)):
                rrr = [res[j] for j in range(0, len(res)) if
                       t[j] > node_mid_t[i] and t[j] < node_mid_t[i + 1]]
                if not rrr:
                    rrr = 0
                r_t1[i] = np.sum(rrr)
            r_t = r_t1

            node_mmm_t = [i for i in node_mid_t]
            if form == 0:
                node_mmm_t = np.insert(node_mmm_t, 0, node0_t, 0)
                node_mmm_t = np.append(node_mmm_t, node1_t)
            B = collocation_matrix(node_mmm_t, degree[1], 0, t, form)
            if ([sum(B[:, i]) for i in range(0, B.shape[
                1])] != 0):

                node_new_t = [i for i in node_mid_t]
                M_t = M_t - 1
            else:
                break

        if M_x == m_x:
            node_new_x = node_mid_x
        if M_z == m_z:
            node_new_z = node_mid_z
        if M_t == m_t:
            node_new_t = node_mid_t

        if M_x == m_x and M_t == m_t and M_z == m_z:
            print('No new knot is added!')
        if form == 0:
            node_new_x = np.insert(node_new_x, 0, node0_x, 0)
            node_new_x = np.append(node_new_x, node1_x)
            node_new_z = np.insert(node_new_z, 0, node0_z, 0)
            node_new_z = np.append(node_new_z, node1_z)
            node_new_t = np.insert(node_new_t, 0, node0_t, 0)
            node_new_t = np.append(node_new_t, node1_t)
            node_n = {'node_x': node_new_x, 'node_z': node_new_z, 'node_t': node_new_t}

    return node_n


# ==============================================================================

def exact_line_search_KS(B, B_x, B_t, B_2x, B_4x, u, Bbeta, beta, theta, a1):
    nabla = Bbeta - beta
    Nc = B.shape[0]
    A1 = (B @ nabla) * (B_x @ nabla)
    B1 = (B_t @ nabla + (B @ beta) * (B_x @ nabla) + (B @ nabla) * (
            B_x @ beta) + B_2x @ nabla + theta * B_4x @ nabla)
    C1 = (B_t @ beta + (B @ beta) * (B_x @ beta) + B_2x @ beta + theta * B_4x @ beta)

    a = (2 / Nc * A1.T @ A1).cpu().detach().numpy()
    b = (3 / 2 / Nc * (A1.T @ B1 + B1.T @ A1)).cpu().detach().numpy()
    c = (1 / Nc * (A1.T @ C1 + B1.T @ B1 + C1.T @ A1) + a1 / Nc * nabla.T @ B.T @ B @ nabla).cpu().detach().numpy()
    d = (1 / 2 / Nc * (B1.T @ C1 + C1.T @ B1) + a1 / 2 / Nc * (
            nabla.T @ B.T @ (B @ beta - u) + (B @ beta - u).T @ B @ nabla)).cpu().detach().numpy()
    a, b, c, d = a / max(abs(a), abs(b), abs(c), abs(d)), b / max(abs(a), abs(b), abs(c), abs(d)), c / max(abs(a),
                                                                                                           abs(b),
                                                                                                           abs(c),
                                                                                                           abs(d)), d / max(
        abs(a), abs(b), abs(c), abs(d))
    x = solve(a, b, c, d)
    ind = np.isreal(x)
    ii = [i for i in range(0, len(ind)) if ind[i] == True]
    x = x[ii]
    x = x.real
    x = np.unique(x)
    x = x[np.where(x <= 1)]
    x = x[np.where(x >= 0)]
    if len(x) == 0:
        if a / 4 + b / 3 + c / 2 + d < 0:
            x = 1.
        else:
            x = 0.
        gamma = torch.from_numpy(np.array(x))
    elif len(x) == 1:
        gamma = torch.from_numpy(np.array(x))
    else:
        y = a / 4 * x ** 4 + b / 3 * x ** 3 + c / 2 * x ** 2 + d * x
        min_index = np.argmin(y, axis=0)
        x = x[min_index[0]]
        gamma = torch.from_numpy(np.array(x))
    return gamma

def solve(a, b, c, d):
    if (a == 0 and b == 0):  # Case for handling Liner Equation
        return np.array([(-d * 1.0) / c])  # Returning linear root as numpy array.

    elif (a == 0):  # Case for handling Quadratic Equations

        D = c * c - 4.0 * b * d  # Helper Temporary Variable
        if D >= 0:
            D = math.sqrt(D)
            x1 = (-c + D) / (2.0 * b)
            x2 = (-c - D) / (2.0 * b)
        else:
            D = math.sqrt(-D)
            x1 = (-c + D * 1j) / (2.0 * b)
            x2 = (-c - D * 1j) / (2.0 * b)

        return np.array([x1, x2])  # Returning Quadratic Roots as numpy array.

    f = findF(a, b, c)  # Helper Temporary Variable
    g = findG(a, b, c, d)  # Helper Temporary Variable
    h = findH(g, f)  # Helper Temporary Variable

    if f == 0 and g == 0 and h == 0:  # All 3 Roots are Real and Equal
        if (d / a) >= 0:
            x = (d / (1.0 * a)) ** (1 / 3.0) * -1
        else:
            x = (-d / (1.0 * a)) ** (1 / 3.0)
        return np.array([x, x, x])  # Returning Equal Roots as numpy array.

    elif h <= 0:  # All 3 roots are Real

        i = math.sqrt(((g ** 2.0) / 4.0) - h)  # Helper Temporary Variable
        j = i ** (1 / 3.0)  # Helper Temporary Variable
        k = math.acos(-(g / (2 * i)))  # Helper Temporary Variable
        L = j * -1  # Helper Temporary Variable
        M = math.cos(k / 3.0)  # Helper Temporary Variable
        N = math.sqrt(3) * math.sin(k / 3.0)  # Helper Temporary Variable
        P = (b / (3.0 * a)) * -1  # Helper Temporary Variable

        x1 = 2 * j * math.cos(k / 3.0) - (b / (3.0 * a))
        x2 = L * (M + N) + P
        x3 = L * (M - N) + P

        return np.array([x1, x2, x3])  # Returning Real Roots as numpy array.

    elif h > 0:  # One Real Root and two Complex Roots
        R = -(g / 2.0) + math.sqrt(h)  # Helper Temporary Variable
        if R >= 0:
            S = R ** (1 / 3.0)  # Helper Temporary Variable
        else:
            S = (-R) ** (1 / 3.0) * -1  # Helper Temporary Variable
        T = -(g / 2.0) - math.sqrt(h)
        if T >= 0:
            U = (T ** (1 / 3.0))  # Helper Temporary Variable
        else:
            U = ((-T) ** (1 / 3.0)) * -1  # Helper Temporary Variable

        x1 = (S + U) - (b / (3.0 * a))
        x2 = -(S + U) / 2 - (b / (3.0 * a)) + (S - U) * math.sqrt(3) * 0.5j
        x3 = -(S + U) / 2 - (b / (3.0 * a)) - (S - U) * math.sqrt(3) * 0.5j

        return np.array([x1, x2, x3])  # Returning One Real Root and two Complex Roots as numpy array.


# Helper function to return float value of f.
def findF(a, b, c):
    return ((3.0 * c / a) - ((b ** 2.0) / (a ** 2.0))) / 3.0


# Helper function to return float value of g.
def findG(a, b, c, d):
    return (((2.0 * (b ** 3.0)) / (a ** 3.0)) - ((9.0 * b * c) / (a ** 2.0)) + (27.0 * d / a)) / 27.0


# Helper function to return float value of h.
def findH(g, f):
    return ((g ** 2.0) / 4.0 + (f ** 3.0) / 27.0)



def sca_KS(raw_data,raw_data_ic,raw_data_bc, B, B_t, B_x, B_2x, B_4x,B_ic,B_bc,a1,a2,a3):
    device = torch.device("cuda")
    U = raw_data['u']
    u = np.random.rand(len(U), 1)
    for i in range(0, len(U)):
        u[i] = U[i]
    u = (torch.tensor(u, dtype=torch.float64)).to(device)

    IC = raw_data_ic['u']
    ic = np.random.rand(len(IC), 1)
    for i in range(0, len(IC)):
        ic[i] = IC[i]
    ic = (torch.tensor(ic, dtype=torch.float64)).to(device)

    BC = raw_data_bc['u']
    bc = np.random.rand(len(BC), 1)
    for i in range(0, len(BC)):
        bc[i] = BC[i]
    bc = (torch.tensor(bc, dtype=torch.float64)).to(device)

    t0 = raw_data['t']
    x0 = raw_data['x']

    # Operate BSCA
    theta = -np.random.rand(3, 1)
    theta = (torch.tensor(theta, dtype=torch.float64)).to(device)

    beta = np.random.rand(B.shape[1], 1)

    beta = (torch.tensor(beta, dtype=torch.float64)).to(device)
    gamma = (torch.tensor(1, dtype=torch.float64)).to(device)
    beta.requires_grad_()
    c = (torch.tensor(1e-7,dtype=torch.float64)).to(device)

    Nc = B.shape[0]
    Nic = B_ic.shape[0]
    Nbc = B_bc.shape[0]
    # Begin
    F = (torch.cat(((B @ beta) * (B_x @ beta), B_2x @ beta, B_4x @ beta), 1)).to(device)
    ite = 0
    fit_res = 1 / 2 * torch.mean((u - B @ beta) ** 2)
    ic_res = 1 / 2 * torch.mean((ic - B_ic @ beta) ** 2)
    bc_res = 1 / 2 * torch.mean((bc - B_bc @ beta) ** 2)
    pde_res = 1 / 2 * torch.mean((B_t @ beta - F @ theta) ** 2)
    res = a1*fit_res +a2*ic_res +a3*bc_res + pde_res
    print('Stepsize | Iterations | Fitting residual | IC residual | BC residual | PDE residual | Total residual | Theta')

    Res = np.array([fit_res.cpu().detach().numpy(),ic_res.cpu().detach().numpy(),bc_res.cpu().detach().numpy(), pde_res.cpu().detach().numpy()]).reshape([1, 4])

    f_grads = autograd.grad(outputs=pde_res, inputs=beta, retain_graph=True)[0]
    I = torch.eye(B.shape[1]).to(device)
    Bbeta = torch.inverse(a1 / Nc * B.T @ B+a2/Nic* B_ic.T @ B_ic +a3/Nbc* B_bc.T @ B_bc+ c * I) @ (a1 / Nc * B.T @ u + a2/Nic* B_ic.T @ ic +a3/Nbc* B_bc.T @ bc + c * beta - f_grads)

    print(gamma.cpu().detach().numpy(), '|', ite, '| ', fit_res.cpu().detach().numpy(), ' | ',ic_res.cpu().detach().numpy(), ' | ',bc_res.cpu().detach().numpy(), ' | ',
          pde_res.cpu().detach().numpy(), ' | ',
          res.cpu().detach().numpy(), ' |', (theta.cpu().detach().numpy()).reshape([1, len(theta)]))
    gamma = (exact_line_search_KS_SINDy(B, B_x, B_t, B_2x, B_4x,B_ic,B_bc, u,ic,bc, Bbeta, beta, theta, a1, a2, a3)).to(device)
    beta = beta + gamma * (Bbeta - beta)
    F.copy_(torch.cat(((B @ beta) * (B_x @ beta), B_2x @ beta, B_4x @ beta), 1))
    ite = 1

    ## Update theta
    with torch.no_grad():
        theta.copy_(torch.inverse(F.T @ F) @ F.T @ (B_t @ beta))
    Para = np.array([theta.cpu().detach().numpy()]).reshape([1, len(theta)])
    fit_res = 1 / 2 * torch.mean((u - B @ beta) ** 2)
    ic_res = 1 / 2 * torch.mean((ic - B_ic @ beta) ** 2)
    bc_res = 1 / 2 * torch.mean((bc - B_bc @ beta) ** 2)
    pde_res = 1 / 2 * torch.mean((B_t @ beta - F @ theta) ** 2)
    res = a1*fit_res +a2*ic_res +a3*bc_res + pde_res

    r = np.array([fit_res.cpu().detach().numpy(),ic_res.cpu().detach().numpy(),bc_res.cpu().detach().numpy(), pde_res.cpu().detach().numpy()]).reshape([1, 4])
    Res = np.concatenate((Res, r), axis=0)
    maxiter = 300
    for i in range(0, maxiter):
        with torch.no_grad():
            f_grads.copy_(autograd.grad(outputs=pde_res, inputs=beta, retain_graph=True)[0])
            Bbeta.copy_(torch.inverse(a1 / Nc * B.T @ B+a2/Nic* B_ic.T @ B_ic +a3/Nbc* B_bc.T @ B_bc+ c * I) @ (a1 / Nc * B.T @ u + a2/Nic* B_ic.T @ ic +a3/Nbc* B_bc.T @ bc + c * beta - f_grads))
        if i == maxiter - 1:
            break
        else:
            with torch.no_grad():
                # gamma = gamma * (1 - gamma * epsilon)
                gamma =  (exact_line_search_KS_SINDy(B, B_x, B_t, B_2x, B_4x,B_ic,B_bc, u,ic,bc, Bbeta, beta, theta, a1, a2, a3)).to(device)
                beta += gamma * (Bbeta - beta)
                F.copy_(torch.cat(((B @ beta) * (B_x @ beta), B_2x @ beta, B_4x @ beta), 1))
                ## Update theta
                theta.copy_(torch.inverse(F.T @ F) @ F.T @ (B_t @ beta))
                para = np.array([theta.cpu().detach().numpy()]).reshape([1, len(theta)])
                Para = np.concatenate((Para, para), axis=0)
                # Print iteration and residual of each term
            fit_res = 1 / 2 * torch.mean((u - B @ beta) ** 2)
            ic_res = 1 / 2 * torch.mean((ic - B_ic @ beta) ** 2)
            bc_res = 1 / 2 * torch.mean((bc - B_bc @ beta) ** 2)
            pde_res = 1 / 2 * torch.mean((B_t @ beta - F @ theta) ** 2)
            res = a1 * fit_res + a2 * ic_res + a3 * bc_res + pde_res
            ite += 1
            r = np.array([fit_res.cpu().detach().numpy(),ic_res.cpu().detach().numpy(),bc_res.cpu().detach().numpy(), pde_res.cpu().detach().numpy()]).reshape([1, 4])
            Res = np.concatenate((Res, r), axis=0)
    print(gamma.cpu().detach().numpy(), '|', ite, '| ', fit_res.cpu().detach().numpy(), ' | ',
          ic_res.cpu().detach().numpy(), ' | ', bc_res.cpu().detach().numpy(), ' | ',
          pde_res.cpu().detach().numpy(), ' | ',
          res.cpu().detach().numpy(), ' |', (theta.cpu().detach().numpy()).reshape([1, len(theta)]))
    fit_res = 1 / 2 * torch.mean((u - B @ beta) ** 2)
    ic_res = 1 / 2 * torch.mean((ic - B_ic @ beta) ** 2)
    bc_res = 1 / 2 * torch.mean((bc - B_bc @ beta) ** 2)
    pde_res = 1 / 2 * torch.mean((B_t @ beta - F @ theta) ** 2)
    res = a1*fit_res +a2*ic_res +a3*bc_res + pde_res
    Residual = Res
    print('\n')
    R = {'residual': (a1 / 2 * (u - B @ beta) ** 2 + 1 / 2 * (((B_t @ beta - F @ theta) ** 2)) ** 2).cpu().detach().numpy(),
         'x': x0,
         't': t0}
    R_fit = {'residual': ((u - B @ beta) ** 2).cpu().detach().numpy(),
             'x': x0,
             't': t0}
    R_phy = {'residual': ((B_t @ beta - F @ theta) ** 2).cpu().detach().numpy(),
             'x': x0,
             't': t0}
    return beta, theta, R, R_fit, R_phy, res, fit_res, pde_res, Residual,Para

def SCA_KS_SINDy(raw_data,raw_data_ic,raw_data_bc,B , B_t, B_x, B_2x, B_3x, B_4x,B_5x,B_ic,B_bc,a1,a2,a3, mu, threshold,maxiter,epsilon):
    device = torch.device("cuda")
    threshold = torch.tensor(threshold, dtype=torch.float64)

    U = raw_data['u']
    u = np.random.rand(len(U), 1)
    for i in range(0, len(U)):
        u[i] = U[i]
    u = (torch.tensor(u, dtype=torch.float64)).to(device)

    IC = raw_data_ic['u']
    ic = np.random.rand(len(IC), 1)
    for i in range(0, len(IC)):
        ic[i] = IC[i]
    ic = (torch.tensor(ic, dtype=torch.float64)).to(device)

    BC = raw_data_bc['u']
    bc = np.random.rand(len(BC), 1)
    for i in range(0, len(BC)):
        bc[i] = BC[i]
    bc = (torch.tensor(bc, dtype=torch.float64)).to(device)

    t0 = raw_data['t']
    x0 = raw_data['x']

    # Operate BSCA
    beta = np.random.rand(B.shape[1], 1)
    beta = (torch.tensor(beta, dtype=torch.float64)).to(device)
    beta.requires_grad_()

    index = [[i for i in range(0, 35)]]
    Phi = (library_KS(B, B_x, B_2x, B_3x, B_4x, B_5x, beta, index[0])).to(device)

    Lambda = np.zeros([Phi.shape[1], 1])
    Lambda = (torch.tensor(Lambda, dtype=torch.float64)).to(device)
    Para = np.array([Lambda.cpu().detach().numpy()]).reshape([1, len(Lambda)])
    gamma = (torch.tensor(1,dtype=torch.float64)).to(device)

    c =( torch.tensor(1e-5,dtype=torch.float64)).to(device)
    Nc = B.shape[0]
    Nic = B_ic.shape[0]
    Nbc = B_bc.shape[0]

    # Begin
    fit_res = 1 / 2 * torch.mean((u - B @ beta) ** 2)
    ic_res = 1 / 2 * torch.mean((ic - B_ic @ beta) ** 2)
    bc_res = 1 / 2 * torch.mean((bc - B_bc @ beta) ** 2)
    pde_res = 1 / 2 * torch.mean((B_t @ beta - Phi @ Lambda) ** 2)
    res = a1*fit_res + a2*ic_res + a3*bc_res + pde_res + mu / Nc * torch.sum(abs(Lambda))
    print('Stepsize | Iterations | Fitting residual | IC residual | BC residual | PDE residual | Total residual | Theta')
    print(gamma.cpu().detach().numpy(), '|', 0 , '| ', fit_res.cpu().detach().numpy(), ' | ',ic_res.cpu().detach().numpy(), ' | ',bc_res.cpu().detach().numpy(), ' | ',
          pde_res.cpu().detach().numpy(), ' | ', res.cpu().detach().numpy(), ' | ',
          ((Lambda[0:3]).cpu().detach().numpy()).reshape([1, 3]))
    Res = np.array([fit_res.cpu().detach().numpy(),ic_res.cpu().detach().numpy(),bc_res.cpu().detach().numpy(), pde_res.cpu().detach().numpy(),res.cpu().detach().numpy()]).reshape([1, 5])

    f_grads = (autograd.grad(outputs=pde_res, inputs=beta, retain_graph=True)[0]).to(device)
    I = (torch.eye(B.shape[1])).to(device)
    Bbeta = torch.inverse(a1 / Nc * B.T @ B+a2/Nic* B_ic.T @ B_ic +a3/Nbc* B_bc.T @ B_bc+ c * I) @ (a1 / Nc * B.T @ u + a2/Nic* B_ic.T @ ic +a3/Nbc* B_bc.T @ bc + c * beta - f_grads)
    beta = beta + gamma * (Bbeta - beta)
    Phi.copy_(library_KS(B, B_x, B_2x, B_3x, B_4x, B_5x, beta, index[0]))
    ## Update theta
    with torch.no_grad():
        model = fp.prox_method(Phi.cpu().detach().numpy(), ((B_t @ beta).cpu().detach().numpy()).reshape([len(u), ]), mu,
                               5000, 100000, 1e-12,1)
        model.train(method="FISTA")
        Lambda.copy_(torch.tensor((model.x).reshape([len(Lambda), 1]), dtype=torch.float64))
        # Lambda.copy_(torch.inverse(Phi.T @ Phi) @ Phi.T @ (B_t @ beta))
        para = np.array([Lambda.cpu().detach().numpy()]).reshape([1, len(Lambda)])
        Para = np.concatenate((Para, para), axis=0)

    fit_res = 1 / 2 * torch.mean((u - B @ beta) ** 2)
    ic_res = 1 / 2 * torch.mean((ic - B_ic @ beta) ** 2)
    bc_res = 1 / 2 * torch.mean((bc - B_bc @ beta) ** 2)
    pde_res = 1 / 2 * torch.mean((B_t @ beta - Phi @ Lambda) ** 2)
    res = a1*fit_res +a2*ic_res +a3*bc_res + pde_res + mu / Nc * torch.sum(abs(Lambda))
    print(gamma.cpu().detach().numpy(), '|', 1, '| ', fit_res.cpu().detach().numpy(), ' | ',
          ic_res.cpu().detach().numpy(), ' | ', bc_res.cpu().detach().numpy(), ' | ',
          pde_res.cpu().detach().numpy(), ' | ', res.cpu().detach().numpy(), ' | ',
          ((Lambda[0:3]).cpu().detach().numpy()).reshape([1, 3]))
    r = np.array([fit_res.cpu().detach().numpy(),ic_res.cpu().detach().numpy(),bc_res.cpu().detach().numpy(), pde_res.cpu().detach().numpy(),res.cpu().detach().numpy()]).reshape([1, 5])
    Res = np.concatenate((Res, r), axis=0)


    for i in range(0, maxiter):
        with torch.no_grad():
            f_grads.copy_(autograd.grad(outputs=pde_res, inputs=beta, retain_graph=True)[0])
            Bbeta.copy_(torch.inverse(a1 / Nc * B.T @ B+a2/Nic* B_ic.T @ B_ic +a3/Nbc* B_bc.T @ B_bc+ c * I) @ (a1 / Nc * B.T @ u + a2/Nic* B_ic.T @ ic +a3/Nbc* B_bc.T @ bc + c * beta - f_grads))
        gamma = gamma * (1 - gamma * epsilon)
        with torch.no_grad():
            beta += gamma * (Bbeta - beta)
            Phi.copy_(library_KS(B, B_x, B_2x, B_3x, B_4x, B_5x, beta, index[0]))
            ## Update theta
            model = fp.prox_method(Phi.cpu().detach().numpy(), ((B_t @ beta).cpu().detach().numpy()).reshape([len(u), ]), mu,
                                   5000, 100000, 1e-12,1)
            model.train(method="FISTA")
            Lambda.copy_(torch.tensor((model.x).reshape([len(Lambda), 1]), dtype=torch.float64))
            # Lambda.copy_(torch.inverse(Phi.T @ Phi) @ Phi.T @ (B_t @ beta))
            para = np.array([Lambda.cpu().detach().numpy()]).reshape([1, len(Lambda)])
            Para = np.concatenate((Para, para), axis=0)

        # Print iteration and residual of each term
        fit_res = 1 / 2 * torch.mean((u - B @ beta) ** 2)
        ic_res = 1 / 2 * torch.mean((ic - B_ic @ beta) ** 2)
        bc_res = 1 / 2 * torch.mean((bc - B_bc @ beta) ** 2)
        pde_res = 1 / 2 * torch.mean((B_t @ beta - Phi @ Lambda) ** 2)
        res = a1*fit_res +a2*ic_res +a3*bc_res + pde_res+ mu / Nc * torch.sum(abs(Lambda))

        r = np.array([fit_res.cpu().detach().numpy(),ic_res.cpu().detach().numpy(),bc_res.cpu().detach().numpy(), pde_res.cpu().detach().numpy(),res.cpu().detach().numpy()]).reshape([1, 5])
        Res = np.concatenate((Res, r), axis=0)

        if i == maxiter - 2:
            break
        else:
            print(gamma.cpu().detach().numpy(), '|', i, '| ', fit_res.cpu().detach().numpy(), ' | ',
                  ic_res.cpu().detach().numpy(), ' | ', bc_res.cpu().detach().numpy(), ' | ',
                  pde_res.cpu().detach().numpy(), ' | ', res.cpu().detach().numpy(), ' | ',
                  ((Lambda[0:3]).cpu().detach().numpy()).reshape([1, 3]))

    print('\n')
    # 下面开始把低于阈值的权重赋为0，并进行BSCA重新进行参数估计
    while len(torch.where(abs(Lambda[:, 0]) < threshold)[0]) != 0:
        Phi, Lambda, indices, beta, ind, Para1, residual = STLASSO_KS(raw_data, raw_data_ic, raw_data_bc, index, u, B,
                                                                   B_t,
                                                                   B_x, B_2x, B_3x, B_4x, B_5x, B_ic, B_bc,
                                                                   Lambda, 0, a1, a2, a3, epsilon, threshold, maxiter,
                                                                   mu)
        Res = np.concatenate((Res, residual), axis=0)
        index.append((indices[0].cpu().detach().numpy()).reshape([len(indices[0]), ]))
        pa = np.zeros([Para1.shape[0], Para.shape[1]])
        for i in range(0, Para1.shape[1]):
            pa[:, ind[i]] = Para1[:, i]
        Para = np.concatenate((Para, pa), axis=0)
        if len(Lambda) == 0:
            break
    # 最小二乘提高参数估计精度
    Phi, Lambda, indices, beta, ind, Para2, residual = STLASSO_KS(raw_data, raw_data_ic, raw_data_bc, index, u, B, B_t,
                                                               B_x,
                                                               B_2x, B_3x, B_4x, B_5x, B_ic, B_bc,
                                                               Lambda, 1, a1, a2, a3, epsilon, threshold, 300, mu)
    Res = np.concatenate((Res, residual), axis=0)
    pa = np.zeros([Para2.shape[0], Para.shape[1]])
    for i in range(0, Para2.shape[1]):
        pa[:, ind[i]] = Para2[:, i]
    Para = np.concatenate((Para, pa), axis=0)
    Residual = Res
    print('\n')
    R_fit = {'residual': ((u - B @ beta) ** 2).cpu().detach().numpy(),
             'x': x0,
             't': t0}
    R_phy = {'residual': ((B_t @ beta - Phi @ Lambda) ** 2).cpu().detach().numpy(),
             'x': x0,
             't': t0}
    return beta, Lambda, R_fit, R_phy, fit_res, pde_res, Residual, Phi, index, Para


def library_KS(B, B_x, B_2x, B_3x, B_4x, B_5x, beta, ind):
    u = (B @ beta)
    u2 = (B @ beta) ** 2
    u3 = (B @ beta) ** 3
    u4 = (B @ beta) ** 4
    u5 = (B @ beta) ** 5
    ux = (B_x @ beta)
    u2x = (B_2x @ beta)
    u3x = (B_3x @ beta)
    u4x = (B_4x @ beta)
    u5x = (B_5x @ beta)
    library = torch.cat((u * ux, u2x, u4x , u3x, u, u2, u3, u4, u5, ux, u5x,
                         u * u2x, u * u3x, u * u4x, u * u5x, u2 * ux,
                         u2 * u2x, u2 * u3x, u2 * u4x, u2 * u5x, u3 * ux, u3 * u2x, u3 * u3x, u3 * u4x, u3 * u5x,
                         u4 * ux,
                         u4 * u2x, u4 * u3x, u4 * u4x, u4 * u5x, u5 * ux, u5 * u2x, u5 * u3x, u5 * u4x, u5 * u5x), 1)
    Phi = library[:, torch.tensor(ind, dtype=torch.int)]
    return Phi

def STLASSO_KS(raw_data,raw_data_ic,raw_data_bc, index, u, B, B_t, B_x, B_2x, B_3x, B_4x, B_5x,B_ic,B_bc, Lambda,  exact,a1,a2,a3,epsilon, threshold, maxiter,mu):
    device = torch.device("cuda")
    U = raw_data['u']
    u = np.random.rand(len(U), 1)
    for i in range(0, len(U)):
        u[i] = U[i]
    u = (torch.tensor(u, dtype=torch.float64)).to(device)

    IC = raw_data_ic['u']
    ic = np.random.rand(len(IC), 1)
    for i in range(0, len(IC)):
        ic[i] = IC[i]
    ic = (torch.tensor(ic, dtype=torch.float64)).to(device)

    BC = raw_data_bc['u']
    bc = np.random.rand(len(BC), 1)
    for i in range(0, len(BC)):
        bc[i] = BC[i]
    bc = (torch.tensor(bc, dtype=torch.float64)).to(device)

    # Operate BSCA
    indices = torch.where(abs(Lambda[:, 0]) > threshold)
    indexx = [index[i] for i in range(0, len(index))]
    indexx.append((indices[0].cpu().detach().numpy()).reshape([len(indices[0]), ]))
    ind = indexx[0]
    for i in range(0, len(indexx)):
        ind = [ind[j] for j in indexx[i]]

    beta = np.random.rand(B.shape[1], 1)
    beta = (torch.tensor(beta, dtype=torch.float64)).to(device)
    F = (library_KS(B, B_x, B_2x, B_3x, B_4x, B_5x, beta, ind)).to(device)
    theta = -np.zeros([F.shape[1], 1])
    theta = (torch.tensor(theta, dtype=torch.float64)).to(device)
    Para = np.array([theta.cpu().detach().numpy()]).reshape([1, len(theta)])

    gamma = (torch.tensor(1, dtype=torch.float64)).to(device)
    beta.requires_grad_()
    c = (torch.tensor(1e-5,dtype=torch.float64)).to(device)
    Nc = B.shape[0]
    Nic = B_ic.shape[0]
    Nbc = B_bc.shape[0]
    # Begin
    ite = 0
    fit_res = 1 / 2 * torch.mean((u - B @ beta) ** 2)
    ic_res = 1 / 2 * torch.mean((ic - B_ic @ beta) ** 2)
    bc_res = 1 / 2 * torch.mean((bc - B_bc @ beta) ** 2)
    pde_res = 1 / 2 * torch.mean((B_t @ beta - F @ theta) ** 2)
    res = a1 * fit_res + a2 * ic_res + a3 * bc_res + pde_res
    print('Stepsize | Iterations | Fitting residual | IC residual | BC residual | PDE residual | Total residual | Theta')

    Res = np.array([fit_res.cpu().detach().numpy(),ic_res.cpu().detach().numpy(),bc_res.cpu().detach().numpy(), pde_res.cpu().detach().numpy(),res.cpu().detach().numpy()]).reshape([1, 5])

    f_grads = autograd.grad(outputs=pde_res, inputs=beta, retain_graph=True)[0]
    I = (torch.eye(B.shape[1])).to(device)
    Bbeta = torch.inverse(a1 / Nc * B.T @ B+a2/Nic* B_ic.T @ B_ic +a3/Nbc* B_bc.T @ B_bc+ c * I) @ (a1 / Nc * B.T @ u + a2/Nic* B_ic.T @ ic +a3/Nbc* B_bc.T @ bc + c * beta - f_grads)

    print(gamma.cpu().detach().numpy(), '|', ite, '| ', fit_res.cpu().detach().numpy(), ' | ',
          ic_res.cpu().detach().numpy(), ' | ', bc_res.cpu().detach().numpy(), ' | ',
          pde_res.cpu().detach().numpy(), ' | ',
          res.cpu().detach().numpy(), ' |', (theta.cpu().detach().numpy()).reshape([1, len(theta)]))

    gamma = gamma * (1 - gamma * epsilon)
    beta = beta + gamma * (Bbeta - beta)
    F = library_KS(B, B_x, B_2x, B_3x, B_4x, B_5x, beta, ind)
    ite = 1

    ## Update theta
    with torch.no_grad():
        if exact==1:
            theta.copy_(torch.inverse(F.T @ F) @ F.T @ (B_t @ beta))
        else:
            model = fp.prox_method(F.cpu().detach().numpy(),
                                   ((B_t @ beta).cpu().detach().numpy()).reshape([len(u), ]), mu,
                                   5000, 100000, 1e-12, 1)
            model.train(method="FISTA")
            theta.copy_(torch.tensor((model.x).reshape([len(theta), 1]), dtype=torch.float64))
        para = np.array([theta.cpu().detach().numpy()]).reshape([1, len(theta)])
        Para = np.concatenate((Para, para), axis=0)
    ic_res = 1 / 2 * torch.mean((ic - B_ic @ beta) ** 2)
    bc_res = 1 / 2 * torch.mean((bc - B_bc @ beta) ** 2)
    pde_res = 1 / 2 * torch.mean((B_t @ beta - F @ theta) ** 2)
    res = a1 * fit_res + a2 * ic_res + a3 * bc_res + pde_res
    print(gamma.cpu().detach().numpy(), '|', ite, '| ', fit_res.cpu().detach().numpy(), ' | ',
          ic_res.cpu().detach().numpy(), ' | ', bc_res.cpu().detach().numpy(), ' | ',
          pde_res.cpu().detach().numpy(), ' | ',
          res.cpu().detach().numpy(), ' |', (theta.cpu().detach().numpy()).reshape([1, len(theta)]))
    r = np.array([fit_res.cpu().detach().numpy(),ic_res.cpu().detach().numpy(),bc_res.cpu().detach().numpy(), pde_res.cpu().detach().numpy(),res.cpu().detach().numpy()]).reshape([1, 5])

    Res = np.concatenate((Res, r), axis=0)

    for i in range(0, maxiter):
        with torch.no_grad():
            f_grads.copy_(autograd.grad(outputs=pde_res, inputs=beta, retain_graph=True)[0])
            Bbeta.copy_(torch.inverse(a1 / Nc * B.T @ B+a2/Nic* B_ic.T @ B_ic +a3/Nbc* B_bc.T @ B_bc+ c * I) @ (a1 / Nc * B.T @ u + a2/Nic* B_ic.T @ ic +a3/Nbc* B_bc.T @ bc + c * beta - f_grads))

        if i == maxiter - 2:
            break
        else:
            with torch.no_grad():
                # gamma = gamma * (1 - gamma * epsilon)
                gamma = gamma * (1 - gamma * epsilon)
                beta += gamma * (Bbeta - beta)
                F.copy_(library_KS(B, B_x, B_2x, B_3x, B_4x, B_5x, beta, ind))
                ## Update theta
                with torch.no_grad():
                    if exact == 1:
                        theta.copy_(torch.inverse(F.T @ F) @ F.T @ (B_t @ beta))
                    else:
                        model = fp.prox_method(F.cpu().detach().numpy(),
                                               ((B_t @ beta).cpu().detach().numpy()).reshape([len(u), ]), mu,
                                               5000, 100000, 1e-12, 1)
                        model.train(method="FISTA")
                        theta.copy_(torch.tensor((model.x).reshape([len(theta), 1]), dtype=torch.float64))

                para = np.array([theta.cpu().detach().numpy()]).reshape([1, len(theta)])
                Para = np.concatenate((Para, para), axis=0)
                # Print iteration and residual of each term
            fit_res = 1 / 2 * torch.mean((u - B @ beta) ** 2)
            ic_res = 1 / 2 * torch.mean((ic - B_ic @ beta) ** 2)
            bc_res = 1 / 2 * torch.mean((bc - B_bc @ beta) ** 2)
            pde_res = 1 / 2 * torch.mean((B_t @ beta - F @ theta) ** 2)
            res = a1 * fit_res + a2 * ic_res + a3 * bc_res + pde_res
            ite += 1
            r = np.array([fit_res.cpu().detach().numpy(),ic_res.cpu().detach().numpy(),bc_res.cpu().detach().numpy(), pde_res.cpu().detach().numpy(),res.cpu().detach().numpy()]).reshape([1, 5])
            Res = np.concatenate((Res, r), axis=0)
            print(gamma.cpu().detach().numpy(), '|', ite, '| ', fit_res.cpu().detach().numpy(), ' | ',
          ic_res.cpu().detach().numpy(), ' | ', bc_res.cpu().detach().numpy(), ' | ',
          pde_res.cpu().detach().numpy(), ' | ',
          res.cpu().detach().numpy(), ' |', (theta.cpu().detach().numpy()).reshape([1, len(theta)]))
    print('\n')
    return F, theta, indices, beta, ind, Para,Res

def exact_line_search_KS_SINDy(B, B_x, B_t, B_2x, B_4x,B_ic,B_bc, u,ic,bc, Bbeta, beta, Lambda, a1,a2,a3):
    nabla = Bbeta - beta
    Nc = B.shape[0]
    A1 = -Lambda[0] * (B @ nabla) * (B_x @ nabla)
    B1 = (B_t @ nabla - Lambda[0] * (B @ beta) * (B_x @ nabla) - Lambda[0] * (B @ nabla) * (
            B_x @ beta) - Lambda[1] * B_2x @ nabla - Lambda[2] * B_4x @ nabla)
    C1 = (B_t @ beta - Lambda[0] * (B @ beta) * (B_x @ beta) - Lambda[1] * B_2x @ beta - Lambda[2] * B_4x @ beta)

    a = (2 / Nc * A1.T @ A1).cpu().detach().numpy()
    b = (3 / 2 / Nc * (A1.T @ B1 + B1.T @ A1)).cpu().detach().numpy()
    c = (1 / Nc * (A1.T @ C1 + B1.T @ B1 + C1.T @ A1) + a1 / Nc * nabla.T @ B.T @ B @ nabla).cpu().detach().numpy()
    d = (1 / 2 / Nc * (B1.T @ C1 + C1.T @ B1) + a1 / 2 * (torch.mean((B@Bbeta-u)**2)-torch.mean((B@beta-u)**2))+a2 / 2 * (torch.mean((B_ic@Bbeta-ic)**2)-torch.mean((B_ic@beta-ic)**2))+a3 / 2 * (torch.mean((B_bc@Bbeta-bc)**2)-torch.mean((B_bc@beta-bc)**2))).cpu().detach().numpy()
    a, b, c, d = a / max(abs(a), abs(b), abs(c), abs(d)), b / max(abs(a), abs(b), abs(c), abs(d)), c / max(abs(a),
                                                                                                           abs(b),
                                                                                                           abs(c),
                                                                                                           abs(d)), d / max(
        abs(a), abs(b), abs(c), abs(d))
    x = solve(a, b, c, d)
    ind = np.isreal(x)
    ii = [i for i in range(0, len(ind)) if ind[i] == True]
    x = x[ii]
    x = x.real
    x = np.unique(x)
    x = x[np.where(x <= 1)]
    x = x[np.where(x >= 0)]
    if len(x) == 0:
        if a / 4 + b / 3 + c / 2 + d < 0:
            x = 1.
        else:
            x = 0.
        gamma = torch.from_numpy(np.array(x))
    elif len(x) == 1:
        gamma = torch.from_numpy(np.array(x))
    else:
        y = a / 4 * x ** 4 + b / 3 * x ** 3 + c / 2 * x ** 2 + d * x
        min_index = np.argmin(y, axis=0)
        x = x[min_index[0]]
        gamma = torch.from_numpy(np.array(x))
    return gamma

def sca_2D_NS(beta,theta,col_data,raw_data,raw_data_ic,raw_data_bc, B_x, B_y, B_x_P, B_y_P, B_xxt_P, B_xxx_P, B_yyt_P, B_yyy_P, B_xxy_P,
              B_xyy_P, B_xxxx_P, B_xxyy_P, B_yyyy_P,B_x_ic,B_y_ic,B_x_bc,B_y_bc,maxiter,a1,a2,a3,a4,a5,a6):
    device=torch.device('cuda')
    U = raw_data['u']
    V = raw_data['v']
    U_ic = raw_data_ic['u']
    V_ic = raw_data_ic['v']
    U_bc = raw_data_bc['u']
    V_bc = raw_data_bc['v']
    t = raw_data['t']
    x = raw_data['x']
    y = raw_data['y']
    tp = col_data['t']
    xp = col_data['x']
    yp = col_data['y']
    # Initialization
    u = np.random.rand(len(U), 1)
    for i in range(0, len(U)):
        u[i] = U[i]
    u = torch.tensor(u, dtype=torch.float64).to(device)

    v = np.random.rand(len(V), 1)
    for i in range(0, len(V)):
        v[i] = V[i]
    v = torch.tensor(v, dtype=torch.float64).to(device)

    u_ic = np.random.rand(len(U_ic), 1)
    for i in range(0, len(U_ic)):
        u_ic[i] = U_ic[i]
    u_ic = torch.tensor(u_ic, dtype=torch.float64).to(device)

    v_ic = np.random.rand(len(V_ic), 1)
    for i in range(0, len(V_ic)):
        v_ic[i] = V_ic[i]
    v_ic = torch.tensor(v_ic, dtype=torch.float64).to(device)

    u_bc = np.random.rand(len(U_bc), 1)
    for i in range(0, len(U_bc)):
        u_bc[i] = U_bc[i]
    u_bc = torch.tensor(u_bc, dtype=torch.float64).to(device)

    v_bc = np.random.rand(len(V_bc), 1)
    for i in range(0, len(V_bc)):
        v_bc[i] = V_bc[i]
    v_bc = torch.tensor(v_bc, dtype=torch.float64).to(device)

    c = torch.tensor(1e-3, dtype=torch.float64)
    Nc = B_x_P.shape[0]
    Nd = B_x.shape[0]
    Nic= B_x_ic.shape[0]
    Nbc = B_x_bc.shape[0]
    # Begin
    ite = 0
    fit_res1 = 1/ 2*torch.mean((u - B_y @ beta) ** 2)
    fit_res2 = 1/ 2*torch.mean((v + B_x @ beta) ** 2)
    ic_res1 = 1/ 2*torch.mean((u_ic - B_y_ic @ beta) ** 2)
    ic_res2 = 1/ 2*torch.mean((v_ic + B_x_ic @ beta) ** 2)
    bc_res1 = 1/ 2*torch.mean((u_bc - B_y_bc @ beta) ** 2)
    bc_res2 = 1/ 2*torch.mean((v_bc + B_x_bc @ beta) ** 2)
    Phi = torch.cat(
        (-(B_y_P @ beta) * (B_xxx_P @ beta + B_xyy_P @ beta), (B_x_P @ beta) * (B_xxy_P @ beta + B_yyy_P @ beta), -(B_xxxx_P @ beta + B_xxyy_P @ beta), -(B_xxyy_P @ beta + B_yyyy_P @ beta)), 1).to(device)
    pde_res = 1/2*torch.mean(((B_xxt_P + B_yyt_P) @ beta - Phi@theta) ** 2)
    res = a1  * fit_res1 + a2 * fit_res2 + a3 * ic_res1 + a4 * ic_res2 + a5 * bc_res1 + a6 * bc_res2 + pde_res

    print('Stepsize | Iterations | Fitting error1 | Fitting error2 | IC error1  | IC error2 | BC error1 | BC error2 | PDE error | Total error | Theta')
    print(0, '|', ite, '| ', fit_res1.cpu().detach().numpy(), ' | ',
          fit_res2.cpu().detach().numpy(),
          ' | ', ic_res1.cpu().detach().numpy(), ' | ',
          ic_res2.cpu().detach().numpy(),
          ' | ', bc_res1.cpu().detach().numpy(), ' | ',
          bc_res2.cpu().detach().numpy(),
          ' | ', pde_res.cpu().detach().numpy(), ' | ', res.cpu().detach().numpy(), ' | ', (theta.reshape([1,len(theta)])).cpu().detach().numpy())
    Para = np.array([theta.cpu().detach().numpy()]).reshape([1, len(theta)])
    Res = np.array([fit_res1.cpu().detach().numpy(), fit_res2.cpu().detach().numpy(),ic_res1.cpu().detach().numpy(), ic_res2.cpu().detach().numpy(), bc_res1.cpu().detach().numpy(),
                    bc_res2.cpu().detach().numpy(), pde_res.cpu().detach().numpy(), res.cpu().detach().numpy()]).reshape([1, 8])

    f_grads = 0.5*autograd.grad(outputs=pde_res, inputs=beta, retain_graph=True)[0]
    I = (torch.eye(B_x_P.shape[1])).to(device)
    Bbeta = torch.inverse(
        c * I + a1 / Nd * B_y.T @ B_y + a2 / Nd * B_x.T @ B_x+a3 / Nic * B_y_ic.T @ B_y_ic + a4 / Nic * B_x_ic.T @ B_x_ic+a5 / Nbc * B_y_bc.T @ B_y_bc + a6 / Nbc * B_x_bc.T @ B_x_bc) @ (
                    a1 / Nd * B_y.T @ u - a2 / Nd * B_x.T @ v +a3 / Nic * B_y_ic.T @ u_ic - a4 / Nic * B_x_ic.T @ v_ic +a5 / Nbc * B_y_bc.T @ u_bc - a6 / Nbc * B_x_bc.T @ v_bc +
                    c * beta - f_grads)
    gamma = (exact_line_search_NS_SINDy(u, v,u_ic, v_ic,u_bc, v_bc, beta, Bbeta, theta, B_x, B_y, B_x_P, B_y_P, B_xxt_P, B_xxx_P, B_yyt_P, B_yyy_P, B_xxy_P,
                         B_xyy_P, B_xxxx_P, B_xxyy_P, B_yyyy_P,B_x_ic,B_y_ic,B_x_bc,B_y_bc, Nc, a1, a2,a3,a4,a5,a6)).to(device)
    # gamma = gamma * (1 - gamma * epsilon)
    if maxiter!=0:
        beta = beta + gamma * (Bbeta - beta)
        ite = 1
        Phi = torch.cat(
            (-(B_y_P @ beta) * (B_xxx_P @ beta + B_xyy_P @ beta), (B_x_P @ beta) * (B_xxy_P @ beta + B_yyy_P @ beta),
             -(B_xxxx_P @ beta + B_xxyy_P @ beta), -(B_xxyy_P @ beta + B_yyyy_P @ beta)), 1).to(device)
        ## Update theta
        with torch.no_grad():
            theta.copy_(torch.inverse(Phi.T @ Phi) @ (Phi.T @ ((B_xxt_P + B_yyt_P) @ beta)))
        pa = np.array([theta.cpu().detach().numpy()]).reshape([1, len(theta)])
        Para = np.concatenate((Para, pa), axis=0)
        fit_res1 = 1 / 2 * torch.mean((u - B_y @ beta) ** 2)
        fit_res2 = 1 / 2 * torch.mean((v + B_x @ beta) ** 2)
        ic_res1 = 1 / 2 * torch.mean((u_ic - B_y_ic @ beta) ** 2)
        ic_res2 = 1 / 2 * torch.mean((v_ic + B_x_ic @ beta) ** 2)
        bc_res1 = 1 / 2 * torch.mean((u_bc - B_y_bc @ beta) ** 2)
        bc_res2 = 1 / 2 * torch.mean((v_bc + B_x_bc @ beta) ** 2)
        pde_res = 1 / 2 * torch.mean(((B_xxt_P + B_yyt_P) @ beta - Phi @ theta) ** 2)
        res = a1  * fit_res1 + a2 * fit_res2 + a3 * ic_res1 + a4 * ic_res2 + a5 * bc_res1 + a6 * bc_res2 + pde_res

        r = np.array([fit_res1.cpu().detach().numpy(), fit_res2.cpu().detach().numpy(),ic_res1.cpu().detach().numpy(), ic_res2.cpu().detach().numpy(),
                      bc_res1.cpu().detach().numpy(), bc_res2.cpu().detach().numpy(), pde_res.cpu().detach().numpy(), res.cpu().detach().numpy()]).reshape([1, 8])
        Res = np.concatenate((Res, r), axis=0)
        print(gamma.cpu().detach().numpy(), '|', ite, '| ', fit_res1.cpu().detach().numpy(), ' | ',
          fit_res2.cpu().detach().numpy(),
          ' | ', ic_res1.cpu().detach().numpy(), ' | ',
          ic_res2.cpu().detach().numpy(),
          ' | ', bc_res1.cpu().detach().numpy(), ' | ',
          bc_res2.cpu().detach().numpy(),
          ' | ', pde_res.cpu().detach().numpy(), ' | ', res.cpu().detach().numpy(), ' | ', (theta.reshape([1,len(theta)])).cpu().detach().numpy())

    for i in range(0, maxiter):
        f = 0.5 * torch.mean(((B_xxt_P + B_yyt_P) @ beta - Phi @ theta) ** 2)
        f_grads.copy_(autograd.grad(outputs=f, inputs=beta, retain_graph=True)[0])
        with torch.no_grad():
            Bbeta.copy_(torch.inverse(
        c * I + a1 / Nd * B_y.T @ B_y + a2 / Nd * B_x.T @ B_x+a3 / Nic * B_y_ic.T @ B_y_ic + a4 / Nic * B_x_ic.T @ B_x_ic+a5 / Nbc * B_y_bc.T @ B_y_bc + a6 / Nbc * B_x_bc.T @ B_x_bc) @ (
                    a1 / Nd * B_y.T @ u - a2 / Nd * B_x.T @ v +a3 / Nic * B_y_ic.T @ u_ic - a4 / Nic * B_x_ic.T @ v_ic +a5 / Nbc * B_y_bc.T @ u_bc - a6 / Nbc * B_x_bc.T @ v_bc +
                    c * beta - f_grads))

        if i == maxiter-1:
            break
        else:
            with torch.no_grad():
                gamma = (exact_line_search_NS_SINDy(u, v,u_ic, v_ic,u_bc, v_bc, beta, Bbeta, theta, B_x, B_y, B_x_P, B_y_P, B_xxt_P, B_xxx_P, B_yyt_P, B_yyy_P, B_xxy_P,
                         B_xyy_P, B_xxxx_P, B_xxyy_P, B_yyyy_P,B_x_ic,B_y_ic,B_x_bc,B_y_bc, Nc, a1, a2,a3,a4,a5,a6)).to(device)
                # gamma = gamma * (1 - gamma * epsilon)
                beta += gamma * (Bbeta - beta)
                Phi.copy_(torch.cat(
                    (-(B_y_P @ beta) * (B_xxx_P @ beta + B_xyy_P @ beta),
                     (B_x_P @ beta) * (B_xxy_P @ beta + B_yyy_P @ beta),
                     -(B_xxxx_P @ beta + B_xxyy_P @ beta), -(B_xxyy_P @ beta + B_yyyy_P @ beta)), 1))
                theta.copy_(torch.inverse(Phi.T @ Phi) @ (Phi.T @ ((B_xxt_P + B_yyt_P) @ beta)))
                pa = np.array([theta.cpu().detach().numpy()]).reshape([1, len(theta)])
                Para = np.concatenate((Para, pa), axis=0)
                # Print iteration and residual of each term
                fit_res1 = 1 / 2 * torch.mean((u - B_y @ beta) ** 2)
                fit_res2 = 1 / 2 * torch.mean((v + B_x @ beta) ** 2)
                ic_res1 = 1 / 2 * torch.mean((u_ic - B_y_ic @ beta) ** 2)
                ic_res2 = 1 / 2 * torch.mean((v_ic + B_x_ic @ beta) ** 2)
                bc_res1 = 1 / 2 * torch.mean((u_bc - B_y_bc @ beta) ** 2)
                bc_res2 = 1 / 2 * torch.mean((v_bc + B_x_bc @ beta) ** 2)
                pde_res = 1/2*torch.mean(((B_xxt_P + B_yyt_P) @ beta - Phi @ theta) ** 2)
                res = a1  * fit_res1 + a2 * fit_res2 + a3 * ic_res1 + a4 * ic_res2 + a5 * bc_res1 + a6 * bc_res2 + pde_res
            r = np.array([fit_res1.cpu().detach().numpy(), fit_res2.cpu().detach().numpy(),ic_res1.cpu().detach().numpy(), ic_res2.cpu().detach().numpy(),
                          bc_res1.cpu().detach().numpy(), bc_res2.cpu().detach().numpy(), pde_res.cpu().detach().numpy(), res.cpu().detach().numpy()]).reshape([1, 8])
            Res = np.concatenate((Res, r), axis=0)
            print(gamma.cpu().detach().numpy(), '|', i+2, '| ', fit_res1.cpu().detach().numpy(), ' | ',
                  fit_res2.cpu().detach().numpy(),
                  ' | ', ic_res1.cpu().detach().numpy(), ' | ',
          ic_res2.cpu().detach().numpy(),
          ' | ', bc_res1.cpu().detach().numpy(), ' | ',
          bc_res2.cpu().detach().numpy(),
          ' | ',pde_res.cpu().detach().numpy(), ' | ', res.cpu().detach().numpy(), ' | ', (theta.reshape([1,len(theta)])).cpu().detach().numpy())
    Residual = Res
    print('\n')
    R_fit1 = {'residual': (((u - B_y @ beta) ** 2).reshape([len(u),])).cpu().detach().numpy(),
              'x': x, 'z': y, 't': t}
    R_fit2 = {'residual': (((v + B_x @ beta) ** 2).reshape([len(u),])).cpu().detach().numpy(),
              'x': x, 'z': y, 't': t}
    R_phy = {'residual': ((((B_xxt_P + B_yyt_P) @ beta - Phi @ theta) ** 2).reshape([len(xp),])).cpu().detach().numpy(),
              'x': xp, 'z': yp, 't': tp}
    return beta, theta, fit_res1, fit_res2, pde_res, Residual,R_fit1,R_fit2,R_phy,Para

def sca_2D_NS_SINDy(raw_data,raw_data_ic,raw_data_bc, B_x,B_y, B_x_P, B_y_P,B_xx_P, B_yy_P,B_xxt_P, B_xxx_P,B_yyt_P, B_yyy_P,
                                                                                                       B_xxy_P,
                                                                                                       B_xyy_P, B_xxxx_P,
                                                                                                       B_xxyy_P, B_yyyy_P,
                                                                                                       B_xxxy_P, B_xyyy_P,B_x_ic,
                                                                                                      B_y_ic, B_x_bc,
                                                                                                      B_y_bc,  mu, beta, Lambda,
                                                                                                       maxiter, epsilon,
                                                                                                       ind, a1, a2, a3,
                                                                                                      a4,a5, a6, gamma):
    device = torch.device("cuda")

    U = raw_data['u']
    V = raw_data['v']
    U_ic = raw_data_ic['u']
    V_ic = raw_data_ic['v']
    U_bc = raw_data_bc['u']
    V_bc = raw_data_bc['v']

    # Initialization
    u = np.random.rand(len(U), 1)
    for i in range(0, len(U)):
        u[i] = U[i]
    u = torch.tensor(u, dtype=torch.float64).to(device)

    v = np.random.rand(len(V), 1)
    for i in range(0, len(V)):
        v[i] = V[i]
    v = torch.tensor(v, dtype=torch.float64).to(device)

    u_ic = np.random.rand(len(U_ic), 1)
    for i in range(0, len(U_ic)):
        u_ic[i] = U_ic[i]
    u_ic = torch.tensor(u_ic, dtype=torch.float64).to(device)

    v_ic = np.random.rand(len(V_ic), 1)
    for i in range(0, len(V_ic)):
        v_ic[i] = V_ic[i]
    v_ic = torch.tensor(v_ic, dtype=torch.float64).to(device)

    u_bc = np.random.rand(len(U_bc), 1)
    for i in range(0, len(U_bc)):
        u_bc[i] = U_bc[i]
    u_bc = torch.tensor(u_bc, dtype=torch.float64).to(device)

    v_bc = np.random.rand(len(V_bc), 1)
    for i in range(0, len(V_bc)):
        v_bc[i] = V_bc[i]
    v_bc = torch.tensor(v_bc, dtype=torch.float64).to(device)
    c = (torch.tensor(1e-3, dtype=torch.float64)).to(device)

    Phi = (library_NS(B_x_P, B_y_P, B_xx_P, B_yy_P, B_xxx_P, B_yyy_P, B_xxy_P, B_xyy_P, B_xxxx_P, B_xxyy_P,
                       B_xxxy_P,
                       B_xyyy_P,
                       B_yyyy_P, beta, ind)).to(device)

    Nc = B_xx_P.shape[0]
    Nd = B_x.shape[0]
    Nic = B_y_ic.shape[0]
    Nbc = B_y_bc.shape[0]
    # Begin
    ite = 0
    fit_res1 = 1 / 2*torch.mean((u - B_y @ beta) ** 2)
    fit_res2 = 1 / 2*torch.mean((v + B_x @ beta) ** 2)
    ic_res1 = 1 / 2* torch.mean((u_ic - B_y_ic @ beta) ** 2)
    ic_res2 = 1 / 2* torch.mean((v_ic + B_x_ic @ beta) ** 2)
    bc_res1 = 1 / 2*torch.mean((u_bc - B_y_bc @ beta) ** 2)
    bc_res2 = 1 / 2*torch.mean((v_bc + B_x_bc @ beta) ** 2)
    pde_res = 1 / 2*torch.mean(((B_xxt_P + B_yyt_P) @ beta - Phi @ Lambda) ** 2)
    res = a1  * fit_res1 + a2 * fit_res2 + a3 * ic_res1 + a4  * ic_res2+a5 * bc_res1 + a6 * bc_res2 + pde_res + mu / Nc * torch.sum(abs(Lambda))

    print('Stepsize | Iterations | Fitting error1 | Fitting error2 | IC error1 | IC error2 | BC error1 | BC error2 | PDE error | Total error | Theta')
    print(0, '|', ite, '| ', fit_res1.cpu().detach().numpy(), ' | ',
          fit_res2.cpu().detach().numpy(),
          ' | ', ic_res1.cpu().detach().numpy(), ' | ',
          ic_res2.cpu().detach().numpy(),
          ' | ', bc_res1.cpu().detach().numpy(), ' | ',
          bc_res2.cpu().detach().numpy(),
          ' | ',  pde_res.cpu().detach().numpy(), ' | ', res.cpu().detach().numpy(), ' | ',
          (Lambda[0:4].cpu().detach().numpy()).reshape([1, 4]))

    Res = np.array([fit_res1.cpu().detach().numpy(), fit_res2.cpu().detach().numpy(),ic_res1.cpu().detach().numpy(), ic_res2.cpu().detach().numpy(), bc_res1.cpu().detach().numpy(), bc_res2.cpu().detach().numpy(), pde_res.cpu().detach().numpy(),res.cpu().detach().numpy()]).reshape([1, 8])
    Para = np.array([Lambda.cpu().detach().numpy()]).reshape([1, len(Lambda)])

    f = 0.5 * torch.mean(((B_xxt_P + B_yyt_P) @ beta - Phi @ Lambda) ** 2)
    f_grads = autograd.grad(outputs=f, inputs=beta, retain_graph=True)[0]
    I = (torch.eye(B_x_P.shape[1])).to(device)

    Bbeta = torch.inverse(
        c * I + a1 / Nd * B_y.T @ B_y + a2 / Nd * B_x.T @ B_x + a3 / Nic * B_y_ic.T @ B_y_ic + a4 / Nic * B_x_ic.T @ B_x_ic + a5 / Nbc * B_y_bc.T @ B_y_bc + a6 / Nbc * B_x_bc.T @ B_x_bc) @ (
                    a1 / Nd * B_y.T @ u - a2 / Nd * B_x.T @ v + a3 / Nic * B_y_ic.T @ u_ic - a4 / Nic * B_x_ic.T @ v_ic + a5 / Nbc * B_y_bc.T @ u_bc - a6 / Nbc * B_x_bc.T @ v_bc +
                    c * beta - f_grads)

    beta = beta + gamma * (Bbeta - beta)

    Phi.copy_((library_NS(B_x_P, B_y_P, B_xx_P, B_yy_P, B_xxx_P, B_yyy_P, B_xxy_P, B_xyy_P, B_xxxx_P, B_xxyy_P, B_xxxy_P,
                     B_xyyy_P,
                     B_yyyy_P, beta, ind)).to(device))
    ite = 1
    ## Update theta
    with torch.no_grad():
        model = fp.prox_method(Phi.cpu().detach().numpy(), ((B_xxt_P + B_yyt_P) @ beta).cpu().detach().numpy().reshape([B_xxt_P.shape[0], ]),
                               mu, 10000, 100000, 1e-12, 1)
        model.train(method="FISTA")
        Lambda.copy_(torch.tensor((model.x).reshape([len(Lambda), 1]), dtype=torch.float64))
        pa = np.array([Lambda.cpu().detach().numpy()]).reshape([1, len(Lambda)])
        Para = np.concatenate((Para, pa), axis=0)

        # Lambda.copy_(
        #     torch.inverse(Phi.T @ Phi) @ (Phi.T @ ((B_xxt_P + B_yyt_P) @ beta)))

    fit_res1 = 1 / 2 * torch.mean((u - B_y @ beta) ** 2)
    fit_res2 = 1 / 2 * torch.mean((v + B_x @ beta) ** 2)
    ic_res1 = 1 / 2 * torch.mean((u_ic - B_y_ic @ beta) ** 2)
    ic_res2 = 1 / 2 * torch.mean((v_ic + B_x_ic @ beta) ** 2)
    bc_res1 = 1 / 2 * torch.mean((u_bc - B_y_bc @ beta) ** 2)
    bc_res2 = 1 / 2 * torch.mean((v_bc + B_x_bc @ beta) ** 2)
    pde_res = 1 / 2 * torch.mean(((B_xxt_P + B_yyt_P) @ beta - Phi @ Lambda) ** 2)
    res = a1 * fit_res1 + a2 * fit_res2 + a3 * ic_res1 + a4 * ic_res2 + a5 * bc_res1 + a6 * bc_res2 + pde_res + mu / Nc * torch.sum(abs(Lambda))
    r = np.array([fit_res1.cpu().detach().numpy(), fit_res2.cpu().detach().numpy(),ic_res1.cpu().detach().numpy(), ic_res2.cpu().detach().numpy(), bc_res1.cpu().detach().numpy(), bc_res2.cpu().detach().numpy(), pde_res.cpu().detach().numpy(),res.cpu().detach().numpy()]).reshape([1, 8])
    Res=np.concatenate((Res, r), axis=0)
    print(gamma.cpu().detach().numpy(), '|', ite, '| ', fit_res1.cpu().detach().numpy(), ' | ',
          fit_res2.cpu().detach().numpy(),
          ' | ', ic_res1.cpu().detach().numpy(), ' | ',
          ic_res2.cpu().detach().numpy(),
          ' | ', bc_res1.cpu().detach().numpy(), ' | ',
          bc_res2.cpu().detach().numpy(),
          ' | ', pde_res.cpu().detach().numpy(), ' | ', res.cpu().detach().numpy(), ' | ',
          (Lambda[0:4].cpu().detach().numpy()).reshape([1, 4]))
    gamma = gamma * (1 - gamma * epsilon)
    for i in range(0, maxiter):
        f = 0.5 * torch.mean(((B_xxt_P + B_yyt_P) @ beta - Phi @ Lambda) ** 2)
        f_grads.copy_(autograd.grad(outputs=f, inputs=beta, retain_graph=True)[0])

        with torch.no_grad():
            Bbeta.copy_(torch.inverse(
                c * I + a1 / Nd * B_y.T @ B_y + a2 / Nd * B_x.T @ B_x + a3 / Nic * B_y_ic.T @ B_y_ic + a4 / Nic * B_x_ic.T @ B_x_ic + a5 / Nbc * B_y_bc.T @ B_y_bc + a6 / Nbc * B_x_bc.T @ B_x_bc) @ (
                                a1 / Nd * B_y.T @ u - a2 / Nd * B_x.T @ v + a3 / Nic * B_y_ic.T @ u_ic - a4 / Nic * B_x_ic.T @ v_ic + a5 / Nbc * B_y_bc.T @ u_bc - a6 / Nbc * B_x_bc.T @ v_bc +
                                c * beta - f_grads))

        if i == maxiter - 1:
            break
        else:
            with torch.no_grad():
                beta += gamma * (Bbeta - beta)

                Phi.copy_(
                    library_NS(B_x_P, B_y_P, B_xx_P, B_yy_P, B_xxx_P, B_yyy_P, B_xxy_P, B_xyy_P, B_xxxx_P, B_xxyy_P,
                               B_xxxy_P,
                               B_xyyy_P,
                               B_yyyy_P, beta, ind))
                model = fp.prox_method(Phi.cpu().detach().numpy(),
                                       ((B_xxt_P + B_yyt_P) @ beta).cpu().detach().numpy().reshape([B_xxt_P.shape[0], ]), mu, 10000,
                                       100000, 1e-12, 1)
                model.train(method="FISTA")
                Lambda.copy_(torch.tensor((model.x).reshape([len(Lambda), 1]), dtype=torch.float64))
                pa = np.array([Lambda.cpu().detach().numpy()]).reshape([1, len(Lambda)])
                Para = np.concatenate((Para, pa), axis=0)
                # Lambda.copy_(
                #     torch.inverse(Phi.T @ Phi) @ (Phi.T @ ((B_xxt_P + B_yyt_P) @ beta)))
            # Print iteration and residual of each term
            fit_res1 = 1 / 2 * torch.mean((u - B_y @ beta) ** 2)
            fit_res2 = 1 / 2 * torch.mean((v + B_x @ beta) ** 2)
            ic_res1 = 1 / 2 * torch.mean((u_ic - B_y_ic @ beta) ** 2)
            ic_res2 = 1 / 2 * torch.mean((v_ic + B_x_ic @ beta) ** 2)
            bc_res1 = 1 / 2 * torch.mean((u_bc - B_y_bc @ beta) ** 2)
            bc_res2 = 1 / 2 * torch.mean((v_bc + B_x_bc @ beta) ** 2)
            pde_res = 1 / 2 * torch.mean(((B_xxt_P + B_yyt_P) @ beta - Phi @ Lambda) ** 2)
            res = a1 * fit_res1 + a2 * fit_res2 + a3 * ic_res1 + a4 * ic_res2 + a5 * bc_res1 + a6 * bc_res2 + pde_res + mu / Nc * torch.sum(abs(Lambda))

            r = np.array([fit_res1.cpu().detach().numpy(), fit_res2.cpu().detach().numpy(),ic_res1.cpu().detach().numpy(), ic_res2.cpu().detach().numpy(), bc_res1.cpu().detach().numpy(), bc_res2.cpu().detach().numpy(), pde_res.cpu().detach().numpy(),res.cpu().detach().numpy()]).reshape([1, 8])
            Res=np.concatenate((Res, r), axis=0)
            print(gamma.cpu().detach().numpy(), '|', i+2, '| ', fit_res1.cpu().detach().numpy(), ' | ',
                  fit_res2.cpu().detach().numpy(),
                  ' | ', ic_res1.cpu().detach().numpy(), ' | ',
                  ic_res2.cpu().detach().numpy(),
                  ' | ', bc_res1.cpu().detach().numpy(), ' | ',
                  bc_res2.cpu().detach().numpy(),
                  ' | ', pde_res.cpu().detach().numpy(), ' | ', res.cpu().detach().numpy(), ' | ',
                  (Lambda[0:4].cpu().detach().numpy()).reshape([1, 4]))
            gamma = gamma * (1 - gamma * epsilon)
    Residual = Res
    print('\n')

    return beta, Lambda, fit_res1, fit_res2, pde_res, Residual,Para, gamma

def exact_line_search_NS_SINDy(u, v,u_ic, v_ic,u_bc, v_bc, beta, Bbeta, theta, B_x, B_y, B_x_P, B_y_P, B_xxt_P, B_xxx_P, B_yyt_P,
                               B_yyy_P,
                               B_xxy_P, B_xyy_P, B_xxxx_P, B_xxyy_P, B_yyyy_P,B_x_ic,B_y_ic,B_x_bc,B_y_bc, Nc, a1, a2,a3,a4,a5,a6):
    nabla = Bbeta - beta
    A1 = (theta[1]*(B_x_P @ nabla) * ((B_xxy_P + B_yyy_P) @ nabla) - theta[0]*(B_y_P @ nabla) * ((B_xxx_P + B_xyy_P) @ nabla))
    B1 = theta[1]*(B_x_P @ beta) * ((B_xxy_P + B_yyy_P) @ nabla) + theta[1]*(B_x_P @ nabla) * ((B_xxy_P + B_yyy_P) @ beta) + theta[2] * (
            B_xxxx_P + B_xxyy_P) @ nabla + theta[3] * (B_xxyy_P + B_yyyy_P) @ nabla - theta[0]*(B_y_P @ beta) * ((B_xxx_P + B_xyy_P) @ nabla) - theta[0]*(
                 B_y_P @ nabla) * ((B_xxx_P + B_xyy_P) @ beta) - (B_xxt_P + B_yyt_P) @ nabla
    C1 = theta[1]*(B_x_P @ beta) * ((B_xxy_P + B_yyy_P) @ beta) + theta[2] * (B_xxxx_P + B_xxyy_P) @ beta + theta[3] * (B_xxyy_P + B_yyyy_P) @ beta- theta[0]*(
            B_y_P @ beta) * ((B_xxx_P + B_xyy_P) @ beta) - (B_xxt_P + B_yyt_P) @ beta
    a = (1 / 2 / Nc * (4 * A1.T @ A1)).cpu().detach().numpy()
    b = (1 / 2 / Nc * 3 * (A1.T @ B1 + B1.T @ A1)).cpu().detach().numpy()
    c = (1 / 2 / Nc * 2 * (A1.T @ C1 + B1.T @ B1 + C1.T @ A1)).cpu().detach().numpy()
    d = (1 / 2 / Nc * (B1.T @ C1 + C1.T @ B1)).cpu().detach().numpy() + (a1 / 2 * (torch.mean(
        (u - B_y @ Bbeta) ** 2) - torch.mean(
        (u - B_y @ beta) ** 2)) + a2 / 2 * (torch.mean((v + B_x @ Bbeta) ** 2) - torch.mean(
        (v + B_x @ beta) ** 2))).cpu().detach().numpy()+ (a3 / 2 * (torch.mean(
        (u_ic - B_y_ic @ Bbeta) ** 2) - torch.mean(
        (u_ic - B_y_ic @ beta) ** 2)) + a4 / 2 * (torch.mean((v_ic + B_x_ic @ Bbeta) ** 2) - torch.mean(
        (v_ic + B_x_ic @ beta) ** 2))).cpu().detach().numpy()+ (a5 / 2 * (torch.mean(
        (u_bc - B_y_bc @ Bbeta) ** 2) - torch.mean(
        (u_bc - B_y_bc @ beta) ** 2)) + a6 / 2 * (torch.mean((v_bc + B_x_bc @ Bbeta) ** 2) - torch.mean(
        (v_bc + B_x_bc @ beta) ** 2))).cpu().detach().numpy()
    a, b, c, d = a / max(abs(a), abs(b), abs(c), abs(d)), b / max(abs(a), abs(b), abs(c), abs(d)), c / max(abs(a),
                                                                                                           abs(b),
                                                                                                           abs(c),
                                                                                                           abs(d)), d / max(
        abs(a), abs(b), abs(c), abs(d))
    x = solve(a, b, c, d)
    ind = np.isreal(x)
    ii = [i for i in range(0, len(ind)) if ind[i] == True]
    x = x[ii]
    x = x.real
    x = np.unique(x)
    x = x[np.where(x <= 1)]
    x = x[np.where(x >= 0)]
    if len(x) == 0:
        if a / 4 + b / 3 + c / 2 + d < 0:
            x = 1.
        else:
            x = 0.
        gamma = torch.from_numpy(np.array(x))
    elif len(x) == 1:
        gamma = torch.from_numpy(np.array(x))
    else:
        y = a / 4 * x ** 4 + b / 3 * x ** 3 + c / 2 * x ** 2 + d * x
        min_index = np.argmin(y, axis=0)
        x = x[min_index[0]]
        gamma = torch.from_numpy(np.array(x))
    return gamma

def library_NS(B_x_P, B_y_P, B_xx_P, B_yy_P, B_xxx_P, B_yyy_P, B_xxy_P, B_xyy_P, B_xxxx_P, B_xxyy_P, B_xxxy_P, B_xyyy_P,
               B_yyyy_P, beta, ind):
    u = (B_y_P @ beta)
    v = -(B_x_P @ beta)
    w = -(B_xx_P @ beta + B_yy_P @ beta)

    uv = u * v
    uw = u * w
    vw = v * w

    u2 = u ** 2
    v2 = v ** 2
    w2 = w ** 2

    wx = -(B_xxx_P @ beta + B_xyy_P @ beta)
    wy = -(B_xxy_P @ beta + B_yyy_P @ beta)
    wxx = -(B_xxxx_P @ beta + B_xxyy_P @ beta)
    wyy = -(B_xxyy_P @ beta + B_yyyy_P @ beta)
    wxy = -(B_xxxy_P @ beta + B_xyyy_P @ beta)

    library = torch.cat(
        (u * wx, v * wy, wxx, wyy, u, v, uv, uw, vw, u2, v2, w2, u * wy, u * wxx, u * wxy,
         v * wx, v * wxx, v * wxy,w * wx, w * wy,
         uv * wx, uv * wy, uv * wxx, uv * wxy, uw * wx, uw * wy, uw * wxx, uw * wxy,
         vw * wx, vw * wy, vw * wxy, u2 * wx, u2 * wy, u2 * wxx, u2 * wxy,
         v2 * wx, v2 * wy, v2 * wxx, v2 * wxy, w2 * wxy), 1)

    Phi = library[:, torch.tensor(ind, dtype=torch.int)]
    return Phi


def fit_data_NS(raw_data, B_x_P, B_y_P,beta,lr,a, maxiter):
    device = torch.device("cuda")
    t = raw_data['t']
    x = raw_data['x']
    y = raw_data['y']
    U = raw_data['u']
    V = raw_data['v']

    # Initialization
    u = np.random.rand(len(U), 1)
    for i in range(0, len(U)):
        u[i] = U[i]
    u = (torch.tensor(u, dtype=torch.float64)).to(device)

    v = np.random.rand(len(V), 1)
    for i in range(0, len(V)):
        v[i] = V[i]
    v = (torch.tensor(v, dtype=torch.float64)).to(device)

    # Begin
    ite = 0
    fit_res1 = torch.mean((u - B_y_P @ beta) ** 2).cpu()
    fit_res2 = torch.mean((v + B_x_P @ beta) ** 2).cpu()

    print('Stepsize | Iterations | Fitting error1 | Fitting error2')
    print(lr.cpu().detach().numpy(), '|', ite, '| ', fit_res1.detach().numpy(), ' | ',
          fit_res2.detach().numpy())
    cov1=B_y_P.T @ B_y_P
    cov2=B_x_P.T @ B_x_P

    for ite in range(0, maxiter):
        grad = a * (cov1 @ beta - B_y_P.T @ u) + cov2 @ beta + B_x_P.T @ v
        beta = beta - lr * grad
        # Print iteration and residual of each term
        fit_res1 = torch.mean((u - B_y_P @ beta) ** 2).cpu()
        fit_res2 = torch.mean((v + B_x_P @ beta) ** 2).cpu()

    print(lr.cpu().detach().numpy(), '|', ite, '| ', fit_res1.detach().numpy(), ' | ',
                  fit_res2.detach().numpy())
    R_fit1 = {'residual': (((u - B_y_P @ beta) ** 2).reshape([len(u),])).cpu().detach().numpy(),
              'x': x, 'z': y, 't': t}
    R_fit2 = {'residual': (((v + B_x_P @ beta) ** 2).reshape([len(u),])).cpu().detach().numpy(),
              'x': x, 'z': y, 't': t}
    print('\n')
    return R_fit1,R_fit2,beta

def fit_data_KS(raw_data, B):
    device = torch.device("cuda")
    t = raw_data['t']
    x = raw_data['x']
    U = raw_data['u']

    # Initialization
    beta = np.ones([B.shape[1], 1])
    beta = (torch.tensor(beta, dtype=torch.float64)).to(device)
    beta.requires_grad_()

    u = np.random.rand(len(U), 1)
    for i in range(0, len(U)):
        u[i] = U[i]
    u = (torch.tensor(u, dtype=torch.float64)).to(device)

    # Begin
    ite = 0
    fit_res = torch.mean((u - B @ beta) ** 2).cpu()
    gamma = torch.tensor(0.2, dtype=torch.float64)
    print('Stepsize | Iterations | Fitting error')
    print(gamma.cpu().detach().numpy(), '|', ite, '| ', fit_res.detach().numpy())
    cov=B.T @ B
    maxiter = 5000
    for i in range(0, maxiter):
        grad =  cov @ beta - B.T @ u
        beta = beta - gamma * grad
        # Print iteration and residual of each term
        fit_res = torch.mean((u - B @ beta) ** 2).cpu()

    print(gamma.cpu().detach().numpy(), '|', i, '| ', fit_res.detach().numpy())
    R_fit = {'residual': ((u - B @ beta) ** 2).cpu().detach().numpy(),
              'x': x, 't': t}
    print('\n')
    return R_fit,beta


def fit_data_LNN(q,qt,qtt,q_true, qt_true, qtt_true, t, B, B_d,B_2d,a1,a2):
    beta = np.linalg.inv(a1*B.T@B+a2*B_d.T@B_d+B_2d.T@B_2d)@(a1*B.T@q+a2*B_d.T@qt+B_2d.T@qtt)
    print('Fitting error1 | Fitting error2 | Fitting error3')
    # Print iteration and residual of each term
    fit_res1 = np.mean((q_true - B @ beta) ** 2)
    fit_res2 = np.mean((qt_true - B_d @ beta) ** 2)
    fit_res3 = np.mean((qtt_true - B_2d @ beta) ** 2)
    print( fit_res1, ' | ', fit_res2, ' | ', fit_res3)
    R_fit1 = {'residual': ((q - B @ beta) ** 2), 't': t}
    R_fit2 = {'residual': ((qt - B_d @ beta) ** 2), 't': t}
    R_fit3 = {'residual': ((qtt - B_2d @ beta) ** 2), 't': t}
    print('\n')
    return R_fit1,R_fit2,R_fit3,beta