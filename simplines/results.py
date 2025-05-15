from numpy import empty
import numpy as np

# ==========================================================
def find_span( knots, degree, x ):
    # Knot index at left/right boundary
    low  = degree
    high = 0
    high = len(knots)-1-degree

    # Check if point is exactly on left/right boundary, or outside domain
    if x <= knots[low ]: returnVal = low
    elif x >= knots[high]: returnVal = high-1
    else:
        # Perform binary search
        span = (low+high)//2
        while x < knots[span] or x >= knots[span+1]:
            if x < knots[span]:
                high = span
            else:
                low  = span
            span = (low+high)//2
        returnVal = span

    return returnVal
# ==========================================================
def all_bsplines( knots, degree, x, span ):
    left   = empty( degree  , dtype=float )
    right  = empty( degree  , dtype=float )
    values = empty( degree+1, dtype=float )

    values[0] = 1.0
    for j in range(0,degree):
        left [j] = x - knots[span-j]
        right[j] = knots[span+1+j] - x
        saved    = 0.0
        for r in range(0,j+1):
            temp      = values[r] / (right[r] + left[j-r])
            values[r] = saved + right[r] * temp
            saved     = left[j-r] * temp
        values[j+1] = saved

    return values

def basis_funs_all_ders( knots, degree, x, span, n ):
    """
    Evaluate value and n derivatives at x of all basis functions with
    support in interval [x_{span-1}, x_{span}].

    ders[i,j] = (d/dx)^i B_k(x) with k=(span-degree+j),
                for 0 <= i <= n and 0 <= j <= degree+1.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    x : float
        Evaluation point.

    span : int
        Knot span index.

    n : int
        Max derivative of interest.

    Results
    -------
    ders : numpy.ndarray (n+1,degree+1)
        2D array of n+1 (from 0-th to n-th) derivatives at x of all (degree+1)
        non-vanishing basis functions in given span.

    Notes
    -----
    The original Algorithm A2.3 in The NURBS Book [1] is here improved:
        - 'left' and 'right' arrays are 1 element shorter;
        - inverse of knot differences are saved to avoid unnecessary divisions;
        - innermost loops are replaced with vector operations on slices.

    """
    left  = np.empty( degree )
    right = np.empty( degree )
    ndu   = np.empty( (degree+1, degree+1) )
    a     = np.empty( (       2, degree+1) )
    ders  = np.zeros( (     n+1, degree+1) ) # output array

    # Number of derivatives that need to be effectively computed
    # Derivatives higher than degree are = 0.
    ne = min( n, degree )

    # Compute nonzero basis functions and knot differences for splines
    # up to degree, which are needed to compute derivatives.
    # Store values in 2D temporary array 'ndu' (square matrix).
    ndu[0,0] = 1.0
    for j in range(0,degree):
        left [j] = x - knots[span-j]
        right[j] = knots[span+1+j] - x
        saved    = 0.0
        for r in range(0,j+1):
            # compute inverse of knot differences and save them into lower triangular part of ndu
            ndu[j+1,r] = 1.0 / (right[r] + left[j-r])
            # compute basis functions and save them into upper triangular part of ndu
            temp       = ndu[r,j] * ndu[j+1,r]
            ndu[r,j+1] = saved + right[r] * temp
            saved      = left[j-r] * temp
        ndu[j+1,j+1] = saved

    # Compute derivatives in 2D output array 'ders'
    ders[0,:] = ndu[:,degree]
    for r in range(0,degree+1):
        s1 = 0
        s2 = 1
        a[0,0] = 1.0
        for k in range(1,ne+1):
            d  = 0.0
            rk = r-k
            pk = degree-k
            if r >= k:
               a[s2,0] = a[s1,0] * ndu[pk+1,rk]
               d = a[s2,0] * ndu[rk,pk]
            j1 = 1   if (rk  > -1 ) else -rk
            j2 = k-1 if (r-1 <= pk) else degree-r
            a[s2,j1:j2+1] = (a[s1,j1:j2+1] - a[s1,j1-1:j2]) * ndu[pk+1,rk+j1:rk+j2+1]
            d += np.dot( a[s2,j1:j2+1], ndu[rk+j1:rk+j2+1,pk] )
            if r <= pk:
               a[s2,k] = - a[s1,k-1] * ndu[pk+1,r]
               d += a[s2,k] * ndu[r,pk]
            ders[k,r] = d
            j  = s1
            s1 = s2
            s2 = j

    # Multiply derivatives by correct factors
    r = degree
    for k in range(1,ne+1):
        ders[k,:] = ders[k,:] * r
        r = r * (degree-k)

    return ders

# ==========================================================
def point_on_bspline_surface(Tu, Tv, P, u, v):
    pu = len(Tu) - P.shape[0] - 1
    pv = len(Tv) - P.shape[1] - 1
    d = P.shape[-1]

    span_u = find_span( Tu, pu, u )
    span_v = find_span( Tv, pv, v )

    basis_x =basis_funs_all_ders( Tu, pu, u, span_u, 1 )
    basis_y =basis_funs_all_ders( Tv, pv, v, span_v, 1 )

    bu   = basis_x[0,:]
    bv   = basis_y[0,:]
    
    derbu   = basis_x[1,:]
    derbv   = basis_y[1,:]
    c = np.zeros(d)
    cx = np.zeros(d)
    cy = np.zeros(d)
    for ku in range(0, pu+1):
        for kv in range(0, pv+1):
            c[:] += bu[ku]*bv[kv]*P[span_u-pu+ku, span_v-pv+kv,:]
            cx[:] += derbu[ku]*bv[kv]*P[span_u-pu+ku, span_v-pv+kv,:]
            cy[:] += bu[ku]*derbv[kv]*P[span_u-pu+ku, span_v-pv+kv,:]

    return c, cx, cy

from numpy import zeros, linspace, meshgrid, asarray
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def sol_field_2d(Npoints,  uh , knots, degree):
    # Using computed control points U we compute solution
    # in new discretisation by Npoints 

    pu, pv = degree
    nx, ny = Npoints
    Tu, Tv = knots

    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1

    xs = linspace(0., 1., nx)
    
    ys = linspace(0., 1., ny)
    P = zeros((nu, nv,1))
    
    i = list(range(nu))
    for j in range(nv):
        P[i, j, 0] = uh[i,j]    

    Q  = zeros((nx, ny, 3))
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            Q[i,j,:]   = point_on_bspline_surface(Tu, Tv, P, x, y)
    X, Y = meshgrid(xs, ys)

    return Q[:,:,0], Q[:,:,1], Q[:,:,2], X, Y
