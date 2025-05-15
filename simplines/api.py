import numpy as np
from functools import partial

from pyccel.epyccel import epyccel

from .linalg import StencilMatrix
from .linalg import StencilVector
from .spaces import TensorSpace

__all__ = ['assemble_matrix', 'assemble_vector', 'assemble_scalar', 'compile_kernel', 'apply_dirichlet']

#==============================================================================
def assemble_matrix(core, V, fields=None, out=None):
    if out is None:
        out = StencilMatrix(V.vector_space, V.vector_space)

    # ...
    args = []
    if isinstance(V, TensorSpace):
        args += list(V.nelements)
        args += list(V.degree)
        args += list(V.spans)
        args += list(V.basis)
        args += list(V.weights)
        args += list(V.points)

    else:
        args = [V.nelements,
                V.degree,
                V.spans,
                V.basis,
                V.weights,
                V.points]
    # ...

    if not(fields is None):
        assert(isinstance(fields, (list, tuple)))

        args += [u._data for u in fields]

    core( *args, out._data )

    return out

#==============================================================================
def assemble_vector(core, V, fields=None, values = None, out=None):
    if out is None:
        out = StencilVector(V.vector_space)

    # ...
    args = []
    if isinstance(V, TensorSpace):
        args += list(V.nelements)
        args += list(V.degree)
        args += list(V.spans)
        args += list(V.basis)
        args += list(V.weights)
        args += list(V.points)

    else:
        args = [V.nelements,
                V.degree,
                V.spans,
                V.basis,
                V.weights,
                V.points]
    # ...

    if not(fields is None):
        assert(isinstance(fields, (list, tuple)))

        args += [x._data for x in fields]

    if not(values is None):
        args += [x for x in values]

    core( *args, out._data )

    return out

#==============================================================================
def assemble_scalar(core, V, fields=None, values = None):
    # ...
    args = []
    if isinstance(V, TensorSpace):
        args += list(V.nelements)
        args += list(V.degree)
        args += list(V.spans)
        args += list(V.basis)
        args += list(V.weights)
        args += list(V.points)

    else:
        args = [V.nelements,
                V.degree,
                V.spans,
                V.basis,
                V.weights,
                V.points]
    # ...

    if not(fields is None):
        assert(isinstance(fields, (list, tuple)))

        args += [x._data for x in fields]
    
    if not(values is None):
        args += [x for x in values]
    return core( *args )

#==============================================================================
def compile_kernel(core, arity, pyccel=True):
    assert(arity in [0,1,2])

    if pyccel:
        core = epyccel(core)

    if arity == 2:
        return partial(assemble_matrix, core)

    elif arity == 1:
        return partial(assemble_vector, core)

    elif arity == 0:
        return partial(assemble_scalar, core)


#==============================================================================
def apply_dirichlet(V, x):
    if isinstance(x, StencilMatrix):
        if V.dim == 1:
            n1 = V.nbasis

            # ... resetting bnd dof to 0
            x[0,:] = 0.
            x[n1-1,:] = 0.
            # ...

            # boundary x = 0
            x[0,0] = 1.

            # boundary x = 1
            x[n1-1,0] = 1.

            return x

        elif V.dim == 2:
            n1,n2 = V.nbasis

            # ... resetting bnd dof to 0
            x[0,:,:,:] = 0.
            x[n1-1,:,:,:] = 0.
            x[:,0,:,:] = 0.
            x[:,n2-1,:,:] = 0.
            # ...

            # boundary x = 0
            x[0,:,0,0] = 1.

            # boundary x = 1
            x[n1-1,:,0,0] = 1.

            # boundary y = 0
            x[:,0,0,0] = 1.

            # boundary y = 1
            x[:,n2-1,0,0] = 1.

            return x

        elif V.dim == 3:
            n1,n2,n3 = V.nbasis

            # ... resetting bnd dof to 0
            x[0,:,:,:,:,:] = 0.
            x[n1-1,:,:,:,:,:] = 0.
            x[:,0,:,:,:,:] = 0.
            x[:,n2-1,:,:,:,:] = 0.
            x[:,:,0,:,:,:] = 0.
            x[:,:,n3-1,:,:,:] = 0.
            # ...

            # boundary x = 0
            x[0,:,:, 0,0,0] = 1.

            # boundary x = 1
            x[n1-1,:,:, 0,0,0] = 1.

            # boundary y = 0
            x[:,0,:, 0,0,0] = 1.

            # boundary y = 1
            x[:,n2-1,:, 0,0,0] = 1.

            # boundary z = 0
            x[:,:,0, 0,0,0] = 1.

            # boundary z = 1
            x[:,:,n3-1, 0,0,0] = 1.

            return x
        else :
            raise NotImplementedError('Only 1d, 2d and 3d are available')

    elif isinstance(x, StencilVector):
        if V.dim == 1:
            n1 = V.nbasis

            x[0] = 0.
            x[n1-1] = 0.

            return x

        elif V.dim == 2:
            n1,n2 = V.nbasis

            x[0,:] = 0.
            x[n1-1,:] = 0.
            x[:,0] = 0.
            x[:,n2-1] = 0.

            return x

        elif V.dim == 3:
            n1, n2, n3 = V.nbasis

            x[0,:,:] = 0.
            x[n1-1,:,:] = 0.

            x[:,0,:] = 0.
            x[:,n2-1,:] = 0.

            x[:,:,0] = 0.
            x[:,:,n3-1] = 0.

            return x

        else:
            raise NotImplementedError('Only 1d, 2d and 3d are available')

    else:
        raise TypeError('Expecting StencilMatrix or StencilVector')
