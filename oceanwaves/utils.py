import numpy as np


def iterable(arr):
    try:
        iter(arr)
        return arr
    except:
        return (arr,)


def expand_and_repeat(mtx, lengths, dims, inverse=False,
                      inclusive=False):
    
    mtx = np.asarray(mtx)
    lengths = iterable(lengths)
    dims = iterable(dims)

    inverse = True if inclusive else inverse
    
    if len(dims) != len(lengths) and not inverse:
        raise ValueError('Length of dimension (%d) and length (%d) arrays '
                         'should match.' % (len(dims), len(lengths)))
    if len(dims) != mtx.ndim and inverse:
        raise ValueError('Length of dimension (%d) and matrix shape (%d) arrays '
                         'should match.' % (len(dims), mtx.ndim))
        
    dims0 = dims
    if inclusive:
        dims = [x
                for x in range(len(lengths))
                if x not in dims0]
        lengths = list(np.asarray(lengths)[dims])
    elif inverse:
        dims = [x
                for x in range(len(lengths) + len(dims))
                if x not in dims0]
        
    n = 0
    shp = mtx.shape
    ndim = mtx.ndim + len(dims)
    for i in range(ndim):
        if i in dims:
            l = lengths[dims.index(i)]
            mtx = np.expand_dims(mtx, i).repeat(l, axis=i)
        elif len(shp) > n:
            l = shp[n]
            n += 1
        else:
            raise ValueError('Dimension %d is missing.' % i)

    return mtx


