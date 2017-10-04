import numpy as np


def iterable(arr):
    '''Returns an iterable'''
    
    try:
        iter(arr)
        return arr
    except:
        return (arr,)


def expand_and_repeat(mtx, shape=None, repeat=None,
                      exist_dims=None, expand_dims=None):
    '''Expands matrix and repeats matrix contents along new dimensions

    Provide ``shape`` and ``exist_dims`` or ``expand_dims``, or
    ``repeat`` and ``expand_dims``.

    Parameters
    ----------
    mtx : numpy.ndarray
      Input matrix
    shape : tuple, optional
      Target shape of output matrix
    repeat : tuple or int, optional
      Repititions along new dimensions
    exist_dims : tuple or int, optional
      Indices of dimensions in target shape that are present in input matrix
    expand_dims : tuple or int, optional
      Indices of dimensions in target shape that are not present in input matrix

    Returns
    -------
    numpy.ndarray
      Matrix with target shape

    Examples
    --------
    >>> expand_and_repeat([[1,2,3],[4,5,6]], shape=(2,3,4), exist_dims=(0,1))
    >>> expand_and_repeat([[1,2,3],[4,5,6]], shape=(2,3,4), expand_dims=(2,))
    >>> expand_and_repeat([[1,2,3],[4,5,6]], shape=(2,3,4), expand_dims=2)
    >>> expand_and_repeat([[1,2,3],[4,5,6]], repeat=(4,), expand_dims=(2,))
    >>> expand_and_repeat([[1,2,3],[4,5,6]], repeat=4, expand_dims=2)

    '''
    
    mtx = np.asarray(mtx)
    
    if shape is not None:
        shape = iterable(shape)
        
        if mtx.ndim > len(shape):
            raise ValueError('Nothing to expand. Number of matrix '
                             'dimensions (%d) is larger than the '
                             'dimensionality of the target shape '
                             '(%d).' % (mtx.ndim, len(shape)))
        
        if exist_dims is not None:
            exist_dims = iterable(exist_dims)

            if len(exist_dims) != len(set(exist_dims)):
                raise ValueError('Existing dimensions should be unique.')
            
            if mtx.ndim != len(exist_dims):
                raise ValueError('Number of matrix dimensions (%d) '
                                 'should match the number of existing '
                                 'dimensions (%d).' % (mtx.ndim, len(exist_dims)))

            expand_dims = [i
                           for i in range(len(shape))
                           if i not in exist_dims]
                             
        elif expand_dims is not None:
            expand_dims = iterable(expand_dims)
            
            if len(expand_dims) != len(set(expand_dims)):
                raise ValueError('Expanding dimensions should be unique.')
            
            if len(shape) - mtx.ndim != len(expand_dims):
                raise ValueError('Dimensionality of the target shape '
                                 'minus the number of matrix dimensions '
                                 '(%d) should match the number of expanding '
                                 'dimensions (%d).' % (len(shape) - mtx.ndim, len(expand_dims)))
            
            exist_dims = [i
                          for i in range(len(shape))
                          if i not in expand_dims]
            
        else:
            raise ValueError('Target shape undetermined. Provide '
                             '``exist_dims`` or ``expand_dims``.')

        repeat = [n
                  for i, n in enumerate(shape)
                  if i in expand_dims]

        for i1, i2 in enumerate(exist_dims):
            if shape[i2] != mtx.shape[i1]:
                raise ValueError('Current matrix dimension (%d = %d) '
                                 'should match target shape (%d = %d).' % (i1, mtx.shape[i1], i2, shape[i2]))

    elif repeat is not None and expand_dims is not None:
        repeat = iterable(repeat)
        expand_dims = iterable(expand_dims)

        if len(expand_dims) != len(set(expand_dims)):
            raise ValueError('Expanding dimensions should be unique.')

        if len(repeat) != len(expand_dims):
            raise ValueError('Number of repititions (%d) should '
                             'match the number of expanding '
                             'dimensions (%d).' % (len(repeat), len(expand_dims)))
            
    else:
        raise ValueError('Target shape undetermined. Provide '
                         '``shape`` and ``exist_dims`` or '
                         '``expand_dims``, or ``repeat`` and ``expand_dims``.')

    for i, n in zip(expand_dims, repeat):
        mtx = np.expand_dims(mtx, i).repeat(n, axis=i)

    return mtx


def trapz_and_repeat(mtx, x, axis=-1):

    if axis < 0:
        axis += len(mtx.shape)

    return expand_and_repeat(np.trapz(mtx, x, axis=axis),
                             shape=mtx.shape, expand_dims=axis)
