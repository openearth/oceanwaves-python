from nose.tools import *
from oceanwaves.utils import *


ITERABLES = [(1, 1),
             (1., 1),
             ('a', 1),
             ('abc', 3),
             ([1,2,3], 3),
             ((1,2,3), 3),
             ({1:1,2:2,3:3}, 3)] # iterable, length

INTEGRATIONS = [(0, 9),
                (1, 18),
                (-1, 18),
                (-2, 9)] # axis, sum

MTX = np.asarray([[1,1,1],
                  [2,2,2]])

### CHECK ITERABLE CONVERSION

def test_iterables():
    '''Test if any variable is properly transformed into an interable'''
    for a, l in ITERABLES:
        yield check_iterable, a, l


def check_iterable(a, l):

    # check if iterable
    iter(iterable(a))

    # check if length is correct
    assert_equals(len(iterable(a)), l)
    

### CHECK MATRIX EXPANSION

def test_expansion_1():
    '''Test matrix expansion when no expansion is needed'''
    mtx = expand_and_repeat(MTX, shape=(2,3), exist_dims=(0,1))
    assert_equal(mtx.shape, MTX.shape)
    assert_equal(mtx.sum(), MTX.sum())

    
def test_expansion_2():
    '''Test matrix expansion at end of matrix'''
    mtx = expand_and_repeat(MTX, shape=(2,3,4), exist_dims=(0,1))
    assert_equal(mtx.shape, MTX.shape + (4,))
    assert_equal(mtx.sum(), 4 * MTX.sum())


def test_expansion_3():
    '''Test matrix expansion in the middle of matrix'''
    mtx = expand_and_repeat(MTX, shape=(2,4,3), exist_dims=(0,2))
    assert_equal(mtx.shape, (2,4,3))
    assert_equal(mtx.sum(), 4 * MTX.sum())


def test_expansion_4():
    '''Test matrix expansion at the start of matrix'''
    mtx = expand_and_repeat(MTX, shape=(4,2,3), exist_dims=(1,2))
    assert_equal(mtx.shape, (4,2,3))
    assert_equal(mtx.sum(), 4 * MTX.sum())


def test_expansion_5():
    '''Test matrix expansion of multiple dimensions at once'''
    mtx = expand_and_repeat(MTX, shape=(1,2,3,4), exist_dims=(1,2))
    assert_equal(mtx.shape, (1,2,3,4))
    assert_equal(mtx.sum(), 4 * MTX.sum())


def test_expansion_6():
    '''Test matrix expansion of multiple dimensions using `expand_dims` construct'''
    mtx = expand_and_repeat(MTX, shape=(1,2,3,4), expand_dims=(0,3))
    assert_equal(mtx.shape, (1,2,3,4))
    assert_equal(mtx.sum(), 4 * MTX.sum())

    
def test_expansion_7():
    '''Test matrix expansion of multiple dimensions using `repeat` construct'''
    mtx = expand_and_repeat(MTX, repeat=(1,4), expand_dims=(0,3))
    assert_equal(mtx.shape, (1,2,3,4))
    assert_equal(mtx.sum(), 4 * MTX.sum())
    

@raises(ValueError)
def test_expansion_exception1():
    '''Test exception if output shape is smaller than input shape'''
    mtx = expand_and_repeat(MTX, shape=(2,), expand_dims=())


@raises(ValueError)
def test_expansion_exception2():
    '''Test exception if existing dimensions are not unique'''
    mtx = expand_and_repeat(MTX, shape=(2,3,4), exist_dims=(0,0))


@raises(ValueError)
def test_expansion_exception3():
    '''Test exception if existing dimensions exceed current matrix dimensions'''
    mtx = expand_and_repeat(MTX, shape=(2,3,4), exist_dims=(0,1,2))

    
@raises(ValueError)
def test_expansion_exception4():
    '''Test exception if expanding dimensions are not unique'''
    mtx = expand_and_repeat(MTX, shape=(2,3,4), expand_dims=(2,2))


@raises(ValueError)
def test_expansion_exception5():
    '''Test exception if expanding dimensions exceed new dimensions in traget shape'''
    mtx = expand_and_repeat(MTX, shape=(2,3,4), expand_dims=(1,2))

    
@raises(ValueError)
def test_expansion_exception6():
    '''Test exception if `exist_dims` not `expand_dims` are given'''
    mtx = expand_and_repeat(MTX, shape=(2,3,4))

    
@raises(ValueError)
def test_expansion_exception7():
    '''Test exception if target shape dimensions do not match current matrix dimensions'''
    mtx = expand_and_repeat(MTX, shape=(2,3,4), exist_dims=(0,2))
    

@raises(ValueError)
def test_expansion_exception8():
    '''Test exception if expanding dimensions are not unique'''
    mtx = expand_and_repeat(MTX, repeat=4, expand_dims=(2,2))

    
@raises(ValueError)
def test_expansion_exception9():
    '''Test exception when number of expanding dimensions do not match number of repititions'''
    mtx = expand_and_repeat(MTX, repeat=4, expand_dims=(1,2))

    
@raises(ValueError)
def test_expansion_exception10():
    '''Test exception if `shape` not `repeat` is given'''
    mtx = expand_and_repeat(MTX, expand_dims=2)

    
### CHECK DIMENSION CONSERVING INTEGRATION

def test_integrations():
    '''Test integration with conservation of dimensions'''
    for i, s in INTEGRATIONS:
        yield check_integration, i, s


def check_integration(i, s):
    x = range(MTX.shape[i])
    mtx = trapz_and_repeat(MTX, x, axis=i)
    assert_equals(mtx.sum(), s)
    assert_equals(mtx.shape, MTX.shape)
