from nose.tools import *
from oceanwaves import *


@raises(NotImplementedError)
def test_datawell():
    '''Test reading of datawell files'''
    ow = OceanWaves.from_datawell('../data/datawell')
