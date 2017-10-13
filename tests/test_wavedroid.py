from nose.tools import *
from oceanwaves import *


@raises(NotImplementedError)
def test_wavedroid():
    ow = OceanWaves.from_wavedroid('../data/wavedroid')
