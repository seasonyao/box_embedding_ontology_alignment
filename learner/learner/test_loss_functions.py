import pytest
from .loss_functions import *

def test_func_list_to_dict_with_two_tuples():
    def add(x,y):
        return x+y
    def mul(x,y):
        return x*y
    out = func_list_to_dict((2,add), (100,mul))
    assert out["2*add"](3,4) == 14
    assert out["100*mul"](3,4) == 1200
