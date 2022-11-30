#!/usr/bin/env python3

import sys

sys.path.append('..')
import ppafm.atomicUtils as au

def test_ZsToElems():

    Zs = [3, 2, 1, 7, 8]
    elems = au.ZsToElems(Zs)
    assert elems == ['Li', 'He', 'H', 'N', 'O']

if __name__ == '__main__':
    test_ZsToElems()
