# coding: utf8
# Copyright 2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Assertions for input checking
:Authors: **Simon Albright**
"""


def equal_array_lengths(*args, msg, exception):
    
    if len(set(len(a) for a in args)) > 1:
        raise exception(msg)


def equal_arrays(*args, msg, exception):
    
    equal_array_lengths(*args, msg = msg, exception = exception)
    for arg in list(zip(*args)):
        if not all(arg[0] == a for a in arg):
            raise exception(msg)


def not_none(*args, msg, exception):
    
    if all(v is None for v in args):
        raise exception(msg)


def _count_none(*args):
    
    nNone = 0
    for a in args:
        if a is None:
            nNone += 1
    
    return nNone

def single_none(*args, msg, exception):

    nNone = _count_none(*args)

    if nNone != 1:
        raise exception(msg)


def single_not_none(*args, msg, exception):
    
    nNone = _count_none(*args)

    if nNone != len(args) - 1:
        raise exception(msg)
   
     
def all_not_none(*args, msg, exception):
    
    if any(v is None for v in args):
        raise exception(msg)