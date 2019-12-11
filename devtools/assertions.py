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

def check_array_lengths(*arrays, msg, exception):
    
    if len(set(len(a) for a in arrays)) > 1:
        raise exception(msg)
    