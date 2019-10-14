# coding: utf8
# Copyright 2014-2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**The rcsetup module contains the default values and the validation code for
customization using BlonD's rc settings. It is based on matplotlib.rcsetup.**

:Authors: **Markus Schwarz**
'''

class ValidateInStrings(object):
    def __init__(self, key, valid, ignorecase=False):
        'valid is a list of legal strings'
        self.key = key
        self.ignorecase = ignorecase

        def func(s):
            if ignorecase:
                return s.lower()
            else:
                return s
        self.valid = {func(k): k for k in valid}

    def __call__(self, s):
        if self.ignorecase:
            s = s.lower()
        if s in self.valid:
            return self.valid[s]
        raise ValueError('Unrecognized %s string %r: valid strings are %s'
                         % (self.key, s, list(self.valid.values())))

def validate_scale_means(bl):
    if isinstance(bl, str):
        return _validate_named_scale_means(bl)


def validate_bool(b):
    """Convert b to a boolean or raise"""
    if isinstance(b, str):
        b = b.lower()
    if b in ('t', 'y', 'yes', 'on', 'true', '1', 1, True):
        return True
    elif b in ('f', 'n', 'no', 'off', 'false', '0', 0, False):
        return False
    else:
        raise ValueError('Could not convert "%s" to boolean' % b)

def validate_float(s):
    """Convert s to float or raise."""
    try:
        return float(s)
    except ValueError:
        raise ValueError('Could not convert "%s" to float' % s)

_validate_named_scale_means = ValidateInStrings('scale_means',
         ['RMS','FWHM','fourSigma_RMS','fourSigma_FWHM','full_bunch_length'])

_defaultBLonDRcParams = {
        'distribution.scale_means' : ['RMS', validate_scale_means],
        'distribution.store_data' : [False, validate_bool]  # store data in object or just return
        }