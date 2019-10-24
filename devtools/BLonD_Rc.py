# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:55:51 2019

@author: schwarz
"""

from collections.abc import MutableMapping
import pprint
import re

from . BLonD_Rc_setup import _defaultBLonDRcParams
#import BLonD_Rc_setup

class BLonDRcParams(MutableMapping, dict):

    """
    A dictionary object including validation

    Based on matplotlib RcParams
    """

    validate = {key: converter
                for key, (default, converter) in _defaultBLonDRcParams.items()}

    # validate values on the way in
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __setitem__(self, key, val):
        try:
            try:
                cval = self.validate[key](val)
            except ValueError as ve:
                raise ValueError("Key %s: %s" % (key, str(ve)))
            dict.__setitem__(self, key, cval)
        except KeyError:
            raise KeyError(
                "%s is not a valid rc parameter " % (key) +
                "(see rcParams.keys() for a list of valid parameters)")

    def __getitem__(self, key):

        return dict.__getitem__(self, key)

    def __repr__(self):
        class_name = self.__class__.__name__
        indent = len(class_name) + 1
        repr_split = pprint.pformat(dict(self), indent=1,
                                    width=80 - indent).split('\n')
        repr_indented = ('\n' + ' ' * indent).join(repr_split)
        return '{}({})'.format(class_name, repr_indented)

    def __str__(self):
        return '\n'.join(map('{0[0]}: {0[1]}'.format, sorted(self.items())))

    def __iter__(self):
        """Yield sorted list of keys."""
        yield from sorted(dict.__iter__(self))

    def __len__(self):
        return dict.__len__(self)

    def find_all(self, pattern):
        """
        Return the subset of this RcParams dictionary whose keys match,
        using :func:`re.search`, the given ``pattern``.

        .. note::

            Changes to the returned dictionary are *not* propagated to
            the parent RcParams dictionary.

        """
        pattern_re = re.compile(pattern)
        return BLonDRcParams((key, value)
                             for key, value in self.items()
                             if pattern_re.search(key))

    def copy(self):
        return {k: dict.__getitem__(self, k) for k in self}
    
rcBLonDparams = BLonDRcParams([(key, default) for key, (default, _) in _defaultBLonDRcParams.items()])

def rc(group, **kwargs):
    """
    Set the current rc params.  *group* is the grouping for the rc, e.g.,
    for ``distribution.scale_means`` the group is ``distribution``. Group may
    also be a list or tuple of group names, e.g., (*xtick*, *ytick*).
    *kwargs* is a dictionary attribute name/value pairs, e.g.,::

      rc('lines', linewidth=2, color='r')

    sets the current rc params and is equivalent to::

      rcParams['lines.linewidth'] = 2
      rcParams['lines.color'] = 'r'


    This enables you to easily switch between several configurations.  Use
    ``matplotlib.style.use('default')`` or :func:`~matplotlib.rcdefaults` to
    restore the default rc params after changes.
    """

    if isinstance(group, str):
        group = (group,)
    for g in group:
        for k, v in kwargs.items():
            name = k
            key = '%s.%s' % (g, name)
            try:
                rcBLonDparams[key] = v
            except KeyError:
                raise KeyError(('Unrecognized key "%s" for group "%s" and '
                                'name "%s"') % (key, g, name))
