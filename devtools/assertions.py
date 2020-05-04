# coding: utf8
# Copyright 2020 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Assertions for input checking
:Authors: **Simon Albright**, **Alexandre Lasheen**
"""


def equal_array_lengths(*args, msg, exception):
    r"""Checks in the passed arguments that all of them are arrays of
    equal lengths. If np.ndarray are passed, only the first dimension is
    checked presently.

    Parameters
    ----------
        args: list, np.ndarray
            The arrays to be checked
        msg: str
            A custom error message
        exception: Exception object
            The exception to be raised

    Examples
    --------
    >>> msg = 'Arrays have different lengths!'
    >>> exception = Exception
    >>>
    >>> list_1 = [1,2,3]
    >>> list_2 = [4,5]
    >>> list_3 = [6,7,8]
    >>>
    >>> my_args = (list_1, list_2, list_3)
    >>> equal_array_lengths(*my_args, msg=msg, exception=exception)
    >>> # OR 'equal_array_lengths(list_1, list_2, list_3, msg=msg, exception=exception)'
    Exception: Arrays have different lengths!

    >>> my_args = (list_1, list_3)
    >>> equal_array_lengths(*my_args, msg=msg, exception=exception)
    >>> # -> does not raise any exception

    """

    if len(set(len(a) for a in args)) > 1:
        raise exception(msg)


def equal_arrays(*args, msg, exception):
    r"""Checks in the passed arguments that all of them are arrays with equal
    values. If np.ndarray are passed, only the first dimension is checked
    presently.

    Parameters
    ----------
        args: list, np.ndarray
            The arrays to be checked
        msg: str
            A custom error message
        exception: Exception object
            The exception to be raised

    Examples
    --------
    >>> msg = 'Arrays have different lengths!'
    >>> exception = Exception
    >>>
    >>> list_1 = [1,2,3]
    >>> list_2 = [4,5,6]
    >>> list_3 = [1,2,3]
    >>>
    >>> my_args = (list_1, list_2, list_3)
    >>> equal_arrays(*my_args, msg=msg, exception=exception)
    >>> # OR 'equal_arrays(list_1, list_2, list_3, msg=msg, exception=exception)'
    Exception: Arrays have different lengths!

    >>> my_args = (list_1, list_3)
    >>> equal_arrays(*my_args, msg=msg, exception=exception)
    >>> # -> does not raise any exception

    """

    equal_array_lengths(*args, msg=msg, exception=exception)
    for arg in list(zip(*args)):
        if not all(arg[0] == a for a in arg):
            raise exception(msg)


def not_none(*args, msg, exception):
    r""" Checks in the passed arguments that all of them are not None, raises an
     exception otherwise.

    Parameters
    ----------
        args: any type
            The arguments to be checked
        msg: str
            A custom error message
        exception: Exception object
            The exception to be raised

    Examples
    --------
    >>> msg = 'All the arguments are None!'
    >>> exception = Exception
    >>>
    >>> my_args = (None, None, None)
    >>> not_none(*my_args, msg=msg, exception=exception)
    >>> # OR 'not_none(None, None, None, msg=msg, exception=exception)'
    Exception: All the arguments are None!

    >>> my_args = (1, None, None)
    >>> not_none(*my_args, msg=msg, exception=exception)
    >>> # -> does not raise any exception

    """

    if all(v is None for v in args):
        raise exception(msg)


def _count_none(*args):
    r""" Counts the number of None passed in arguments.

    Parameters
    ----------
        args: any type
            The arguments to be checked

    Returns
    -------
        nNone: int
            The number of None counted in the arguments

    Examples
    --------
    >>> my_args = (1, 2, None)
    >>> nNone = _count_none(*my_args)
    >>> print(nNone)
    1

    """

    nNone = 0
    for a in args:
        if a is None:
            nNone += 1

    return nNone


def single_none(*args, msg, exception):
    r""" Checks in the passed arguments that there is a single None, raises an
    exception otherwise.

    Parameters
    ----------
        args: any type
            The arguments to be checked
        msg: str
            A custom error message
        exception: Exception object
            The exception to be raised

    Examples
    --------
    >>> msg = 'More that one argument is None!'
    >>> exception = Exception
    >>>
    >>> my_args = (1, None, None)
    >>> single_none(*my_args, msg=msg, exception=exception)
    >>> # OR 'single_not_none(1, None, None, msg=msg, exception=exception)'
    Exception: More that one argument is None!

    >>> my_args = (1, 2, None)
    >>> single_none(*my_args, msg=msg, exception=exception)
    >>> # -> does not raise any exception



    """

    nNone = _count_none(*args)

    if nNone != 1:
        raise exception(msg)


def single_not_none(*args, msg, exception):
    r""" Checks in the passed arguments that exactly one argument is not
    None, raises an exception otherwise.

    Parameters
    ----------
        args: any type
            The arguments to be checked
        msg: str
            A custom error message
        exception: Exception object
            The exception to be raised

    Examples
    --------
    >>> msg = 'More that one argument is not None!'
    >>> exception = Exception
    >>>
    >>> my_args = (1, 2, None)
    >>> single_not_none(*my_args, msg=msg, exception=exception)
    >>> # OR 'single_not_none(1, 2, None, msg=msg, exception=exception)'
    Exception: More that one argument is not None!

    >>> my_args = (1, None, None)
    >>> single_not_none(*my_args, msg=msg, exception=exception)
    >>> # -> does not raise any exception

    """

    nNone = _count_none(*args)

    if nNone != (len(args) - 1):
        raise exception(msg)


def all_not_none(*args, msg, exception):
    r""" Checks in the passed arguments that all of them are not None, raises an
    exception otherwise.

    Parameters
    ----------
        args: any type
            The arguments to be checked
        msg: str
            A custom error message
        exception: Exception object
            The exception to be raised

    Examples
    --------
    >>> msg = 'There is a least one None!'
    >>> exception = Exception
    >>>
    >>> my_args = (1, 2, None)
    >>> all_not_none(*my_args, msg=msg, exception=exception)
    >>> # OR 'all_not_none(1, 2, None, msg=msg, exception=exception)'
    Exception: There is a least one None!

    >>> my_args = (1, 2, 1)
    >>> all_not_none(*my_args, msg=msg, exception=exception)
    >>> # -> does not raise any exception

    """

    if any(v is None for v in args):
        raise exception(msg)
