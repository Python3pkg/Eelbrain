# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"A few basic operations needed throughout Eelbrain"
from collections import defaultdict
import functools
import logging
import re
from warnings import warn


LOG_LEVELS = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO,
              'WARNING': logging.WARNING, 'ERROR': logging.ERROR,
              'CRITICAL': logging.CRITICAL}


def deprecated(version, replacement):
    """Decorator to deprecate functions and methods

    Also handles docstring of the deprecated function.

    Parameters
    ----------
    version : str
        Version in which the feature will be removed.
    replacement : callable | str
        Either a verbal description, or a pointer to the direct replacement
        function which takes the same arguments. In the latter case, the
        replacement is automatically called and the deprecated function is not
        used anymore.
    """
    def dec(func):
        msg = ('%s is deprecated and will be removed in version %s' %
               (func.__name__, version))
        if isinstance(replacement, str):
            msg += '; ' + replacement
            call_func = func
        elif replacement is not None:
            msg += "; use %s instead" % replacement.__name__
            call_func = replacement
        else:
            raise TypeError("replacement=%r" % (replacement,))
        func.__doc__ = msg

        @functools.wraps(func)
        def new(*args, **kwargs):
            warn(msg, DeprecationWarning)
            return call_func(*args, **kwargs)

        return new
    return dec


def log_level(arg):
    """Convert string to logging module constant"""
    if isinstance(arg, int):
        return arg
    elif isinstance(arg, str):
        try:
            return LOG_LEVELS[arg.upper()]
        except KeyError:
            raise ValueError("Invalid log level: %s. mus be one of %s" %
                             (arg, ', '.join(LOG_LEVELS)))
    else:
        raise TypeError("Invalid log level: %s. need int or str." % repr(arg))


def set_log_level(level, logger_name='eelbrain'):
    """Set the minimum level of messages to be logged

    Parameters
    ----------
    level : str | int
        Level as string (debug, info, warning, error, critical) or
        corresponding constant from the logging module.
    logger_name : str
        Name of the logger for which to set the logging level. The default is
        the Eelbrain logger.
    """
    logging.getLogger(logger_name).setLevel(log_level(level))


class intervals:
    """Iterate over each successive pair in a sequence.

    Examples
    --------
    >>> for i in intervals([1, 2, 3, 45]):
    ...     print i
    ...
    (1, 2)
    (2, 3)
    (3, 45)
    """
    def __init__(self, seq):
        self.seq = seq
        self.i = 0
        if len(self.seq) < 2:
            raise StopIteration

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        if len(self.seq) <= self.i:
            raise StopIteration
        return self.seq[self.i - 1], self.seq[self.i]


class LazyProperty(object):
    "http://blog.pythonisito.com/2008/08/lazy-descriptors.html"
    def __init__(self, func):
        self._func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def __get__(self, obj, klass=None):
        if obj is None: return None
        result = obj.__dict__[self.__name__] = self._func(obj)
        return result


def _natural_keys(text):
    "Sorting key for natural sorting"
    # after http://stackoverflow.com/a/5967539/166700
    return [int(c) if c.isdigit() else c for c in re.split('(\d+)', text)]


def natsorted(seq):
    return sorted(seq, key=_natural_keys)


def n_decimals(number):
    "Number of meaningful decimals in ``number``, at least 1"
    s = str(number)
    if '.' in s:
        return s[::-1].index('.')
    else:
        return 1


class keydefaultdict(defaultdict):
    "http://stackoverflow.com/a/2912455/166700"
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret
