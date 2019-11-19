import scipy.stats
import numpy as np
from math import log, exp
import scipy.special as sc

__all__ = ['JohnsonSLGen', 'johnsonsl']


_norm_pdf_C = np.sqrt(2 * np.pi)
_norm_pdf_logC = np.log(_norm_pdf_C)


def _norm_pdf(x):
    return np.exp(-x ** 2 / 2.0) / _norm_pdf_C


def _norm_cdf(x):
    return sc.ndtr(x)


def _norm_ppf(q):
    return sc.ndtri(q)


# noinspection PyMethodOverriding
class JohnsonSLGen(scipy.stats.rv_continuous):
    """A Johnson SL continuous random variable.
    %(before_notes)s
    See Also
    --------
    johnsonsu
    Notes
    -----
    The probability density function for `johnsonsb` is::
        johnsonsb.pdf(x, a, b) = b / (x*(1-x)) * phi(a + b * log(x/(1-x)))
    for ``0 < x < 1`` and ``a, b > 0``, and ``phi`` is the normal pdf.
    %(example)s
    """

    def _argcheck(self, a, b):  # a is gamma b is delta
        return  # array of 1s where aregs are ok and 0 where not ok

    def _pdf(self, x, a, b):
        trm = _norm_pdf(a + b * log(x))
        return b * trm

    def _cdf(self, x, a, b):
        return _norm_cdf(a + b * log(x))

    def _ppf(self, q, a, b):
        return exp((_norm_ppf(q) - a) / b)


johnsonsl = JohnsonSLGen(a=0.0, b=1.0, name='johnsonsl')
