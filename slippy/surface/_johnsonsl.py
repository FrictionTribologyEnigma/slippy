import scipy.stats
from math import log, exp

__all__ = ['johnsonsl']


# noinspection PyMethodOverriding,PyPep8Naming
class johnsonsl_gen(scipy.stats.rv_continuous):
    """A Johnson SL continuous random variable.
    %(before_notes)s
    See Also
    --------
    johnsonsu
    johnsonsb
    Notes
    -----
    The probability density function for `johnsonsl` is::
        johnsonsl.pdf(x, a, b) = b * phi(a + b * log(x))
    for ``0 < x < 1`` and ``a, b > 0``, and ``phi`` is the normal pdf.
    %(example)s
    """
    def _argcheck(self, a, b):  # a is gamma b is delta
        return a == a

    def _pdf(self, x, a, b):
        trm = scipy.stats.norm.pdf(a+b*log(x))
        return x*trm

    def _cdf(self, x, a, b):
        return scipy.stats.norm.cdf(a+b*log(x))

    def _ppf(self, q, a, b):
        return exp((scipy.stats.norm.ppf(q)-a)/b)


johnsonsl = johnsonsl_gen(a=0.0, b=1.0, name='johnsonsl')
