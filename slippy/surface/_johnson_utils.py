"""
utilities used by slippy.surface
johnson fit by moments
johnson fit by quantiles
johnson sl distribution
"""

import math
import typing
import warnings

import numpy as np
import scipy.stats

__all__ = ['_johnsonsl', '_fit_johnson_by_moments', '_fit_johnson_by_quantiles']


def sl_distribution_fit(mean, sd, root_beta_1, omega, return_rv):
    dist_type = 1  # log normal
    if root_beta_1 < 0:
        xlam = -1
    else:
        xlam = 1
    u = xlam * mean
    delta = 1 / math.sqrt(math.log(omega))
    gamma = 0.5 * delta * math.log(omega * (omega - 1) / (sd ** 2))
    xi = u - math.exp((0.5 / delta - gamma) / delta)
    if return_rv:
        return _johnsonsl(gamma, delta, scale=xlam, loc=xi)
    else:
        return dist_type, xi, xlam, gamma, delta


def _johnsonsl(a, b, loc=0, scale=1):
    sq = np.log(b / scale) * b ** 2

    lin = np.log(b / scale) * (2 * a * b - 2 * b ** 2 * np.log(scale))

    scale_log = np.exp((lin / sq) / (-2))

    a_log = 1 / b

    return scipy.stats.lognorm(a_log, loc=loc, scale=scale_log)


def _fit_johnson_by_moments(mean: float, sd: float, root_beta_1: float, beta_2: float, return_rv=True) \
        -> typing.Union[scipy.stats.rv_continuous, typing.Tuple[int, float, float, float, float]]:
    """
    Fits a johnson family distribution to the specified moments

    Parameters
    ----------
    mean : float
        The mean of the distribution
    sd : float
        The population standard deviation
    root_beta_1 : float
        The skew of the distribution
    beta_2 : float
        The kurtosis of the distribution (not normalised)
    return_rv: bool, optional (True)
        If False the descriptive parameters found in the reference will be returned, else a scipy.stats.rv_continuous
        object fitted to the desired parameter will be returned

    Returns
    -------
    rv: scipy.stats.rv_continuous
        fitted distribution

    Notes
    -----
    If return_rv is set to False the following will be returned:

    DistType : {1 - log normal johnson Sl, 2 - unbounded johnson Su, 3 bounded
                johnson Sb, 4 - normal, 5 - Boundary johnson St}
        integer, a number corresponding to the type of distribution that has
        been fitted
    xi, xlam, gamma, delta : scalar, shape parameters of the fitted
        distribution, xi is epsilon, xlam is lambda

    When a normal distribution is fitted (type 4) the delta is set to 1/sd and
    gamma is set to mean/sigma, xi and xlam are arbitrarily set to 0.

    When a boundary johnson curve is fitted (type 5, St) the return parameters
    have different meanings. xi and xlam are set to the two values at which
    ordinates occur and delta to the proportion of values at xlam, gamma is
    set arbitrarily to 0.

    See Also
    -------
    scipy.stats.johnsonsu
    scipy.stats.johnsonsb
    scipy.stats.johnsonsl

    Notes
    -----
    Copied from algorithm 99.3 in:
    applied statistics, 1976 vol 25 no 2
    accessible here:
    https://www.jstor.org/stable/pdf/2346692.pdf
    also in c as part of the R supdists package here:
    https://cran.r-project.org/web/packages/SuppDists/index.html

    changes from the original functionality are noted by #CHANGE

    Examples
    --------

    >>>my_scipy_rv=_fit_johnson_by_moments(10,5,1,2)
    returns a scipy rv with a mean of 10, a standard deviation of 5, a skew of
    1 and a kurtosis of 2
    """

    tolerance = 0.01

    beta_1 = root_beta_1 ** 2

    xlam = 0
    gamma = 0
    delta = 0

    if sd < 0:
        raise ValueError("Standard deviation must be grater than or equal to 0")  # error code 1
    elif sd == 0:
        xi = mean
        dist_type = 5  # ST distribution
        if return_rv:
            raise NotImplementedError("ST distributions are not implemented")
        else:
            return dist_type, xi, xlam, gamma, delta

    if beta_2 >= 0:
        if beta_2 < beta_1 + 1 - tolerance:
            raise ValueError("beta 2 must be greater than or equal to beta_1+1")  # error code 2
        if beta_2 <= (beta_1 + tolerance + 1):
            dist_type = 5  # ST distribution
            y = 0.5 + 0.5 * math.sqrt(1 - 4 / (beta_1 + 4))
            if root_beta_1 > 0:
                y = 1 - y
            x = sd / math.sqrt(y * (1 - y))
            xi = mean - y * x
            xlam = xi + x
            delta = y
            if return_rv:
                raise NotImplementedError("ST distributions are not implemented")
            return dist_type, xi, xlam, gamma, delta

    if abs(root_beta_1) < tolerance and abs(beta_2 - 3) < tolerance:
        dist_type = 4  # Normal
        xi = mean
        xlam = sd
        delta = 1.0
        gamma = 0.0  # CHANGE from hill, in line with R package
        if return_rv:
            return scipy.stats.norm(mean, sd)
        else:
            return dist_type, xi, xlam, gamma, delta

    # 80
    # find critical beta_2 (lies on log normal line)
    x = 0.5 * beta_1 + 1
    y = root_beta_1 * math.sqrt(0.25 * beta_1 + 1)
    omega = (x + y) ** (1 / 3) + (x - y) ** (1 / 3) - 1
    beta_2_lognormal = omega * omega * (3 + omega * (2 + omega)) - 3

    if beta_2 < 0:
        beta_2 = beta_2_lognormal

    # if x is 0 log normal, positive - bounded, negative - unbounded
    x = beta_2_lognormal - beta_2

    if abs(x) < tolerance:
        return sl_distribution_fit(mean, sd, root_beta_1, omega, return_rv)

    def get_moments(g, d, max_it=500, tol_outer=1e-5, tol_inner=1e-8):
        """
        Evaluates the first 6 moments of a johnson distribution using goodwin's
        method (SB only)

        Parameters
        ----------
            g : scalar,
                shape parameter
            d : scalar,
                shape parameter
            max_it : int
                maximum number of iterations used for fitting
            tol_outer : float
                The tolerance of the outer loop
            tol_inner : float
                The tolerance for the inner loop

        Returns
        -------
        moments : list of the first 6 moments

        Notes
        -----
        Copied from algorithm 99.3 in
        applied statistics, 1976 vol 25 no 2
        accessible here:
            https://www.jstor.org/stable/pdf/2346692.pdf

        See also
        --------
        sb_fit : fits bounded johnson distributions
        fit_johnson_by_moments : fits johnson family distributions by moments
        """

        moments = 6 * [1]
        b = 6 * [0]
        c = 6 * [0]

        w = g / d

        # trial value of h

        if w > 80:
            raise ValueError("Some value too high, failed to converge")
        e = math.exp(w) + 1
        r = math.sqrt(2) / d
        h = 0.75
        if d < 3:
            h = 0.25 * d
        k = 0
        h *= 2
        while any([abs(A - C) / A > tol_outer for A, C in zip(moments, c)]):
            k += 1
            if k > max_it:
                raise StopIteration("Moment finding failed to converge O")
            if k > 1:
                c = list(moments)

            # no convergence yet try smaller h

            h *= 0.5
            t = w
            u = t
            y = h ** 2
            x = 2 * y
            moments[0] = 1 / e

            for i in range(1, 6):
                moments[i] = moments[i - 1] / e

            v = y
            f = r * h
            m = 0

            # inner loop to evaluate infinite series
            while any([abs(A - B) / A > tol_inner for A, B in zip(moments, b)]):
                m += 1
                if m > max_it:
                    raise StopIteration("Moment finding failed to converge I")
                b = list(moments)
                u = u - f
                z = math.exp(u) + 1
                t = t + f
                el = t > 23.7
                if not el:
                    s = math.exp(t) + 1
                p = math.exp(-v)
                q = p
                for i in range(1, 7):
                    moment = moments[i - 1]
                    moment_a = moment

                    p = p / z
                    if p == 0:
                        break
                    moment_a = moment_a + p
                    if not el:
                        q = q / s
                        moment_a = moment_a + q
                        el = q == 0
                    moments[i - 1] = moment_a
                # 100
                y = y + x
                v = v + y

                if any([moment == 0 for moment in moments]):
                    raise ValueError("for some reason having zero moments"
                                     " is not allowed, you naughty boy")
            # end of inner loop
            v = 1 / math.sqrt(math.pi) * h
            moments = [moment * v for moment in moments]
            # don't need to check all non zero here, just checked!
        # end of outer loop
        return moments

    def sb_fit(mean, sd, root_beta_1, beta_2, tolerance, max_it=1000):
        dist_type = 3

        beta_1 = root_beta_1 ** 2
        neg = root_beta_1 < 0

        # get d as first estimate of delta

        e = beta_1 + 1
        u = 1 / 3
        x = 0.5 * beta_1 + 1
        y = abs(root_beta_1) * math.sqrt(0.25 * beta_1 + 1)
        w = (x + y) ** u + (x - y) ** u - 1
        f = w ** 2 * (3 + w * (2 + w)) - 3
        e = (beta_2 - e) / (f - e)
        if abs(root_beta_1) < tolerance:
            f = 2
        else:
            d = 1 / math.sqrt(math.log(w))
            if d >= 0.04:
                f = 2 - 8.5245 / (d * (d * (d - 2.163) + 11.346))
            else:
                f = 1.25 * d
        f = e * f + 1  # 20

        if f < 1.8:
            d = 0.8 * (f - 1)
        else:
            d = (0.626 * f - 0.408) * (3 - f) ** (-0.479)

        # get g as a first estimate of gamma

        g = 0  # 30
        if beta_1 > tolerance ** 2:
            if d <= 1:
                g = (0.7466 * d ** 1.7973 + 0.5955) * beta_1 ** 0.485
            else:
                if d <= 2.5:
                    u = 0.0623
                    y = 0.5291
                else:
                    u = 0.0124
                    y = 0.5291
                g = beta_1 ** (u * d + y) * (0.9281 + d * (1.0614 * d - 0.7077))

        # main iterations start here
        m = 0

        u = float('inf')
        y = float('inf')

        dd = [0] * 4
        deriv = [0] * 4

        while abs(u) > tolerance ** 2 or abs(y) > tolerance ** 2:
            m += 1
            if m > max_it:
                raise ValueError('solution failed to converge error greater than'
                                 ' the specified tolerance may be present')

            # get first six moments
            moments = get_moments(g, d)
            s = moments[0] ** 2
            h2 = moments[1] - s
            if h2 <= 0:
                raise ValueError("Solution failed to converge")
            t = math.sqrt(h2)
            h2a = t * h2
            h2b = h2 ** 2
            h3 = moments[2] - moments[0] * (3 * moments[1] - 2 * s)
            rbet = h3 / h2a
            h4 = moments[3] - moments[0] * (4 * moments[2]
                                            - moments[0] * (6 * moments[1] - 3 * s))
            bet2 = h4 / h2b
            w = g * d
            u = d ** 2

            # get derivatives

            for j in range(1, 3):
                for k in range(1, 5):
                    t = k
                    if not j == 1:
                        s = ((w - t) * (moments[k - 1] - moments[k]) + (t + 1) *
                             (moments[k] - moments[k + 1])) / u
                    else:
                        s = moments[k] - moments[k - 1]
                    dd[k - 1] = t * s / d
                t = 2 * moments[0] * dd[0]
                s = moments[0] * dd[1]
                y = dd[1] - t
                deriv[j - 1] = (dd[2] - 3 * (s + moments[1] * dd[0] - t * moments[0]
                                             ) * -1.5 * h3 * y / h2) / h2a
                deriv[j + 1] = (dd[3] - 4 * (dd[2] * moments[0] + dd[0] * moments[2]) + 6 *
                                (moments[1] * t + moments[0] * (s - t * moments[0]))
                                - 2 * h4 * y / h2) / h2b

            t = 1 / (deriv[0] * deriv[3] - deriv[1] * deriv[2])
            u = (deriv[3] * (rbet - abs(root_beta_1)) - deriv[1] * (bet2 - beta_2)) * t
            y = (deriv[0] * (bet2 - beta_2) - deriv[2] * (rbet - abs(root_beta_1))) * t

            # new estimates for G and D

            g = g - u
            if beta_1 == 0 or g < 0:
                g = 0
            d = d - y

        # end of iteration

        delta = d
        xlam = sd / math.sqrt(h2)
        if neg:
            gamma = -g
            moments[0] = 1 - moments[0]
        else:
            gamma = g
        xi = mean - xlam * moments[0]

        return dist_type, xi, xlam, gamma, delta

    def su_fit(mean, sd, root_beat_1, beat_2, tollerance):
        dist_type = 2

        beta_1 = root_beta_1 ** 2
        b3 = beta_2 - 3

        # first estimate of e**(delta**(-2))

        w = math.sqrt(math.sqrt(2.0 * beta_2 - 2.8 * beta_1 - 2.0) - 1.0)
        if abs(root_beta_1) < tollerance:
            # symmetrical case results known
            y = 0
        else:
            z = float('inf')
            # johnson iterations
            while abs(beta_1 - z) > tollerance:
                w1 = w + 1
                wm1 = w - 1
                z = w1 * b3
                v = w * (6 + w * (3 + w))
                a = 8 * (wm1 * (3 + w * (7 + v)) - z)
                b = 16 * (wm1 * (6 + v) - b3)
                m = (math.sqrt(a ** 2 - 2 * b * (wm1 * (3 + w * (9 + w * (10 + v))) - 2 * w1 * z)) - a) / b
                z = m * wm1 * (4 * (w + 2) * m + 3 * w1 ** 2) ** 2 / (2 * (2 * m + w1) ** 3)
                v = w ** 2
                w = math.sqrt(math.sqrt(1 - 2 * (1.5 - beta_2 + (beta_1 * (
                        beta_2 - 1.5 - v * (1 + 0.5 * v))) / z)) - 1)
            y = math.log(math.sqrt(m / w) + math.sqrt(m / w + 1))
            if root_beta_1 > 0:
                y *= -1
        x = math.sqrt(1 / math.log(w))
        delta = x
        gamma = y * x
        y = math.exp(y)
        z = y ** 2
        x = sd / math.sqrt(0.5 * (w - 1) * (0.5 * w * (z + 1 / z) + 1))
        xlam = x
        xi = mean + xlam * np.exp(delta ** -2 / 2) * np.sinh(gamma / delta)

        return dist_type, xi, xlam, gamma, delta

    if x > 0:
        try:
            dist_type, xi, xlam, gamma, delta = sb_fit(mean, sd, root_beta_1,
                                                       beta_2, tolerance)
        except TypeError:
            warnings.warn("SB fit iterations failed returning SL distribution")
            print(omega)
            return sl_distribution_fit(mean, sd, root_beta_1, omega, return_rv)

        if return_rv:
            return scipy.stats.johnsonsb(gamma, delta, scale=xlam, loc=xi)
        else:
            return dist_type, xi, xlam, gamma, delta
    else:
        dist_type, xi, xlam, gamma, delta = su_fit(mean, sd, root_beta_1,
                                                   beta_2, tolerance)
        if return_rv:
            return scipy.stats.johnsonsu(gamma, delta, scale=xlam, loc=xi)
        else:
            return dist_type, xi, xlam, gamma, delta


def _fit_johnson_by_quantiles(quantiles):
    """
    Fits a johnson family distribution based on the supplied quartiles

    quantiles relate to normal quantiles of -1.5, -0.5, 0.5, 1.5
    => 0.067, 0.309, 0.691, 0.933 (roughly)

    Parameters
    ----------

    quantiles : array like 4 elements
        The quartiles to be fitted to

    Returns
    -------

    dist : scipy.stats._distn_infrastructure.rv_frozen
        A scipy rv object of the fitted distribution

    See Also
    --------

    References
    ----------

    Examples
    --------

    """

    m = quantiles[3] - quantiles[2]
    n = quantiles[1] - quantiles[0]
    p = quantiles[2] - quantiles[1]
    q0 = (quantiles[1] + quantiles[2]) * 0.5

    mp = m / p
    nop = n / p

    tol = 1e-4

    if mp * nop < 1 - tol:
        # bounded
        pm = p / m
        pn = p / n
        delta = 0.5 / np.arccosh(0.5 * np.sqrt((1 + pm) * (1 + pn)))
        gamma = delta * np.arcsinh((pn - pm) * np.sqrt((1 + pm) * (1 + pn) - 4) / (2 * (pm * pn - 1)))
        xlam = p * np.sqrt(((1 + pm) * (1 + pn) - 2) ** 2 - 4) / (pm * pn - 1)
        xi = q0 - 0.5 * xlam + p * (pn - pm) / (2 * (pm * pn - 1))
        dist = scipy.stats.johnsonsb(gamma, delta, loc=xi, scale=xlam)

    elif mp * nop > 1 + tol:
        # unbounded
        delta = 1 / np.arccosh(0.5 * (mp + nop))
        gamma = delta * np.arcsinh((nop - mp) / (2 * np.sqrt(mp * nop - 1)))
        xlam = 2 * p * np.sqrt(mp * nop - 1) / ((mp + nop - 2) * np.sqrt(mp + nop + 2))
        xi = q0 + p * (nop - mp) / (2 * (mp + nop - 2))
        dist = scipy.stats.johnsonsu(gamma, delta, loc=xi, scale=xlam)

    elif abs(mp - 1) > tol:
        # lognormal
        delta = 1 / np.log(mp)
        gamma = delta * np.log(abs(mp - 1) / (p * np.sqrt(mp)))
        xlam = np.sign(mp - 1)
        xi = q0 - 0.5 * p * (mp + 1) / (mp - 1)
        dist = _johnsonsl(gamma, delta, loc=xi, scale=xlam)

    else:
        # normal
        scale = 1 / m
        loc = q0 * scale
        dist = scipy.stats.norm(loc=loc, scale=scale)

    return dist
