"""
Common sub models for lubricants

"""

import numpy as np

__all__ = ['constant_array_property', 'roelands', 'barus', 'nd_barus', 'nd_roelands', 'dowson_higginson',
           'nd_dowson_higginson']


def constant_array_property(value: float):
    """ Produce a closure that returns an index able constant value

    Parameters
    ----------
    value: float
        The value of the constant

    Returns
    -------
    inner: closure
        A closure that returns a fully populated array the same size as the just_touching_gap keyword argument, this is
        guaranteed to be in the current state dict, and therefore passed as a keyword when sub models are saved.

    Notes
    -----
    Using this closure means that lubrication steps can be writen for the general case, using indexing on fluid
    properties.

    See Also
    --------
    constant_array_property

    Examples
    --------
    >>> closure = constant_array_property(1.23)
    >>> constant_array = closure(just_touching_gap = np.ones((5,5)))
    >>> constant_array.shape
    (5,5)
    >>> constant_array[0,0]
    1,23
    """

    def inner(just_touching_gap: np.ndarray, **kwargs):
        return np.ones_like(just_touching_gap) * value

    return inner


def roelands(eta_0, pressure_0, z):
    """ The roelands pressure viscosity equation

    Parameters
    ----------
    eta_0, pressure_0, z: float
        Coefficients for the equation, see notes for details

    Returns
    -------
    inner: closure
        A callable that produces the viscosity terms according to the Roelands equation, see notes for details

    Notes
    -----
    The roelands equation linking viscosity (eta) to the fluid pressure (p) is given by:
    eta(p) = eta_0*exp((ln(eta_0)+9.67)*(-1+(1+(p/p_0)^z))
    eta_0, p_0 and z are coefficients that depend on the oil and it's temperature.

    """
    ln_eta_0 = np.log(eta_0) + 9.67

    def inner(pressure: np.ndarray, **kwargs):
        return eta_0 * np.exp(ln_eta_0 * (-1 + (1 + pressure / pressure_0) ** z))

    return inner


def nd_roelands(eta_0: float, pressure_0: float, pressure_hertzian: float, z: float):
    """ The roelands pressure viscosity equation in a non dimentional form

    Parameters
    ----------
    eta_0, pressure_0, z: float
        Coefficients for the equation, see notes for details
    pressure_hertzian: float
        The hertzian pressure used to non dimentionalise the pressure term in the equation. Should be the same as is
        used in the reynolds solver

    Returns
    -------
    inner: closure
        A callable that produces the non dimentional viscosity according to the Roelands equation, see notes for details

    Notes
    -----
    The roelands equation linking viscosity (eta) to the non dimentional fluid pressure (nd_p) is given by:
    eta(p)/eta_0 = exp((ln(eta_0)+9.67)*(-1+(1+(nd_p/p_0*p_h)^z))
    eta_0, p_0 and z are coefficients that depend on the oil and it's temperature.
    p_h is the hertzian pressure used to non dimentionalise the pressure term.

    """
    ln_eta_0 = np.log(eta_0) + 9.67
    p_all = pressure_hertzian / pressure_0

    def inner(nd_pressure: np.ndarray, **kwargs):
        return np.exp(ln_eta_0 * (-1 + (1 + p_all * nd_pressure) ** z))

    return inner


def barus(eta_0: float, alpha: float):
    """ The Barus pressure viscosity equation

    Parameters
    ----------
    eta_0, alpha: float
        Coefficients in the equation, see notes for details

    Returns
    -------
    inner: closure
        A callable that returns the resulting viscosity according to the barus equation

    Notes
    -----
    The Barus equation linking pressure (p) to viscosity (eta) is given by:
    eta(p) = eta_0*exp(alpha*p)
    In which eta_0 and alpha are coefficients which depend on the lubricant and it's temperature
    """

    def inner(pressure: np.ndarray, **kwargs):
        return eta_0 * np.exp(alpha * pressure)

    return inner


def nd_barus(pressure_hertzian: float, alpha: float):
    """ A non dimentional form of the Barus equation

    Parameters
    ----------
    alpha: float
        A coefficient in the Barus equation, see notes for details
    pressure_hertzian: float
        The hertzian pressure used to non dimensionalise the pressure

    Returns
    -------
    inner: closure
        A callable that will produce the non dimentional viscosity according to the barus equation

    Notes
    -----
    The non dimentional Barus equation relating the viscosity (eta) to the non dimentional pressure (nd_p) is given by:
    eta(p)/eta_0 = exp(alpha*p_h*nd_p)
    In which alpha is alpha is a coefficient which will depend on the lubricant used and the temperature
    p_h is the hertzian pressure used to non dimentionalise the pressure, this must be the same as is passed to the
    reynolds solver.

    """

    def inner(nd_pressure: np.ndarray, **kwargs):
        return np.exp(alpha * pressure_hertzian * nd_pressure)

    return inner


def dowson_higginson(rho_0: float):
    """ The Dowson Higginson equation relating pressure to density

    Parameters
    ----------
    rho_0: float
        A coefficient of the dowson higginson equation, seen notes for details

    Returns
    -------
    inner: closure
        A callable that returns the density based on the pressure according to the dowson higginson equation

    Notes
    -----
    The dowson higginson equation relating pressure (p) to density (rho) is given by:
    rho(p) = rho_0 * (5.9e8+1.34*p)/(5.9e8+p)
    In which rho_0 is the parameter of the equation which will depend on the lubricant used and it's temperature
    """

    def inner(pressure: np.ndarray, **kwargs):
        return rho_0 * (5.9e8 + 1.34 * pressure) / (5.9e8 + pressure)

    return inner


def nd_dowson_higginson(pressure_hertzian: float):
    """ A non dimentional form of the Dowson Higginson equation relating pressure to density

    Parameters
    ----------
    pressure_hertzian: float
        The hertzian pressure used to non dimentionalise the pressure, this must match the pressure given to the
        reynolds solver

    Returns
    -------
    inner: closure
        A callable that returns the non dimentional density based on the non dimentional pressure

    Notes
    -----
    The non dimentional dowson higginson equation relating non dimensional pressure (nd_p) to density (rho) is given by:
    rho(p)/rho_0 = (5.9e8+1.34*p_h*nd_p)/(5.9e8+p_h*nd_p)
    In which p_h is the hertzian pressure used to non denationalise the pressure and rho_0 is a parameter of the
    dimentional form of the dowson higginson equation. Here the value rho(p)/rho_0 is returned
    """
    constant = 5.9e8 / pressure_hertzian

    def inner(nd_pressure: np.ndarray, **kwargs):
        return (constant + 1.34 * nd_pressure) / (constant + nd_pressure)

    return inner
