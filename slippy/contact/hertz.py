# hertz.py

# TODO stresses in elliptical contacts

import typing
from collections import abc
from collections import namedtuple

import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
import scipy.special as special
import sympy as sp
from sympy.solvers import solve

__all__ = ['hertz_full', 'solve_hertz_line', 'solve_hertz_point', 'HertzLineSolution', 'HertzPointSolution']

HertzLineSolution = namedtuple('HertzLineSolution', ['r_rel', 'e1', 'e2', 'v1', 'v2', 'load',
                                                     'e_star', 'contact_width', 'max_pressure',
                                                     'max_shear_stress', 'max_von_mises'])


def solve_hertz_line(*, r_rel: float = None,
                     e1: float = None, e2: float = None,
                     v1: float = None, v2: float = None,
                     load: float = None,
                     max_pressure: float = None,
                     max_shear_stress: float = None,
                     max_von_mises: float = None,
                     contact_width: float = None,
                     _system: dict = None):
    """
    Finds remaining hertz parameter for a line contact

    Parameters
    ----------
    r_rel: float, optional
        The relative radii of the contact defined as 1/(1/r1+1/r2) where r1,2
        are the radii of the bodies, assuming that the axes are parallel (line
        contact). For a cylinder on the flat r_rel is the radius of the
        cylinder.
    e1,e2 : float, optional
        The Young's moduli of the bodies, if neither is set they will be
        assumed to be equal
    v1,v2 : float, optional
        The Poisson's ratios for the bodies, if neither is set the are assumed
        to be equal
    load : float, optional
        The load per unit length for the contact
    max_pressure : float, optional
        The maximum pressure in the contact region
    max_shear_stress : float, optional
        The maximum shear stress in the first body, swap the materials to
        change the body
    contact_width : float, optional
        The contact half width
    max_von_mises : float, optional
        The maximum von mises stress in the first body, swap the materials to
        change body

    Returns
    -------
    dict
        The system with all of the possible inputs defined

    See Also
    --------
    solve_hertz_point
    hertz_full

    Notes
    -----
    This function will only work for line contacts such as aligned cylinders in contact or a cylinder on a plane
    It also uses approximate formulas based on a Poisson's ratio of 0.3 for stress results, if more accurate results are
    required hertz_full should be used.

    The independent parameters are: r_rel, e1, e2, v1, v2 and load.
    The dependent parameters are: max_pressure, max_shear_stress, contact_width and max_von_mises

    For this function to run either:
    All of the independent parameters are set,
    or
    All but one of the independent parameters are set and exactly one of the dependent parameters
    or
    All of the independent parameters are set apart from both e's or both v's, in this case they are assumed to be equal

    Regardless of which combination has been set the resulting output will contain all of the possible input parameters

    References
    ----------
    Johnson, K. (1985). Contact Mechanics. Cambridge: Cambridge University Press. doi:10.1017/CBO9781139171731

    Examples
    --------
    # Finding the load required to give a specific contact pressure
    result = solve_hertz_line(r_rel=0.01, e1=200e9, e2=200e9, v1=0.3, v2=0.3, max_pressure=1e9, load=None)
    result['load']

    # Finding the stiffness of material required to give a specific contact width
    result = solve_hertz_line(r_rel=0.05, e1=None, e2=None, v1=0.3, v2=0.3, load=1000, contact_width=1e-8)
    """
    _system = {'r_rel': r_rel, 'e1': e1, 'e2': e2, 'v1': v1, 'v2': v2, 'load': load}

    der_params = [max_pressure, max_shear_stress, max_von_mises, contact_width]
    der_none = [el is None for el in der_params]
    if all(der_none):
        return _fill_hertz_solution_line(_system)
    # else work out the first set param then just recursively call to fill in the dict
    mats_none = [el is None for el in [e1, e2, v1, v2]]
    if any(mats_none):
        is_none = 'e_star'

    else:
        _system['e_star'] = 1 / ((1 - v1 ** 2) / e1 + (1 - v2 ** 2) / e2)
        if r_rel is None:
            is_none = 'r_rel'
        else:
            is_none = 'load'
    _system[is_none] = sp.Symbol(is_none)

    if not der_none[0]:
        # max pressure given
        _system[is_none] = max(solve(_system['load'] * _system['e_star'] / np.pi / _system['r_rel'] -
                                     max_pressure ** 2, _system[is_none]))
    elif not der_none[1]:
        # max_shear_stress given
        _system[is_none] = max(solve(0.3 ** 2 * _system['load'] * _system['e_star'] / np.pi /
                                     _system['r_rel'] - max_shear_stress ** 2, _system[is_none]))
    elif not der_none[2]:
        raise NotImplementedError("Not implemented yet, try another stress")
        # max von mises stress given
    elif not der_none[3]:
        # contact width given
        _system[is_none] = max(solve(4 * _system['load'] * _system['r_rel'] / np.pi / _system['e_star'] -
                                     contact_width ** 2, _system[is_none]))
    else:
        raise ValueError("Not enough parameters given!")
    _system[is_none] = float(_system[is_none])
    # sort out materials
    if is_none == 'e_star':
        if sum(mats_none) == 2:
            if mats_none[0] and mats_none[1]:
                # neither E is set
                _system['e1'] = _system['e_star'] * (2 - v1 ** 2 - v2 ** 2)
                _system['e2'] = _system['e1']
            elif mats_none[2] and mats_none[3]:
                # neither v is set
                _system['v1'] = np.sqrt((1 / _system['e_star'] * (1 / e1 + 1 / e2)) - 1)
                _system['v2'] = _system['v1']
            else:
                raise ValueError('Both moduli or both poisson\'s ratios can '
                                 'be found but not a combination')
        elif sum(mats_none) == 1:
            # only one thing not set
            props = ['e1', 'e2', 'v1', 'v2']
            prop_none = props[next((i for i, j in enumerate(mats_none)
                                    if j), None)]

            _system[prop_none] = sp.Symbol(prop_none)
            _system[prop_none] = float(max(solve((1 - _system['v1'] ** 2) / _system['e1'] +
                                                 (1 - _system['v2'] ** 2) / _system['e2'] -
                                                 1 / _system['e_star'], _system[prop_none])))
        else:
            raise ValueError("Not enough material properties set")
    # recursive call to fill in the rest of the dict
    return _fill_hertz_solution_line(_system=_system)


def _fill_hertz_solution_line(_system: dict):
    """
    Fills in the derived parameters of the system given all of the set parameters

    Parameters
    ----------
    _system : dict
        The hertz system with all of the set parameters found, and derived parameters will be overwritten

    Returns
    -------
    system :dict
        The system with all of the derived parameters filled in (a copy of _system)
    """
    system = _system.copy()
    try:
        system['e_star'] = 1 / ((1 - system['v1'] ** 2) / system['e1'] + (1 - system['v2'] ** 2) / system['e2'])
        c = 1 / (1 + 4 * (system['v1'] - 1) * system['v1']) ** 0.5 if system['v1'] <= 0.1938 else \
            1.164 + 2.975 * system['v1'] - 2.906 * system['v1'] ** 2
        system['contact_width'] = np.sqrt(4 * system['load'] * system['r_rel'] / system['e_star'] / np.pi)
        system['max_pressure'] = np.sqrt(system['load'] * system['e_star'] / np.pi / system['r_rel'])
        system['max_shear_stress'] = system['max_pressure'] * 0.3
        system['max_von_mises'] = system['max_pressure'] / c
    except ValueError:
        raise ValueError('Not enough input parameters defined')
    try:
        del system['is_none']
    except KeyError:
        pass
    return HertzLineSolution(**system)


HertzPointSolution = namedtuple('HertzPointSolution', ['r_rel', 'e1', 'e2', 'v1', 'v2', 'load',
                                                       'e_star', 'contact_radius', 'max_pressure', 'max_tensile_stress',
                                                       'max_shear_stress', 'max_von_mises', 'total_displacement'])


def solve_hertz_point(*, r_rel=None,
                      e1=None, e2=None,
                      v1=None, v2=None,
                      load=None,
                      max_pressure=None,
                      max_shear_stress=None,
                      contact_radius=None,
                      max_von_mises=None,
                      total_displacement=None,
                      max_tensile_stress=None):
    """Finds the remaining hertz parameter for a spherical contact

    Parameters
    ----------

    r_rel: float, optional
        The relative radii of the contact defined as 1/(1/r1+1/r2) where r1,2 are the radii of the bodies.
        For a ball on the flat r_rel is the radius of the ball.
    e1,e2 : float, optional
        The Young's moduli of the bodies, if neither is set they will be assumed to be equal
    v1,v2 : float, optional
        The Poisson's ratios for the bodies, if neither is set the are assumed to be equal
    load : float, optional
        The load per unit length for the contact

    max_pressure : float, optional
        The maximum pressure in the contact region
    max_shear_stress : float, optional
        The maximum shear stress in the first body, swap the materials to
        change the body
    contact_radius : float, optional
        The radius of the contact patch
    max_von_mises : float, optional
        The maximum von mises stress in the first body, swap the materials to change body
    total_displacement : float, optional
        The displacement of the bodies towards each other
    max_tensile_stress : float, optional
        The maximum tensile stress in the first body, swap the materials to cahnge the body

    Returns
    -------
    HertzPointSolution : namedtuple
        The system with all of the possible inputs defined

    See Also
    --------
    solve_hertz_line
    hertz_full

    Notes
    -----
    This function will only work for spherical contacts such as a ball on flat, a ball on ball or crossed cylinders.
    It also uses approximate formulas based on a poissions ratio of 0.3 for stress calculations, if more accurate
    results are required hertz_full should be used.

    The independent parameters are: r_rel, e1, e2, v1, v2 and load.
    The dependent parameters are: max_pressure, max_shear_stress, contact_width and max_von_mises

    For this function to run either:
    All of the independent parameters are set,
    or
    All but one of the independent parameters are set and exactly one of the dependent parameters
    or
    All of the independent parameters are set apart from both e's or both v's, in this case they are assumed to be equal

    Regardless of which combination has been set the resulting output will contain all of the possible input parameters

    References
    ----------
    Johnson, K. (1985). Contact Mechanics. Cambridge: Cambridge University Press. doi:10.1017/CBO9781139171731

    Examples
    --------

    # Finding the radius of ball required to give a specific contact pressure
    result = solve_hertz_point(r_rel=None, e1=200e9, e2=200e9, v1=0.3, v2=0.3, max_pressure=1e9, load=500)
    result['load']

    # Finding the stiffness of material required to give a specific contact radius
    result = solve_hertz_point(r_rel=0.05, e1=None, e2=None, v1=0.3, v2=0.3, load=1000, contact_radius=1e-8)

    """

    _system = {'r_rel': r_rel, 'e1': e1, 'e2': e2, 'v1': v1, 'v2': v2, 'load': load}

    der_params = [max_pressure, max_shear_stress, max_von_mises, contact_radius, total_displacement, max_tensile_stress]
    der_none = [el is None for el in der_params]

    if all(der_none):
        return _fill_hertz_solution_point(_system=_system)

    # else work out the first set param then just recursively call to fill in the dict
    mats_none = [el is None for el in [e1, e2, v1, v2]]
    if any(mats_none):
        is_none = 'e_star'
    else:
        _system['e_star'] = 1 / ((1 - v1 ** 2) / e1 + (1 - v2 ** 2) / e2)
        if r_rel is None:
            is_none = 'r_rel'
        else:
            is_none = 'load'
    _system[is_none] = sp.Symbol(is_none)

    if not der_none[0]:  # done
        # max pressure given
        _system[is_none] = max(solve(6 * _system['load'] * _system['e_star'] ** 2 / np.pi ** 3 /
                                     _system['r_rel'] ** 2 - max_pressure ** 3, _system[is_none]))
    elif not der_none[1]:  # done
        # max_shear_stress given
        # assuming that max shear stress is 0.46*p0 (as in v=0.3)
        _system[is_none] = max(solve(0.46 * ((6 * _system['load'] * _system['e_star'] ** 2 / np.pi ** 3 /
                                              _system['r_rel'] ** 2) ** (1 / 3)) - max_shear_stress, _system[is_none]))

    elif not der_none[2]:  # TODO
        # max von mises stress given
        raise NotImplementedError("Von mises stresses are not yet implemented")

    elif not der_none[3]:  # done
        # contact radius given
        _system[is_none] = max(solve(3 * _system['load'] * _system['r_rel'] / 4 / _system['e_star'] -
                                     contact_radius ** 3, _system[is_none]))
    elif not der_none[4]:  # done
        # total deflection given
        _system[is_none] = max(solve(9 * _system['load'] ** 2 / 16 / _system['r_rel'] /
                                     _system['e_star'] ** 2 - total_displacement ** 3, _system[is_none]))
    elif not der_none[5]:  # done
        # max tensile stress given
        if is_none != 'e_star':
            _system[is_none] = max(solve((1 - 2 * _system['v1']) * ((6 * _system['load'] * _system['e_star'] ** 2 /
                                                                     np.pi ** 3 / _system['r_rel'] ** 2) ** (
                                                                            1 / 3)) / 3 -
                                         max_tensile_stress, _system[is_none]))
        elif sum(mats_none) == 2:
            is_none = None  # To stop the materials being solved for next
            del _system['e_star']
            if v1 is None and v2 is None:
                v = sp.Symbol('v')
                v = max(solve((1 - 2 * v) * ((6 * _system['load'] * (1 / ((1 - v ** 2) / _system['e1'] +
                                                                          (1 - v ** 2) / _system[
                                                                              'e2'])) ** 2 / np.pi ** 3 /
                                              _system['r_rel'] ** 2) ** (1 / 3)) / 3 - max_tensile_stress, v))
                v = float(v)
                _system['v1'] = v
                _system['v2'] = v
            elif e1 is None and e2 is None:
                e = sp.Symbol('e')
                e = max(solve((1 - 2 * _system['v1']) * ((6 * _system['load'] * (1 / ((1 - _system['v1'] ** 2) / e +
                                                                                      (1 - _system[
                                                                                          'v2'] ** 2) / e)) ** 2 /
                                                          np.pi ** 3 / _system['r_rel'] ** 2) ** (1 / 3)) / 3 -
                              max_tensile_stress, _system[is_none]))
                e = float(e)
                _system['e1'] = e
                _system['e2'] = e

        elif sum(mats_none) == 1:
            del _system['e_star']
            is_none = [key for key, value in _system.items() if value is None][0]

            x0 = 0.3 if is_none.startswith('v') else 200e9

            root_results = optimize.root(_root_tensile(_system, is_none, max_tensile_stress), np.array(x0))

            _system[is_none] = root_results.x if root_results.success else None

            if _system[is_none] is None:
                raise StopIteration("Result failed to converge")

        _system['e_star'] = 1 / ((1 - _system['v1'] ** 2) / _system['e1'] + (1 - _system['v2'] ** 2) / _system['e2'])

    else:
        raise ValueError("Not enough parameters given!")

    if is_none is not None:
        _system[is_none] = float(_system[is_none])

    # sort out materials
    if is_none == 'e_star':
        if sum(mats_none) == 2:
            if mats_none[0] and mats_none[1]:
                # neither E is set
                _system['e1'] = _system['e_star'] * (2 - v1 ** 2 - v2 ** 2)
                _system['e2'] = _system['e1']
            elif mats_none[2] and mats_none[3]:
                # neither v is set
                _system['v1'] = np.sqrt((1 / _system['e_star'] * (1 / e1 + 1 / e2)) - 1)
                _system['v2'] = _system['v1']
            else:
                raise ValueError('Both moduli or both poisson\'s ratios can '
                                 'be found but not a combination')
        elif sum(mats_none) == 1:
            # only one thing not set
            props = ['e1', 'e2', 'v1', 'v2']
            prop_none = props[next((i for i, j in enumerate(mats_none)
                                    if j), None)]

            _system[prop_none] = sp.Symbol(prop_none)
            _system[prop_none] = max(solve((1 - _system['v1'] ** 2) / _system['e1'] +
                                           (1 - _system['v2'] ** 2) / _system['e2'] -
                                           1 / _system['e_star'], _system[prop_none]))
        else:
            raise ValueError("Not enough material properties set")
    # recursive call to fill in the rest of the dict
    return _fill_hertz_solution_point(_system=_system)


def _root_tensile(system, is_none, max_tensile_stress):
    """
    Helper function, the root of the inner function is the value for is_none in the system

    Paramerters
    -----------
    system : dict
        The system with one parameter set to none
    is_none : str
        The key of the none parameter in the dict
    max_tensile_stress : float
        The maximum tensile stress to be solved for
    """
    system = system.copy()

    def inner(value):
        nonlocal system
        system[is_none] = abs(value)
        return (1 - 2 * system['v1']) * ((6 * system['load'] * (1 / ((1 - system['v1'] ** 2) / system['e1'] +
                                                                     (1 - system['v2'] ** 2) / system['e2'])) ** 2 /
                                          np.pi ** 3 / system['r_rel'] ** 2) ** (1 / 3)) / 3 - max_tensile_stress

    return inner


def _fill_hertz_solution_point(_system: dict):
    """
    Fills in the derived parameters of the _system dict and returns it
    Parameters
    ----------
    _system : dict
        the system from solve hertz point

    Returns
    -------
    system : HertzPointSolution
        A named tuple with all of the derived parameters filled in
    """
    system = _system.copy()

    try:
        system['e_star'] = 1 / ((1 - system['v1'] ** 2) / system['e1'] + (1 - system['v2'] ** 2) / system['e2'])
        c = (1.30075 + 0.87825 * system['v1'] + 0.54373 * system['v1'] ** 2)
        a = (3 * system['load'] * system['r_rel'] / 4 / system['e_star']) ** (1 / 3)
        system['contact_radius'] = a
        system['max_pressure'] = (6 * system['load'] * system['e_star'] ** 2 / np.pi ** 3 /
                                  system['r_rel'] ** 2) ** (1 / 3)
        system['max_tensile_stress'] = (1 - 2 * system['v1']) * system['max_pressure'] / 3
        system['max_shear_stress'] = 0.46 * system['max_pressure']
        system['max_von_mises'] = system['max_pressure'] * c
        system['total_displacement'] = a ** 2 / system['r_rel']

    except ValueError:
        raise ValueError('Not enough input parameters defined')

    try:
        del system['is_none']
    except KeyError:
        pass
    return HertzPointSolution(**system)


def hertz_full(r1: typing.Union[typing.Sequence, float], r2: typing.Union[typing.Sequence, float],
               moduli: typing.Union[typing.Sequence, float], v: typing.Union[typing.Sequence, float],
               load: float, angle: float = 0.0, line: bool = False, integration_error: float = 1e-6,
               root_error: float = 1e-6):
    """Find the hertzian stress solution for the given system

    Finds all the known results to the system defined, including full field
    stress results if possible.

    Parameters
    ----------
    r1, r2 : list
        Two element list of the radii in the first body (r1) and the second
        body (r2) of the radii of the first body in the x and y directions.
        Each element should be a float, use float('inf') to indicate a flat
        surface. If a single number is supplied both elements are set to that
        number: r1=1 is equivalent to r1=[1,1]
    moduli : list
        Two element list of the young's moduli of the first and second bodies.
        See note on units.
    v : list
        Two element list of the poisson's ratios of the first and second bodies.
    load : float
        The load applied for a point contact or the load per unit length for a
        line contact. If a line contact is intended the line keyword should be
        set to True. See note on units.
    angle : float, optional (0)
        The angle between the x axes in radians
    line : bool, optional (False)
        Should be set to True for line contacts, otherwise an error is raised,
        this is done to avoid accidental line contacts changing the definition
        of the load parameter.
    integration_error : float, optional (1e-6)
        The maximum relative error on the integration steps, used only for
        elliptical contacts see [Deeg 1992] for more information.
    root_error : float, optional (1e-6)
        The maximum relative error on the root finding step, used only for
        elliptical contacts see [Deeg 1992] for more information.


    Returns
    -------
    results : dict
        Dictionary of the results:
            line contact : True is it is a line contact
            r_eff : The effective radius of the contact


    See Also
    --------
    solve_hertz_line
    solve_hertz_point

    Notes
    -----
    Units must be consistent: if the young's moduli is given in N/mm**2, the
    radii should be given in mm and the load should be given in N. etc.

    The range for the k parameter (ratio of the contact radii) for elliptical
    contacts is set to 10e-5, practically contacts which are more smaller
    ratios will not converge, in these cases consider treating as a line
    contact.

    References
    ----------
    Unless otherwise stated formulas are taken from:
    Johnson, K. (1985). Contact Mechanics. Cambridge: Cambridge University Press. doi:10.1017/CBO9781139171731

    The depth and magnitude of maximum stresses in point and line contacts are taken from:
    Green, I. (2005). Poisson ratio effects and critical values in spherical and cylindrical.
    International Journal of Applied Mechanics and Engineering, 10(3), 451–462.

    The determination of the ratio of the contact radii is taken from:
    Deeg, Emil W.. “New Algorithms for Calculating Hertzian Stresses , Deformations , and Contact Zone Parameters.”
    (1996).

    Examples
    --------

    """
    inf = float('inf')
    results = dict()
    # check inputs
    r1 = _sanitise_radii(r1)
    r2 = _sanitise_radii(r2)
    moduli = _sanitise_material(moduli, 'moduli')
    v = _sanitise_material(v, 'v')

    angle = angle % np.pi
    angle = angle if angle < np.pi / 2 else angle - np.pi

    if load <= 0:
        raise ValueError("Negative or zero loads are not allowed")

    # find the angles between the principal radii of the surfaces and the
    # principal relative radii
    with np.errstate(divide='ignore'):
        const_a = ((1 / r2[0]) - (1 / r2[1]))
        const_b = ((1 / r1[0]) - (1 / r1[1]))
        t = 2 * angle
        alpha = np.arctan2(const_a * np.sin(t), (const_a * np.cos(t) + const_b)) / 2
        beta = np.arctan2(const_b * np.sin(t), (const_b * np.cos(t) + const_a)) / 2

    if abs(alpha + beta - angle) > 1e-8:
        raise ValueError("There is a bug in this program please report")

    results['alpha'] = alpha
    results['beta'] = beta
    # find principal relative radii
    c2a = np.cos(alpha) ** 2
    c2b = np.cos(beta) ** 2
    s2a = np.sin(alpha) ** 2
    s2b = np.sin(beta) ** 2

    # From contact mechanics by Johnson pg 85 (eq 4.4)
    with np.errstate(divide='ignore'):
        r1_rel = 1 / (c2a / r1[0] + s2a / r1[1] + c2b / r2[0] + s2b / r2[1])
        r2_rel = 1 / (s2a / r1[0] + c2a / r1[1] + s2b / r2[0] + c2b / r2[1])
    results['relative_radii'] = [r1_rel, r2_rel]
    # reduction of the problem
    results['r_e'] = (r1_rel * r2_rel) ** 0.5
    e_star = 1 / ((1 - v[0] ** 2) / moduli[0] + (1 - v[1] ** 2) / moduli[1])
    results['e_star'] = e_star
    # check validity of radii
    if r1_rel < 0 or r2_rel < 0:
        raise ValueError("Relative radii of curvature are negative")
    if r1_rel == r2_rel == inf:
        raise ValueError("Conformal contacts are not supported by the "
                         "hertzian theory")

    if r1_rel / r2_rel > 1e5 or r2_rel / r1_rel > 1e5:
        results['contact_shape'] = 'line'
        if not line:
            raise ValueError('Line contact detected, if this is intentional '
                             'the line key word should be set to True')
        r = min(r1_rel, r2_rel)
        a = np.sqrt(4 * load * r / np.pi / e_star)
        results['contact_radii'] = [rad if rad == inf else a for rad in
                                    [r1_rel, r2_rel]]
        results['contact_area'] = float('inf')
        results['mean_pressure'] = load / a
        p0 = 2 * load / np.pi / a
        results['max_pressure'] = p0
        results['pressure_f'] = _pressure_line_contact(a, load)
        results['surface_tensile_stress_f'] = _pressure_line_contact(a, load, neg=True)

        results['stress_f'] = _stress_line_contact(a, p0)
        c = [1 / (1 + 4 * (v1 - 1) * v1) ** 0.5 if v1 <= 0.1938 else
             1.164 + 2.975 * v1 - 2.906 * v1 ** 2 for v1 in v]
        zeta = [0 if v1 <= 0.1938 else 0.223 + 2.321 * v1 - 2.397 * v1 ** 2 for v1 in v]
        results['max_von_mises_stress'] = [c1 * p0 for c1 in c]
        results['max_von_mises_depth'] = [a * zeta1 for zeta1 in zeta]
        results['total_deflection'] = 1 / np.pi / e_star * load * (np.log(4 * np.pi * e_star * r / load) - 1)

        # The following is taken from deeg

        def line_opt_fn(psi, v1):
            return -1 * max(
                psi - psi * v1 + (-1 + 2 * psi ** 2 * (v1 - 1) + 2 * v1) / (2 * np.sqrt(1 + psi ** 2)),
                psi * (psi / np.sqrt(1 + psi ** 2) - 1),
                v1 * (np.sqrt(1 + psi ** 1) - psi) - 1 / (2 * np.sqrt(1 + psi ** 2)))

        line_opts = [optimize.minimize(line_opt_fn, np.array([0.48086782]), bounds=[[0, 10]], args=v1) for v1 in v]

        results['max_shear_stress_b'] = [out.fun * -1 * p0 if out.success else None for out in line_opts]
        results['depth_of_max_shear'] = [out.x * a if out.success else None for out in line_opts]

    elif r1_rel == r2_rel:
        if line:
            raise ValueError('This is a spherical contact, the line flag '
                             'should be set to false or not set')
        results['contact_shape'] = 'sphere'
        # spherical contact
        a = (3 * load * r1_rel / 4 / e_star) ** (1 / 3)
        results['contact_radii'] = [a, a]
        results['contact_area'] = np.pi * a ** 2
        results['mean_pressure'] = load / results['contact_area']
        results['total_deflection'] = a ** 2 / r1_rel
        p0 = 3 * load / (2 * np.pi * a ** 2)
        results['max_pressure'] = p0
        results['surface_displacement_b_f'] = [_displacement_spherical_contact(moduli[0],
                                                                               v[0], a, p0),
                                               _displacement_spherical_contact(moduli[1], v[1], a, p0)]
        results['pressure_f'] = _pressure_spherical_contact(a, p0)
        results['max_tensile_stress_b'] = [(1 - 2 * v1) * p0 / 3 for v1 in v]
        results['stress_z_axis_b'] = [_stress_z_axis_spherical(a, p0, v1)
                                      for v1 in v]

        # The following is taken from Deeg
        results['max_von_mises_stress_b'] = [p0 * (1.30075 + 0.87825 * v1 +
                                                   0.54373 * v1 ** 2) for v1 in v]
        results['max_von_mises_depth_b'] = [a * (0.38167 + 0.33136 * v1) for v1 in v]

        shear_opts = [
            optimize.minimize(lambda psi: -0.5 * (-1 + 3 / 2 / (1 + psi ** 2) + psi * (1 + v1) * np.arctan(1 / psi)),
                              np.array([0.48086782]), bounds=[[0, 10]]) for v1 in v]

        results['max_shear_stress_b'] = [out.fun[0] * (-1 * p0) if out.success else None for out in shear_opts]
        results['max_shear_depth_b'] = [out.x[0] * a if out.success else None for out in shear_opts]

    else:
        if line:
            raise ValueError('This is an elliptical contact, the line flag '
                             'should be set to false or not set')
        results['contact_shape'] = 'elliptical'

        # elliptical contact parameters are named as in the reference (Deeg)

        theta = [4 * (1 - v1 ** 2) / e1 for v1, e1 in zip(v, moduli)]
        r = 1 / r1[0] + 1 / r1[1] + 1 / r2[0] + 1 / r2[1]
        r1_elliptical = 1 / r1[0] - 1 / r1[1]
        r2_elliptical = 1 / r2[0] - 1 / r2[1]
        omega = np.arccos(
            np.sqrt(r1_elliptical ** 2 + r2_elliptical ** 2 + r1_elliptical * r2_elliptical * np.cos(2 * angle)) / r)
        # k1,k2,k3=0.04,0.56,0.85
        # g0=k1 if omega <=  5/180*np.pi else 0
        # g1=g0 if omega >=  1/180*np.pi else 0
        # g2=k2 if omega <= 73/180*np.pi else 0
        # g3=g2 if omega >   5/180*np.pi else 0
        # g4=k3 if omega >  73/180*np.pi else 0

        # k_init=g0+g2+g4

        # check this works, might be very slow
        try:
            k = optimize.brentq(_k_root, 0.00001, 1,
                                args=(omega, integration_error),
                                rtol=root_error)
        except ValueError:
            raise ValueError("Root finding for elliptical contact failed, "
                             "ensure contact parameters are correct, "
                             "if problem persists please report.")

        f = (2 * _i(k) / np.pi / (np.sin(omega / 2)) ** 2) ** (1 / 3)
        g = (2 * _j(k) / np.pi / (np.cos(omega / 2)) ** 2) ** (1 / 3)

        a = f * (3 * load / 8 / r * (theta[0] + theta[1])) ** (1 / 3)
        b = g * (3 * load / 8 / r * (theta[0] + theta[1])) ** (1 / 3)

        q = a * b * np.pi

        alpha1 = 3 * load * theta[0] * _h(k) / a / 8 / np.pi
        alpha2 = alpha1 * theta[1] / theta[0]
        deflection = alpha1 + alpha2

        p0 = 3 * load / 2 / q

        if r1_rel < r2_rel:
            alpha, beta = alpha + np.pi / 2, beta + np.pi / 2
        # end of the algorithm in the reference

        results['max_pressure'] = p0
        results['contact_radii'] = [a, b]
        results['contact_area'] = q
        results['mean_pressure'] = p0 / q
        results['total_deflection'] = deflection
        results['deflection_b'] = [alpha1, alpha2]
        results['pressure_f'] = _pressure_elliptical_contact(a, b, p0, alpha, beta)
        results['surface_displacement_b_f'] = [_displacement_elliptical(
            a, b, p0, alpha, beta, v1, e1) for v1, e1 in zip(v, moduli)]
        # TODO fill in here from Johnson
        # TODO make own method for finding the maximum stresses
    return results


###############################################################################
# Functions for elliptical contact results
###############################################################################
def _pressure_elliptical_contact(a, b, p0, alpha, beta):
    def pressure(x, y, transform=0):
        """
        Pressure for an elliptical contact

        Parameters
        ----------
        x,y : array-like
            x and y coordinates of the points of interest
        transform : int {0,1,2}, optional (0)
            a flag which defines which axes the result is displayed on. If set
            to 0 the result is displayed on the 'contact axes' which are
            aligned with the principal radii of the conatct ellipse. If set to
            1 or 2 the result is aligned with the axes of the first or second
            body respectively.

        Returns
        -------
        pressure : array
            The contact pressure at each of the points of interest

        Notes
        -----
        The pressure distribution is given by:
            p(x,y)=p0*(1-(x/a)**2-(y/b)**2)**0.5

        References
        ----------

        [1] Johnson, K. (1985). Contact Mechanics. Cambridge: Cambridge
        University Press. doi:10.1017/CBO9781139171731
        """
        if transform:
            x, y = _transform_axes(x, y, [alpha, beta][transform - 1])
        squared = np.clip((1 - (x / a) ** 2 - (y / b) ** 2), 0, float('inf'))
        return p0 * squared ** 0.5

    return pressure


def _displacement_elliptical(a, b, p0, alpha, beta, v, modulus):
    """
    Gives a closure for the surface z displacement for an elliptical contact
    """
    # memoize the elliptical integrals
    l_johnson = None
    m_johnson = None
    n_johnson = None

    def surface_displacement(x, y, transform=0):
        """
        Into surface displacement at the surface for an elliptical contact

        Parameters
        ----------
        x,y : array-like
            x and y coordinates of the points of interest
        transform : int {0,1,2}, optional (0)
            a flag which defines which axes the result is displayed on. If set
            to 0 the result is displayed on the 'contact axes' which are
            aligned with the principal radii of the contact ellipse. If set to
            1 or 2 the result is aligned with the axes of the first or second
            body respectively.

        Returns
        -------
        displacement : array
            The into surface displacement at each of the points of interest

        Notes
        -----
        The pressure distribution is given by:
            p(x,y)=p0*(1-(x/a)**2-(y/b)**2)**0.5

        References
        ----------

        [1] Johnson, K. (1985). Contact Mechanics. Cambridge: Cambridge
        University Press. doi:10.1017/CBO9781139171731
        """
        nonlocal l_johnson, m_johnson, n_johnson
        if l_johnson is None:
            if b > a:
                raise ValueError("Change in a>b or b>a between sources, "
                                 "sort out")
            e = (1 - b ** 2 / a ** 2) ** 0.5

            l_johnson = np.pi * p0 * b * special.ellipk(e)
            m_johnson = np.pi * p0 * b / e ** 2 / a ** 2 * (special.ellipk(e) - special.ellipe(e))
            n_johnson = np.pi * p0 * b / a ** 2 / e ** 2 * (
                (a ** 2 / b ** 2) * special.ellipe(e) - special.ellipk(e))

        if transform:
            x, y = _transform_axes(x, y, [alpha, beta][transform - 1])

        out_of_bounds = np.clip((1 - (x / a) ** 2 - (y / b) ** 2), 0, float('inf')) == 0
        displacement = np.array((1 - v ** 2) / modulus / np.pi * (l_johnson - m_johnson * x ** 2 - n_johnson * y ** 2))

        displacement[out_of_bounds] = float('Nan')

        return displacement

    return surface_displacement


def _transform_axes(x, y, angle):
    """
    Transforms points on one set of axes to another, the two sets share an
    origin and are off set by angle

    """
    x = np.asarray(x)
    y = np.asarray(y)
    x1 = x * np.cos(angle) - y * np.sin(angle)
    y1 = x * np.sin(angle) + y * np.cos(angle)

    return x1, y1


###############################################################################
# Functions to find elliptical contact parameters
###############################################################################
def _k_root(k, aux_angle, int_error=1e-6):
    """
    Defines the equation that k (b/a) is the root of for elliptical contacts
    """
    return ((k ** 3 / np.tan(aux_angle / 2) ** 2) -
            (_j(k, int_error) / _i(k, int_error)))


def _h(k, int_error=1e-6):
    """
    The H parameter in the reference (Deeg), defined in table 6
    """
    return integrate.quad(lambda psi: 1 / np.sqrt(1 - (1 - k ** 2) * np.sin(psi) ** 2),
                          0, np.pi / 2, epsrel=int_error)[0]


def _i(k, int_error=1e-6):
    """
    The I parameter in the reference (Deeg), defined in table 6
    """

    return integrate.quad(lambda psi: np.cos(psi) ** 2 / np.sqrt((1 - (1 - k ** 2) * np.sin(psi) ** 2) ** 3),
                          0, np.pi / 2, epsrel=int_error)[0]


def _j(k, int_error=1e-6):
    """
    The J parameter in the reference (Deeg), defined in table 6
    """
    if k == 0:
        return 0
    return integrate.quad(lambda psi: np.cos(psi) ** 2 / np.sqrt((1 - (1 - 1 / k ** 2) * np.sin(psi) ** 2) ** 3),
                          0, np.pi / 2, epsrel=int_error)[0]


###############################################################################
# Line contact functions
###############################################################################
def _stress_line_contact(a, p0):
    """Gives a closure for full field stresses in a line contact

    Parameters
    ----------
    a : float
        the contact radius of the line contact
    p0 : float
        The maximum pressure of the line contact

    Returns
    -------
    Stress : closure
        Callable closure that gives the stesses in the x and z directions and
        shear stress in the xz plane

    Notes
    -----
    Stress in the y direction is v*(sima_x+sigma_z) where v is the poissions
    ratio

    References
    ----------
    [1] Johnson, K. (1985). Contact Mechanics. Cambridge: Cambridge
    University Press. doi:10.1017/CBO9781139171731
    """

    def stress(x, z):
        """ Full field stresses for a hertzian line contact

        Parameters
        ----------
        x,z : array-like
            x and z coordinates of points of interest

        Returns
        -------
        Stresses : dict
            Dictionary of stresses at the points of interest with keys
            {'sigma_x', 'sigma_z', 'tau_xz'}.

        Notes
        -----
        The stress in the Y direction (along the axis of contact) can be found
        by v*(sigma_x+sigma_z) where v is the poisson's ratio.

        References
        ----------
        [1] Johnson, K. (1985). Contact Mechanics. Cambridge: Cambridge
        University Press. doi:10.1017/CBO9781139171731
        """
        x, z = np.asarray(x), np.asarray(z)
        a2 = a ** 2

        # m and n as defined on page 103 of Johnson
        c = 0.5 * (np.sqrt((a2 - x ** 2 + z ** 2) ** 2 + 4 * x ** 2 * z ** 2) + (a2 - x ** 2 + z ** 2))
        m = np.sign(z) * c
        n = np.sign(x) * c

        sig_x = -1 * p0 / a * (m * (1 + (z ** 2 + n ** 2) / (m ** 2 + n ** 2)) - 2 * z)
        sig_z = -1 * p0 / a * m * (1 + (z ** 2 + n ** 2) / (m ** 2 + n ** 2))
        tau_xz = -1 * p0 / a * n * ((m ** 2 - z ** 2) / (m ** 2 + n ** 2))
        return {'sigma_x': sig_x, 'sigma_z': sig_z, 'tau_xz': tau_xz}

    return stress


def _pressure_line_contact(a, load, neg=False):
    """Gives a closure for the pressure in a line contact

    Parameters
    ----------

    a : float
        Contact radius
    load : float
        load per unit length for the line contact

    Returns
    -------
    surface_pressure : closure
        A callable closure which gives the pressure at any point in the contact
    """

    def surface_pressure(x):
        """The surface pressure in a hertzian line contact

        Surface pressures for a hertzian line contact, no coordinate transforms
        are completed on the input coordinates. The x axis in this function is
        perpendicular to the axes of the cylinders in contact.

        Parameters
        ----------
        x : array-like
            The x-coordinates of the positions of interest

        Returns
        -------
        pressure : array
            The pressure at the points of interest

        Notes
        -----
        For a line contact the radial stress at the surface is always equal to
        the pressure at the surface and compressive[1]

        References
        ----------
        [1] Johnson, K. (1985). Contact Mechanics. Cambridge: Cambridge
        University Press. doi:10.1017/CBO9781139171731

        """
        x = np.clip(np.asarray(x), -1 * a, a)
        pressure = 2 * load / np.pi / a ** 2 * (a ** 2 - x ** 2) ** 0.5
        if neg:
            return -1 * pressure
        else:
            return pressure

    return surface_pressure


# Spherical contact functions

def _stress_z_axis_spherical(a, p0, v):
    """ Gives a closure for the stresses on the z axis in a spherical contact

    Parameters
    ----------
    a : float
        The contact radius
    p0 : float
        The maximum pressure according to hertz
    v : float
        The possions ratio of the surface

    Returns
    -------
    stress : closure
        A callable that can be used to find the stress at any point on the
        z axis
    """

    def stress(z):
        """ Stresses at any point on the z axis for a point contact

        Parameters
        ----------
        z : array-like
            The z coordinates of the points of interest

        Returns
        -------
        stress : dict
            The stresses at the points of interest with keys: {'sigma_r',
            'sigma_theta', 'sigma_z'}
        """
        z = np.asarray(z)
        sig_r = -1 * p0 * (1 + v) * (1 - (z / a) * np.arctan(a / z)) + 1 / (0.5 * (1 + z ** 2 / a ** 2))
        sig_theta = sig_r.copy()
        sig_z = -1 * p0 * (1 + z ** 2 / a ** 2) ** -1
        return {'sigma_r': sig_r, 'sigma_theta': sig_theta, 'sigma_z': sig_z}

    return stress


def _pressure_spherical_contact(a, p0):
    """ Gives a closure for the surface pressures

    Parameters
    ----------
    a : float
        The contact radius
    p0 : float
        The maximum pressure according to hertz

    Returns
    -------
    pressure : closure
        A callable that can be used to find the pressure at any point on the
        surface
    """

    def pressure(x, y=0):
        """Contact pressure for a spherical contact according to hertz

        Parameters
        ----------
        x : array-like
            The x coordinated of the points of interest
        y : array-like, optional (0)
            The y coordinated of the points of interest

        Returns
        -------
        pressure : array
            The surface pressure at the specified points
        """
        r = np.asarray(np.sqrt(x ** 2 + y ** 2))
        squared = np.clip(1 - (r / a) ** 2, 0, float('inf'))
        return p0 * squared ** 0.5

    return pressure


def _displacement_spherical_contact(young_modulus, v, a, p0):
    def displacement(x, y=0):
        """
        Surface displacements for spherical contacts

        Parameters
        ----------

        x,y : float or array-like
            x and y coordinates of the surface points of interest

        Returns
        -------
        result : dict
            With members 'uz' and 'ur'; in to surface and radial displacements
            respectively

        See Also
        --------
        hertz

        References
        ----------
        Contact mechanics. Johnson
        """

        r = np.asarray(np.sqrt(x ** 2 + y ** 2))
        axial = np.zeros_like(r)
        radial = np.zeros_like(r)
        r_in = r[r <= a]
        r_out = r[r > a]

        axial[r <= a] = ((1 - v ** 2) * np.pi * p0) / (young_modulus * 4 * a) * (2 * a ** 2 - r_in ** 2)
        axial[r > a] = ((1 - v ** 2) * p0) / \
                       (young_modulus * 2 * a) * ((2 * a ** 2 - r_out ** 2) * np.arcsin(a / r_out) +
                                                  r_out * a * np.sqrt(1 - a ** 2 / r_out ** 2))
        with np.errstate(divide='ignore', invalid='ignore'):
            radial[r <= a] = ((1 - 2 * v) * (1 + v) * a ** 2 * p0 / (3 * young_modulus * r_in) *
                              (1 - (1 - r_in ** 2 / a ** 2) ** (3 / 2)))

        radial[np.isnan(radial)] = 0

        radial[r > a] = (1 - 2 * v) * (1 + v) * p0 * a ** 2 / (3 * young_modulus * r_out)
        return {'uz': axial, 'ur': radial}

    return displacement


def _sanitise_radii(radii):
    """
    checks on the radii input to hertz
    """
    if type(radii) is not list:
        try:
            radii = [np.float(radii)] * 2
        except TypeError:
            raise TypeError("radii must be list or number, not a"
                            " {}".format(type(radii)))
    else:
        if len(radii) == 1:
            radii = 2 * radii

        if len(radii) != 2:
            raise ValueError("Radii must be a two element list supplied radii"
                             " list has {} elements".format(len(radii)))
        try:
            radii = [np.float(r) for r in radii]
        except ValueError:
            raise TypeError("Elements of radii are not convertible to floats:"
                            "{}".format(radii))
    if any([r == 0 for r in radii]):
        raise ValueError("Radii contains zero values, use float('inf') for fla"
                         "t surfaces")
    return np.array(radii)


def _sanitise_material(e, name):
    """
    Make sure material properties are valid and in the expected form
    """
    if not isinstance(e, abc.Sequence):
        try:
            e = 2 * [np.float(e)]
        except ValueError:
            raise TypeError("Material properties should be sequence type or "
                            "convertible to float, "
                            f'{name} supplied is {type(e)}')
    if len(e) == 1:
        e = list(e) * 2
    if len(e) == 2:
        if any([e == 0 for e in e]) and name == 'e':
            raise ValueError('Zero moduli are not supported, supplied moduli'
                             f' were e={e}')
        return np.array(e, dtype=np.float)
    else:
        raise ValueError(f"Too many elements supplied in {name}")
