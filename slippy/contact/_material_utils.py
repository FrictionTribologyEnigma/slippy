import numpy as np
from collections import namedtuple
import inspect
from functools import wraps

__all__ = ['ElasticProps', '_get_properties', 'Loads', 'Displacements', 'convert_array', 'convert_dict',
           'memoize_components']

ElasticProps = namedtuple('ElasticProperties', 'K E v Lam M G', defaults=(None,)*6)
Loads = namedtuple('Loads', 'x y z', defaults=(None,)*3)
Displacements = namedtuple('Displacements', 'x y z', defaults=(None,)*3)


def memoize_components(static_method=True):
    """ A decorator factory for memoizing the components of an influence matrix or other method with components

    Parameters
    ----------
    static_method: bool, optional (True)
        True if the object to be decorated is an instance or class method

    Notes
    -----
    This returns a decorator that can be used to memoize a callable which finds components. The callable MUST:

    Have it's first argument be the component
    components must be hashable

    The cache is a dict with the same keys as previously passed components, when any of the other input arguments change
    the cache is deleted

    The wrapped callable will have the additional attributes:

    cache : dict
        All of the cached values, use cache.clear() to remove manually
    spec : list
        The other arguments passed to the callable (if any of these change the cache is cleared)
    """
    if not isinstance(static_method, bool):
        raise ValueError('memoize_components is a decorator factory, it cannot be applied as a decorator directly.'
                         ' static_method argument must be a bool')

    def outer(fn):
        # non local variables spec is a list to ensure it's mutable
        spec = []
        cache = []
        sig = inspect.signature(fn)

        if static_method:
            @wraps(fn)
            def inner(component, *args, **kwargs):
                nonlocal cache, spec, sig
                new_spec = sig.bind(None, *args, **kwargs)
                new_spec.apply_defaults()
                try:
                    index = spec.index(new_spec)
                except ValueError:
                    spec.append(new_spec)
                    cache.append(dict())
                    index = len(cache)-1
                if component not in cache[index]:
                    cache[index][component] = fn(component, *args, **kwargs)
                return cache[index][component]
        else:
            @wraps(fn)
            def inner(self, component, *args, **kwargs):
                nonlocal cache, spec, sig
                new_spec = sig.bind(None, None, *args, **kwargs)
                new_spec.apply_defaults()
                try:
                    index = spec.index(new_spec)
                except ValueError:
                    spec.append(new_spec)
                    cache.append(dict())
                    index = len(cache)-1
                if component not in cache[index]:
                    cache[index][component] = fn(component, *args, **kwargs)
                return cache[index][component]

        inner.cache = cache
        inner.spec = spec

        return inner

    return outer


def convert_dict(loads_or_displacements: dict, name: str):
    """ Converts a dict of loads or displacements to a named tuple of the relevant type

    Parameters
    ----------
    loads_or_displacements: dict
        Dict with keys 'x' 'y' and 'z' or any combination, each value must be
        N by M array
    name : str {'l', 'd'}
        or any string which starts with l or d if 'l' a loads named tuple is returned if a 'd' a displacement named
        tuple is returned

    Returns
    -------
    Loads or Displacements named tuple
        The same loads or displacements in a namedtuple with fields x, y, z

    See Also
    --------
    convert_array

    Notes
    -----

    Examples
    --------
    """
    name = name.lower()
    if name.startswith('l'):
        return Loads(**loads_or_displacements)
    elif name.startswith('d'):
        return Displacements(**loads_or_displacements)
    else:
        raise ValueError("Name not recognised should start with 'l' or 'd'")


def convert_array(loads_or_displacements: np.array, name: str):
    """ Converts an array of loads or displacements to a named tuple of loads

    Parameters
    ----------
    loads_or_displacements: numpy.array
            Loads or displacements (3 by N by M array)
    name : str {'l', 'd'}
        or any string which starts with l or d if 'l' a loads named tuple is returned if a 'd' a displacement named
        tuple is returned

    Returns
    -------
    Loads or Displacements named tuple
        The same loads or displacements in a namedtuple with fields x, y, z

    See Also
    --------
    convert_dict

    Notes
    -----

    Examples
    --------
    """
    name = name.lower()
    if name.startswith('l'):
        return Loads(*loads_or_displacements)
    elif name.startswith('d'):
        return Displacements(*loads_or_displacements)
    else:
        raise ValueError("Name not recognised should start with 'l' or 'd'")


def _get_properties(set_props: dict):
    """Get all elastic properties from any pair

    Parameters
    ----------
    set_props : dict
        dict of properties must have exactly 2 members valid keys are: 'K',
        'E', 'v', 'Lam', 'M', 'G'

    Returns
    -------
    out : dict
        dict of all material properties keys are: 'K', 'E', 'v', 'Lam', 'M', 'G'

    Notes
    -----

    Keys refer to:
        - E - Young's modulus
        - v - Poission's ratio
        - K - Bulk Modulus
        - Lam - Lame's first parameter
        - G - Shear modulus
        - M - P wave modulus

    """
    if len(set_props) != 2:
        raise ValueError("Exactly 2 properties must be set,"
                         " {} found".format(len(set_props)))

    valid_keys = ['K', 'E', 'v', 'G', 'Lam', 'M']

    set_params = [key for key in list(set_props.keys()) if key in valid_keys]

    if len(set_params) != 2:
        msg = ("Invalid keys in set_props keys found are: " +
               "{}".format(set_props.keys()) +
               ". Valid keys are: " + " ".join(valid_keys))
        raise ValueError(msg)

    out = set_props.copy()

    set_params = list(set_props.keys())
    set_params.sort()
    # p is properties this saves a lot of space
    p = ElasticProps(**set_props)

    if set_params[0] == 'E':
        if set_params[1] == 'G':
            out['K'] = p.E * p.G / (3 * (3 * p.G - p.E))
            out['Lam'] = p.G * (p.E - 2 * p.G) / (3 * p.G - p.E)
            out['M'] = p.G * (4 * p.G - p.E) / (3 * p.G - p.E)
            out['v'] = p.E / (2 * p.G) - 1
        elif set_params[1] == 'K':
            out['G'] = 3 * p.K * p.E / (9 * p.K - p.E)
            out['Lam'] = 3 * p.K * (3 * p.K - p.E) / (9 * p.K - p.E)
            out['M'] = 3 * p.K * (3 * p.K + p.E) / (9 * p.K - p.E)
            out['v'] = (3 * p.K - p.E) / (6 * p.K)
        elif set_params[1] == 'Lam':
            R = np.sqrt(p.E ** 2 + 9 * p.Lam ** 2 + 2 * p.E * p.Lam)
            out['G'] = (p.E - 3 * p.Lam + R) / 4
            out['K'] = (p.E + 3 * p.Lam + R) / 6
            out['M'] = (p.E - p.Lam + R) / 2
            out['v'] = 2 * p.Lam / (p.E + p.Lam + R)
        elif set_params[1] == 'M':
            S = np.sqrt(p.E ** 2 + 9 * p.M ** 2 - 10 * p.E * p.M)
            out['G'] = (3 * p.M + p.E - S) / 8
            out['K'] = (3 * p.M - p.E + S) / 6
            out['Lam'] = (p.M - p.E + S) / 4
            out['v'] = (p.E - p.M + S) / (4 * p.M)
        else:  # set_params[1]=='v'
            out['G'] = p.E / (2 * (1 + p.v))
            out['K'] = p.E / (3 * (1 - 2 * p.v))
            out['Lam'] = p.E * p.v / ((1 + p.v) * (1 - 2 * p.v))
            out['M'] = p.E * (1 - p.v) / ((1 + p.v) * (1 - 2 * p.v))
    elif set_params[0] == 'G':
        if set_params[1] == 'K':
            out['E'] = 9 * p.K * p.G / (3 * p.K + p.G)
            out['Lam'] = p.K - 2 * p.G / 3
            out['M'] = p.K + 4 * p.G / 3
            out['v'] = (3 * p.K - 2 * p.G) / (2 * (3 * p.K + p.G))
        elif set_params[1] == 'Lam':
            out['E'] = p.G * (3 * p.Lam + 2 * p.G) / (p.Lam + p.G)
            out['K'] = p.Lam + 2 * p.G / 3
            out['M'] = p.Lam + 2 * p.G
            out['v'] = p.Lam / (2 * (p.Lam + p.G))
        elif set_params[1] == 'M':
            out['E'] = p.G * (3 * p.M - 4 * p.G) / (p.M - p.G)
            out['K'] = p.M - 4 * p.G / 3
            out['Lam'] = p.M - 2 * p.G
            out['v'] = (p.M - 2 * p.G) / (2 * p.M - 2 * p.G)
        else:  # set_params[1]=='v'
            out['E'] = 2 * p.G * (1 + p.v)
            out['K'] = 2 * p.G * (1 + p.v) / (3 * (1 - 2 * p.v))
            out['Lam'] = 2 * p.G * p.v / (1 - 2 * p.v)
            out['M'] = 2 * p.G * (1 - p.v) / (1 - 2 * p.v)
    elif set_params[0] == 'K':
        if set_params[1] == 'Lam':
            out['E'] = 9 * p.K * (p.K - p.Lam) / (3 * p.K - p.Lam)
            out['G'] = 3 * (p.K - p.Lam) / 2
            out['M'] = 3 * p.K - 2 * p.Lam
            out['v'] = p.Lam / (3 * p.K - p.Lam)
        elif set_params[1] == 'M':
            out['E'] = 9 * p.K * (p.M - p.K) / (3 * p.K + p.M)
            out['G'] = 3 * (p.M - p.K) / 4
            out['Lam'] = (3 * p.K - p.M) / 2
            out['v'] = (3 * p.K - p.M) / (3 * p.K + p.M)
        else:  # set_params[1]=='v'
            out['E'] = 3 * p.K * (1 - 2 * p.v)
            out['G'] = (3 * p.K * (1 - 2 * p.v)) / (2 * (1 + p.v))
            out['Lam'] = 3 * p.K * p.v / (1 + p.v)
            out['M'] = 3 * p.K * (1 - p.v) / (1 + p.v)
    elif set_params[0] == 'Lam':
        if set_params[1] == 'M':
            out['E'] = (p.M - p.Lam) * (p.M + 2 * p.Lam) / (p.M + p.Lam)
            out['G'] = (p.M - p.Lam) / 2
            out['K'] = (p.M + 2 * p.Lam) / 3
            out['v'] = p.Lam / (p.M + p.Lam)
        else:
            out['E'] = p.Lam * (1 + p.v) * (1 - 2 * p.v) / p.v
            out['G'] = p.Lam(1 - 2 * p.v) / (2 * p.v)
            out['K'] = p.Lam * (1 + p.v) / (3 * p.v)
            out['M'] = p.Lam * (1 - p.v) / p.v
    else:
        out['E'] = p.M * (1 + p.v) * (1 - 2 * p.v) / (1 - p.v)
        out['G'] = p.M * (1 - 2 * p.v) / (2 * (1 - p.v))
        out['K'] = p.M * (1 + p.v) / (3 * (1 - p.v))
        out['Lam'] = p.M * p.v / (1 - p.v)

    return out
