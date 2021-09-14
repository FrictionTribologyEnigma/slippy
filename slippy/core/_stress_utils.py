import numpy as np
import numba
import slippy
from collections.abc import Sequence

__all__ = ['get_derived_stresses', 'solve_cubic']

_cuda_cubic_cache = {}

try:
    import cupy as cp

    def _make_cuda_cubic_solver(dtype):
        eps = slippy.CUBIC_EPS
        s_dtype = str(dtype)
        if not s_dtype.startswith("float"):
            raise ValueError("can only make cubic solver for single and double floats")
        single = 'f' if str(s_dtype).endswith('32') else ''
        if not single and not str(s_dtype).endswith('64'):
            raise ValueError("can only make cubic solver for single and double floats")
        cubic_kernel = cp.ElementwiseKernel(
            "T b, T c, T d", "T r1, T r2, T r3",
            f'''
        if (fabs{single}(d) < {eps}) {{
        // cancel and find remaining roots by quadratic formula
        r1 = 0;
        T diff = sqrt{single}(b*b-4*c)/2;
        r2 = (-b)/2 + diff;
        r3 = (-b)/2 - diff;
        }} else {{
        // convert to depressed cubic
        T p = c-b*b/3;
        T q = 2*b*b*b/27 - b*c/3 + d;
        if (fabs{single}(p) < {eps}) {{
        r1 = cbrt{single}(-q) - b/3;
        r2 = r1;
        r3 = r1;
        }} else if (fabs{single}(q) < {eps}) {{
            r3 = - b/3;
            if (p<0) {{
                T diff = sqrt{single}(-p);
                r2 = diff - b/3;
                r1 = - diff - b/3;
            }} else {{
                r1 = r3;
                r2 = r3;
            }}
        }} else {{
        T e = q*q/4 + p*p*p/27;

        if (fabs{single}(e) < {eps}) {{ // two roots
            r2 = -1.5*q/p - b/3;
            r3 = 3*q/p - b/3;
            T f_prime2 = 3*r2*r3 + 2*b*r2+c;
            T f_prime3 = 3*r3*r3 + 2*b*r3+c;
            if (fabs{single}(f_prime2) < fabs{single}(f_prime3)) {{
                r1 = r2;
            }} else {{
                r1 = r3;
            }}
        }} else if (e>0) {{
            // one root
            T u = cbrt{single}(-q/2 - sqrt{single}(e));
            r1 = u - p/(3*u) - b/3;
            r2 = r1;
            r3 = r1;
        }} else {{
            T u = 2*sqrt{single}(-p/3);
            T t = acos{single}(3*q/p/u)/3;
            T k = 2*3.14159265358979311599796346854/3;
            r1 = u*cos{single}(t) - b/3;
            r2 = u*cos{single}(t-k) - b/3;
            r3 = u*cos{single}(t-2*k) - b/3;
        }}
        }}
        }}
        // sort the roots
        T temp;
        if (r1<r2) {{
            if (r2>r3) {{
                if (r1<r3) {{
                    temp = r2;
                    r2 = r3;
                    r3 = temp;
                }} else {{
                    temp = r1;
                    r1 = r3;
                    r3 = r2;
                    r2 = temp;
                }}
            }}
        }} else {{
            if (r2<r3) {{
                if (r1<r3) {{
                    temp = r1;
                    r1 = r2;
                    r2 = temp;
                }} else {{
                    temp = r1;
                    r1 = r2;
                    r2 = r3;
                    r3 = temp;
                }}
            }} else {{
                temp = r1;
                r1 = r3;
                r3 = temp;
            }}
        }}
        return;
        ''', 'solve_cubic', return_tuple=True)
        return cubic_kernel

    def _solve_cubic_cuda(b, c, d):
        assert isinstance(b, cp.ndarray) and isinstance(c, cp.ndarray) and isinstance(d, cp.ndarray), \
            "Arrays must all be cupy arrays"
        assert b.dtype == c.dtype == d.dtype, "Array dtypes must match"
        if b.dtype not in _cuda_cubic_cache:
            _cuda_cubic_cache[b.dtype] = _make_cuda_cubic_solver(b.dtype)
        return _cuda_cubic_cache[b.dtype](b, c, d)

except ImportError:
    cp = None
    _solve_cubic_cuda = None


def _make_numba_cubic_solver(dtype):
    eps = slippy.CUBIC_EPS
    s_dtype = str(dtype)
    if not s_dtype.startswith("float"):
        raise ValueError("can only make cubic solver for single and double floats")

    def solve_cubic_numba_base(b, c, d, r1, r2, r3):
        for i in range(len(b)):
            if np.abs(d[i]) < eps:
                # cancel and find remaining roots by quadratic formula
                r1[i] = 0
                diff = np.sqrt(b[i] * b[i] - 4 * c[i]) / 2
                r2[i] = (-b[i]) / 2 + diff
                r3[i] = (-b[i]) / 2 - diff
            else:
                # convert to depressed cubic
                p = c[i] - b[i] ** 2 / 3
                q = 2 * b[i] ** 3 / 27 - b[i] * c[i] / 3 + d[i]
                if np.abs(p) < eps:
                    r1[i] = np.sign(-q) * np.abs(q) ** (1 / 3) - b[i] / 3
                    r2[i] = r1[i]
                    r3[i] = r1[i]
                elif np.abs(q) < eps:
                    r3[i] = - b[i] / 3
                    if p < 0:
                        diff = np.sqrt(-p)
                        r2[i] = diff - b[i] / 3
                        r1[i] = - diff - b[i] / 3
                    else:
                        r1[i] = r3[i]
                        r2[i] = r3[i]
                else:
                    e = q * q / 4 + p * p * p / 27
                    if np.abs(e) < eps:
                        r2[i] = -1.5 * q / p - b[i] / 3
                        r3[i] = 3 * q / p - b[i] / 3
                        f_prime2 = 3 * r2[i] ** 2 + 2 * b[i] * r2[i] + c[i]
                        f_prime3 = 3 * r3[i] ** 2 + 2 * b[i] * r3[i] + c[i]
                        if np.abs(f_prime2) < np.abs(f_prime3):
                            r1[i] = r2[i]
                        else:
                            r1[i] = r3[i]
                    elif e > 0:
                        u = -q / 2 - np.sqrt(e)
                        u = np.sign(u) * np.abs(u) ** (1 / 3)
                        r1[i] = u - p / (3 * u) - b[i] / 3
                        r2[i] = r1[i]
                        r3[i] = r1[i]
                    else:
                        u = 2 * np.sqrt(-p / 3)
                        t = np.arccos(3 * q / p / u) / 3
                        k = 2 * np.pi / 3
                        r1[i] = u * np.cos(t) - b[i] / 3
                        r2[i] = u * np.cos(t - k) - b[i] / 3
                        r3[i] = u * np.cos(t - 2 * k) - b[i] / 3
            # sort the array
            r1[i], r2[i], r3[i] = np.sort(np.array([r1[i], r2[i], r3[i]]))

    numba_type = numba.__getattribute__(s_dtype)
    raw_func = numba.guvectorize([(numba_type[:], numba_type[:], numba_type[:],
                                   numba_type[:], numba_type[:], numba_type[:])],
                                 "(n),(n),(n)->(n),(n),(n)",
                                 nopython=True)(solve_cubic_numba_base)

    def full_func(b, c, d):
        r1 = np.zeros_like(b)
        r2 = np.zeros_like(b)
        r3 = np.zeros_like(b)
        raw_func(b, c, d, r1, r2, r3)
        return r1, r2, r3

    return full_func


_numba_cubic_cache = {}


def _solve_cubic_numba(b, c, d):
    assert isinstance(b, np.ndarray) and isinstance(c, np.ndarray) and isinstance(d, np.ndarray), \
        "Arrays must all be numpy arrays"
    assert b.dtype == c.dtype == d.dtype, "Array dtypes must match"
    if b.dtype not in _numba_cubic_cache:
        _numba_cubic_cache[b.dtype] = _make_numba_cubic_solver(b.dtype)
    return _numba_cubic_cache[b.dtype](b, c, d)


def solve_cubic(b, c, d):
    """ Find roots of cubic equation x^3 + bx^2 + cx + d = 0

    Parameters
    ----------
    b, c, d: either numpy or cupy arrays
        Equation coefficients, must all have the same dtype and the same shape, currently supports any floats for numpy
        arrays and single or double floats for cupy arrays
    Returns
    -------
    r1, r2, r3: arrays
        Roots of the equation in ascending size order, arrays will match size, type and dtype of the input. All arrays
        will always be filled, if only a single root is found this will be repeated, where there are 2 roots, the root
        which does not represent a zero crossing will be repeated.
    Notes
    -----
    Both the cuda and numba versions use just in time compilation, functions are also cached for future calls

    """
    if isinstance(b, np.ndarray):
        return _solve_cubic_numba(b, c, d)
    elif cp is not None:
        if isinstance(b, cp.ndarray):
            return _solve_cubic_cuda(b, c, d)
    raise TypeError(f"Cannot solve cubic, unrecognised type {str(type(b))}")


def get_derived_stresses(tensor_components: dict, required_components: Sequence, delete: bool = True) -> dict:
    """Finds derived stress terms from the full stress tensor

    Parameters
    ----------
    tensor_components: dict
        The stress tensor components must have keys: 'xx', 'yy', 'zz', 'xy', 'yz', 'xz' all should be equal size
        arrays
    required_components: Sequence
        The required derived stresses, valid items are: '1', '2', '3' and/or 'vm', relating to principal stresses and
        von mises stress respectively. If tensor components are also present these will not be deleted if delete is
        set to True
    delete: bool, optional (True)
        If True the tensor components will be deleted after computation with the exception of components who's names
        are in required_components

    Returns
    -------
    dict of derived components

    """
    if not all([rc in {'1', '2', '3', 'vm'} for rc in required_components]):
        raise ValueError("Unrecognised derived stress component, allowed components are: '1', '2', '3', 'vm'")

    if isinstance(tensor_components['xx'], np.ndarray):
        xp = np
    else:
        try:
            float(tensor_components['xx'])
            xp = np
        except TypeError:
            xp = slippy.xp
    rtn_dict = dict()
    if 'vm' in required_components:
        rtn_dict['vm'] = xp.sqrt(((tensor_components['xx'] - tensor_components['yy']) ** 2 +
                                  (tensor_components['yy'] - tensor_components['zz']) ** 2 +
                                  (tensor_components['zz'] - tensor_components['xx']) ** 2 +
                                  6 * (tensor_components['xy'] ** 2 +
                                       tensor_components['yz'] ** 2 +
                                       tensor_components['xz'] ** 2)) / 2)
    if '1' in required_components or '2' in required_components or '3' in required_components:
        b = -(tensor_components['xx'] + tensor_components['yy'] + tensor_components['zz'])
        c = (tensor_components['xx'] * tensor_components['yy'] +
             tensor_components['yy'] * tensor_components['zz'] +
             tensor_components['xx'] * tensor_components['zz'] -
             tensor_components['xy'] ** 2 - tensor_components['xz'] ** 2 - tensor_components[
                 'yz'] ** 2)
        d = -((tensor_components['xx'] * tensor_components['yy'] * tensor_components['zz'] +
               2 * tensor_components['xy'] * tensor_components['xz'] * tensor_components['yz'] -
               tensor_components['xx'] * tensor_components['yz'] ** 2 -
               tensor_components['yy'] * tensor_components['xz'] ** 2 -
               tensor_components['zz'] * tensor_components['xy'] ** 2))

        rtn_dict['3'], rtn_dict['2'], rtn_dict['1'] = solve_cubic(b, c, d)
    return rtn_dict
