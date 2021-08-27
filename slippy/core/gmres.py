import slippy
import typing
import numpy as np

__all__ = ['gmres']

try:
    import cupy as cp

    def _cuda_rotmat(a: float, b: float) -> (float, float):
        """ Find the given's rotation matrix
        """
        t = cp.sqrt(a ** 2 + b ** 2)
        c = a / t
        s = b / t
        return c, s

    def _cuda_gmres(f: typing.Callable, x: cp.ndarray, b: cp.ndarray, restart: int, max_it: int, tol: float,
                    m_inv: typing.Callable = None):
        x = cp.array(x)
        n = x.size
        b = cp.array(b)
        precon = m_inv is not None
        it_num = 0
        flag = 0
        norm_b = cp.linalg.norm(b) or 1.0
        r = b - f(x)
        if precon:
            r = m_inv(r)
        error = cp.linalg.norm(r) / norm_b
        if error <= tol:
            return x, error, it_num, flag
        m = restart
        V = cp.zeros((n, m + 1))
        H = cp.zeros((m + 1, m))
        cs = cp.zeros(m)
        sn = cp.zeros(m)
        e1 = cp.zeros(n)
        e1[0] = 1.0
        while True:
            r = b - f(x)
            if precon:
                r = m_inv(r)
            norm_r = cp.linalg.norm(r)
            V[:, 0] = r / norm_r
            s = norm_r * e1
            for i in range(m):
                w = f(V[:, i])
                if precon:
                    w = m_inv(w)
                for k in range(i + 1):
                    H[k, i] = cp.dot(w, V[:, k])
                    w = w - H[k, i] * V[:, k]
                H[i + 1, i] = cp.linalg.norm(w)
                V[:, i + 1] = w / H[i + 1, i]
                for k in range(i):
                    temp = cs[k] * H[k, i] + sn[k] * H[k + 1, i]
                    H[k + 1, i] = -sn[k] * H[k, i] + cs[k] * H[k + 1, i]
                    H[k, i] = temp
                cs[i], sn[i] = _cuda_rotmat(H[i, i], H[i + 1, i])
                temp = cs[i] * s[i]
                norm_r = -sn[i] * s[i]
                if i + 1 < m:
                    s[i + 1] = norm_r
                s[i] = temp
                H[i, i] = cs[i] * H[i, i] + sn[i] * H[i + 1, i]
                H[i + 1, i] = 0.0
                error = cp.abs(norm_r) / norm_b
                if error <= tol:
                    y = cp.linalg.lstsq(H[:i + 1, :i + 1], s[:i + 1], rcond=None)[0]
                    x = x + cp.dot(V[:, :i + 1], y).flatten()
                    break
            if error <= tol:
                break
            y = cp.linalg.lstsq(H[:m + 1, :m + 1], s[:m + 1], rcond=None)[0]
            x = x + cp.dot(V[:, :m], y).flatten()
            r = b - f(x)
            if precon:
                r = m_inv(r)
            error = cp.linalg.norm(r) / norm_b
            if i + 1 < n:
                s[i + 1] = cp.linalg.norm(r)
            if error <= tol:
                break
            if max_it is not None and it_num >= max_it:
                break
            it_num += 1
        if error > tol:
            flag = 1
        return x, error, it_num, flag

except ImportError:
    _cuda_gmres = None
    cp = None


def _rotmat(a: float, b: float) -> (float, float):
    """ Find the given's rotation matrix
    """
    t = np.sqrt(a**2 + b**2)
    c = a/t
    s = b/t
    return c, s


def _fftw_gmres(f: typing.Callable, x: np.ndarray, b: np.ndarray, restart: int, max_it: int, tol: float,
                m_inv: typing.Callable = None):
    x = np.array(x)
    n = x.size
    b = np.array(b)
    precon = m_inv is not None
    it_num = 0
    flag = 0
    norm_b = np.linalg.norm(b) or 1.0
    r = b - f(x)
    if precon:
        r = m_inv(r)
    error = np.linalg.norm(r) / norm_b
    if error <= tol:
        return x, error, it_num, flag
    m = restart
    V = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))
    cs = np.zeros(m)
    sn = np.zeros(m)
    e1 = np.zeros(n)
    e1[0] = 1.0
    while True:
        r = b - f(x)
        if precon:
            r = m_inv(r)
        norm_r = np.linalg.norm(r)
        V[:, 0] = r / norm_r
        s = norm_r * e1
        for i in range(m):
            w = f(V[:, i])
            if precon:
                w = m_inv(w)
            for k in range(i + 1):
                H[k, i] = np.dot(w, V[:, k])
                w = w - H[k, i] * V[:, k]
            H[i + 1, i] = np.linalg.norm(w)
            V[:, i + 1] = w / H[i + 1, i]
            for k in range(i):
                temp = cs[k] * H[k, i] + sn[k] * H[k + 1, i]
                H[k + 1, i] = -sn[k] * H[k, i] + cs[k] * H[k + 1, i]
                H[k, i] = temp
            cs[i], sn[i] = _rotmat(H[i, i], H[i + 1, i])
            temp = cs[i] * s[i]
            norm_r = -sn[i] * s[i]
            if i + 1 < m:
                s[i + 1] = norm_r
            s[i] = temp
            H[i, i] = cs[i] * H[i, i] + sn[i] * H[i + 1, i]
            H[i + 1, i] = 0.0
            error = np.abs(norm_r) / norm_b
            if error <= tol:
                y = np.linalg.lstsq(H[:i + 1, :i + 1], s[:i + 1], rcond=None)[0]
                x = x + np.dot(V[:, :i + 1], y).flatten()
                break
        if error <= tol:
            break
        y = np.linalg.lstsq(H[:m + 1, :m + 1], s[:m + 1], rcond=None)[0]
        x = x + np.dot(V[:, :m], y).flatten()
        r = b - f(x)
        if precon:
            r = m_inv(r)
        error = np.linalg.norm(r) / norm_b
        if i + 1 < n:
            s[i + 1] = np.linalg.norm(r)
        if error <= tol:
            break
        if max_it is not None and it_num >= max_it:
            break
        it_num += 1
    if error > tol:
        flag = 1
    return x, error, it_num, flag


def gmres(f: typing.Callable, x0, b, restart: int, max_it: int, tol: float,
          m_inv: typing.Callable = None, override_cuda: bool = False):
    """Generalised minimum residual solver

    Parameters
    ----------
    f: callable
        A callable representing the matrix vector product of a linear transformation, if the matrix has been
        preconditioned m_inv should also be supplied
    x0: array like
        An initial guess for the solution to the linear system
    b: array like
        The right hand side of the system
    restart: int
        The number of iterations between each restart of the solver. A higher number of iterations generally leads to
        faster convergence but iterations are more computationally costly.
    max_it: int
        The maximum number of restarts used, note: the total number of iterations will be max_it*restart
    tol: float
        The normalised residual used for convergence. When norm(residual)/ norm(b) is smaller than this
        tolerance the solver will exit.
    m_inv: callable, optional (None)
        An optional inverse preconditioner
    override_cuda: bool, optional (False)
        If True this method will only use the CPU implementation, regardless of the slippy.CUDA property, otherwise uses
        the GPU implementation if slippy.CUDA is True
    Returns
    -------
    x: array like
        The solution vector
    failed: bool
        True of the solver failed to converge, otherwise False

    Examples
    --------
    >>> n = 6
    >>> A = np.tril(np.ones((n,n)))
    >>> b = np.ones(n)
    >>> x0 = b*0
    >>> f = lambda x: np.dot(A,x)
    >>> x, failed = gmres(f, x0, b, 4, n, 1e-6)
    x = array([ 1.00000160e+00, -1.62101904e-06,  2.15819307e-08, -1.05948284e-06, 2.47057595e-06, -1.95291141e-06])
    """
    if slippy.CUDA and not override_cuda:
        x, _, _, failed = _cuda_gmres(f, x0, b, restart, max_it, tol, m_inv)
    else:
        x, _, _, failed = _fftw_gmres(f, x0, b, restart, max_it, tol, m_inv)
    return x, failed
