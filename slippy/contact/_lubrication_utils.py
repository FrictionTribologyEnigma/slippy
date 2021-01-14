import numpy as np
from scipy.linalg.lapack import dgtsv

__all__ = ['tdma', 'cyclic_tdma']


def tdma(lower_diagonal, main_diagonal, upper_diagonal, right_hand_side):
    """The thomas algorithm (tdma) solution for tri-diagonal matrix inversion

    Parameters
    ----------
    lower_diagonal: np.ndarray
        The lower diagonal of the matrix length n-1
    main_diagonal: np.ndarray
        The main diagonal of the matrix length n
    upper_diagonal: np.ndarray
        The upper diagonal of the matrix length n-1
    right_hand_side: np.ndarray
        The array bof size max(1, ldb*n_rhs) for column major layout and max(1, ldb*n) for row major layout contains the
        matrix B whose columns are the right-hand sides for the systems of equations.

    Returns
    -------
    x: np.ndarray
        The solution array length n

    Notes
    -----
    Nothing is mutated by this function
    """
    _, _, _, x, _ = dgtsv(lower_diagonal, main_diagonal, upper_diagonal, right_hand_side)
    return x


def cyclic_tdma(lower_diagonal, main_diagonal, upper_diagonal, right_hand_side):
    """The thomas algorithm (TDMA) solution for tri-diagonal matrix inversion with the sherman morison formula applied

    Parameters
    ----------
    lower_diagonal: np.ndarray
        The lower diagonal of the matrix length n, the first element is taken to be the top right element of the matrix
    main_diagonal: np.ndarray
        The main diagonal of the matrix length n
    upper_diagonal: np.ndarray
        The upper diagonal of the matrix length n, the last element is taken to be the bottom left element of the matrix
    right_hand_side: np.ndarray
        The right hand side of the equation

    Returns
    -------
    x: np.ndarray
        The solution array length n

    Notes
    -----
    Nothing is mutated by this function
    """
    # modify b
    gamma = -main_diagonal[0] if main_diagonal[0] else 1.0
    main_diagonal[0] = main_diagonal[0] - gamma
    main_diagonal[-1] = main_diagonal[-1] - lower_diagonal[0] * upper_diagonal[-1] / gamma
    # find Ax=rhs
    _, _, _, x, _ = dgtsv(lower_diagonal[1:], main_diagonal, upper_diagonal[:-1], right_hand_side)
    # make u
    u = np.zeros_like(right_hand_side)
    u[0] = gamma
    u[-1] = upper_diagonal[-1]
    # find Az=u
    _, _, _, z, _ = dgtsv(lower_diagonal[1:], main_diagonal, upper_diagonal[:-1], u)

    # find the factor from the second part of SM formula
    factor = (x[0] + x[-1] * lower_diagonal[0] / gamma) / (1 + z[0] + z[-1] * lower_diagonal[0] / gamma)

    return x - z * factor
