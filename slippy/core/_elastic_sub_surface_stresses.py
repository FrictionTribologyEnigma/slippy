"""
Sub surface stresses for elastic materials from love and lee
"""
import numpy as np
import slippy
from collections.abc import Sequence

__all__ = ['normal_conv_kernels', 'tangential_conv_kernels', 'get_derived_stresses']


def normal_derivative_terms(x, y, z, grid_spacing, cuda):
    if cuda:
        xp = slippy.xp
    else:
        xp = np
    a = grid_spacing[1] / 2
    b = grid_spacing[0] / 2
    a1 = xp.sqrt((y - b) ** 2 + (x - a) ** 2 + z ** 2)
    b2 = xp.sqrt((y - b) ** 2 + (x + a) ** 2 + z ** 2)
    c3 = xp.sqrt((y + b) ** 2 + (x + a) ** 2 + z ** 2)
    d4 = xp.sqrt((y + b) ** 2 + (x - a) ** 2 + z ** 2)
    chi_x_x = (xp.arctan((b - y) / (a - x)) + xp.arctan((b + y) / (a - x))
               - xp.arctan(z * (b - y) / (a1 * (a - x))) - xp.arctan(z * (b + y) / (d4 * (a - x))) +
               xp.arctan((b - y) / (a + x)) + xp.arctan((b + y) / (a + x))
               - xp.arctan(z * (b - y) / (b2 * (a + x))) - xp.arctan(z * (b + y) / (c3 * (a + x))))
    chi_y_y = (xp.arctan((a - x) / (b - y)) + xp.arctan((a + x) / (b - y))
               - xp.arctan(z * (a - x) / (a1 * (b - y))) - xp.arctan(z * (a + x) / (b2 * (b - y))) +
               xp.arctan((a - x) / (b + y)) + xp.arctan((a + x) / (b + y))
               - xp.arctan(z * (a - x) / (d4 * (b + y))) - xp.arctan(z * (a + x) / (c3 * (b + y))))
    z_2 = z ** 2
    vee_z = -(2 * xp.pi
              - xp.arccos(((a - x) * (b - y)) * ((a - x) ** 2 + z_2) ** -0.5 * ((b - y) ** 2 + z_2) ** -0.5)
              - xp.arccos(((a - x) * (b + y)) * ((a - x) ** 2 + z_2) ** -0.5 * ((b + y) ** 2 + z_2) ** -0.5)
              - xp.arccos(((a + x) * (b - y)) * ((a + x) ** 2 + z_2) ** -0.5 * ((b - y) ** 2 + z_2) ** -0.5)
              - xp.arccos(((a + x) * (b + y)) * ((a + x) ** 2 + z_2) ** -0.5 * ((b + y) ** 2 + z_2) ** -0.5))
    chi_x_y = xp.log(((z + a1) * (z + c3)) / ((z + b2) * (z + d4)))
    vee_x_x = -((a - x) / ((a - x) ** 2 + z_2) * ((b - y) / a1 + (b + y) / d4) +
                (a + x) / ((a + x) ** 2 + z_2) * ((b - y) / b2 + (b + y) / c3))
    vee_y_y = -((b - y) / ((b - y) ** 2 + z_2) * ((a - x) / a1 + (a + x) / b2) +
                (b + y) / ((b + y) ** 2 + z_2) * ((a - x) / d4 + (a + x) / c3))
    vee_z_z = -vee_x_x - vee_y_y
    vee_x_z = z / ((a - x) ** 2 + z_2) * ((b - y) / a1 + (b + y) / d4) - z / ((a + x) ** 2 + z_2) * (
        (b - y) / b2 + (b + y) / c3)
    vee_y_z = z / ((b - y) ** 2 + z_2) * ((a - x) / a1 + (a + x) / b2) - z / ((b + y) ** 2 + z_2) * (
        (a - x) / d4 + (a + x) / c3)
    vee_x_y = 1 / a1 + 1 / c3 - 1 / b2 - 1 / d4
    return chi_x_x, chi_y_y, chi_x_y, vee_z, vee_x_x, vee_y_y, vee_z_z, vee_x_z, vee_y_z, vee_x_y


def normal_conv_kernels(span, z, grid_spacing, young, v, cuda=False) -> dict:
    """Get the convolution kernels for the subsurface stresses cause by a normal load in an elastic material

    Parameters
    ----------
    span: Sequence[int, int]
        The span of the convolution kernels in the y and x directions
    z: Sequence[float]
        The heights of interest in the solid
    grid_spacing: Sequence[float, float] or float
        Either a two element sequence of floats, giving the grid spacing in each direction or, a single value,
        indicating a square grid
    young: float
        The Young's modulus of the material
    v: float
        The Poisson's ratio of the material
    cuda: bool, optional (False)
        If True kernels will be made on the GPU

    Returns
    -------
    dict with keys: sxx, syy, szz, sxy, syz, sxz
        The influence matrices for each of the stress components, numpy arrays if cuda is set to false or cupy could not
        be imported else cupy arrays

    References
    ----------
    A. E. H., L. (1929). The stress produced in a semi-infinite solid by pressure on part of the boundary. Philosophical
    Transactions of the Royal Society of London. Series A, Containing Papers of a Mathematical or Physical Character,
    228(659–669), 377–420. https://doi.org/10.1098/rsta.1929.0009
    """
    if not isinstance(grid_spacing, Sequence):
        grid_spacing = (grid_spacing, grid_spacing)
    if len(grid_spacing) == 1:
        grid_spacing = (grid_spacing[0], grid_spacing[0])
    if len(grid_spacing) != 2:
        raise ValueError("Grid spacing should be a number or two element sequence")

    if cuda:
        xp = slippy.xp
    else:
        xp = np

    x = grid_spacing[1] * (xp.arange(span[1]) - span[1] // 2 + (1 - span[1] % 2))
    y = grid_spacing[0] * (xp.arange(span[0]) - span[0] // 2 + (1 - span[0] % 2))
    x = x.reshape((1, 1, -1))
    y = y.reshape((1, -1, 1))
    z = xp.array(z).reshape((-1, 1, 1))
    chi_x_x, chi_y_y, chi_x_y, vee_z, vee_x_x, vee_y_y, vee_z_z, \
        vee_x_z, vee_y_z, vee_x_y = normal_derivative_terms(x, y, z, grid_spacing, cuda)
    shear = young / (2 * (1 + v))
    lam = young * v / ((1 + v) * (1 - 2 * v))
    sxx = 1 / (2 * xp.pi) * (lam / (lam + shear) * vee_z - shear / (lam + shear) * chi_x_x - z * vee_x_x)
    syy = 1 / (2 * xp.pi) * (lam / (lam + shear) * vee_z - shear / (lam + shear) * chi_y_y - z * vee_y_y)
    szz = 1 / (2 * xp.pi) * (vee_z - z * vee_z_z)
    syz = -1 / (2 * xp.pi) * z * vee_y_z
    sxz = -1 / (2 * xp.pi) * z * vee_x_z
    sxy = -1 / (2 * xp.pi) * (shear / (lam + shear) * chi_x_y + z * vee_x_y)
    return {'xx': sxx, 'yy': syy, 'zz': szz, 'xy': sxy, 'yz': syz, 'xz': sxz}


def tangential_derivative_terms(x, y, z, grid_spacing, cuda):
    if cuda:
        xp = slippy.xp
    else:
        xp = np
    a = grid_spacing[1] / 2
    b = grid_spacing[0] / 2
    x1 = -a - x
    x2 = a - x
    y1 = -b - y
    y2 = b - y
    z2 = z ** 2
    rho1 = xp.sqrt(x1 ** 2 + y1 ** 2 + z2)
    rho2 = xp.sqrt(x1 ** 2 + y2 ** 2 + z2)
    rho3 = xp.sqrt(x2 ** 2 + y1 ** 2 + z2)
    rho4 = xp.sqrt(x2 ** 2 + y2 ** 2 + z2)
    f_x_z = xp.log(rho3 + y1) + xp.log(rho2 + y2) - xp.log(rho1 + y1) - xp.log(rho4 + y2)
    f_x_x_x = (y2 / (x2 ** 2 + y2 ** 2) - y1 / (x2 ** 2 + y1 ** 2) - y2 / (x1 ** 2 + y2 ** 2) + y1 / (x1 ** 2 + y1 ** 2)
               - z * y2 * (rho4 ** 2 + x2 ** 2) / (rho4 * (y2 ** 2 * z2 + x2 ** 2 * rho4 ** 2))
               + z * y1 * (rho3 ** 2 + x2 ** 2) / (rho3 * (y1 ** 2 * z2 + x2 ** 2 * rho3 ** 2))
               + z * y2 * (rho2 ** 2 + x1 ** 2) / (rho2 * (y2 ** 2 * z2 + x1 ** 2 * rho2 ** 2))
               - z * y1 * (rho1 ** 2 + x1 ** 2) / (rho1 * (y1 ** 2 * z2 + x1 ** 2 * rho1 ** 2)))
    f1_x_x_x = (z * f_x_x_x
                + xp.log((rho4 + y2) * (rho1 + y1) / ((rho2 + y2) * (rho3 + y1)))
                + x2 ** 2 * (1 / (rho4 * (rho4 + y2)) - 1 / (rho3 * (rho3 + y1)))
                - x1 ** 2 * (1 / (rho2 * (rho2 + y2)) - 1 / (rho1 * (rho1 + y1))))  # corrected from paper
    f1_x_y_y = y2 / (rho4 + z) - y1 / (rho3 + z) - y2 / (rho2 + z) + y1 / (rho1 + z)
    f1_x_x_y = x2 / (rho4 + z) - x1 / (rho2 + z) - x2 / (rho3 + z) + x1 / (rho1 + z)  # corrected from paper
    f_x_y_y = -y2 / (rho4 * (rho4 + z)) + y1 / (rho3 * (rho3 + z)) + y2 / (rho2 * (rho2 + z)) - y1 / (rho1 * (rho1 + z))
    f_x_x_y = -x2 / (rho4 * (rho4 + z)) + x1 / (rho2 * (rho2 + z)) + x2 / (rho3 * (rho3 + z)) - x1 / (
        rho1 * (rho1 + z))  # corrected from paper
    f_x_z_z = -z / (rho4 * (rho4 + y2)) + z / (rho3 * (rho3 + y1)) + z / (rho2 * (rho2 + y2)) - z / (rho1 * (rho1 + y1))
    f_x_x_z = x2 / (rho4 * (rho4 + y2)) - x2 / (rho3 * (rho3 + y1)) - x1 / (rho2 * (rho2 + y2)) + x1 / (
        rho1 * (rho1 + y1))
    f_z_y = -xp.log(rho4 + x2) + xp.log(rho3 + x2) + xp.log(rho2 + x1) - xp.log(rho1 + x1)
    f_z_z = (-xp.arctan(x2 * y2 / (z * rho4)) + xp.arctan(x2 * y1 / (z * rho3))
             + xp.arctan(x1 * y2 / (z * rho2)) - xp.arctan(x1 * y1 / (z * rho1)))
    f_x_y_z = 1 / rho4 - 1 / rho3 - 1 / rho2 + 1 / rho1
    return f_x_z, f_x_x_x, f1_x_x_x, f1_x_y_y, f1_x_x_y, f_x_y_y, f_x_x_y, f_x_z_z, f_x_x_z, f_z_y, f_z_z, f_x_y_z


def tangential_conv_kernels(span, z, grid_spacing, v, cuda=False) -> dict:
    """Get the convolution kernels for the subsurface stresses cause by a tangential traction in an elastic material

    Parameters
    ----------
    span: Sequence[int, int]
        The span of the convolution kernels in the y and x directions
    z: Sequence[float]
        The heights of interest in the solid
    grid_spacing: Sequence[float, float] or float
        Either a two element sequence of floats, giving the grid spacing in each direction or, a single value,
        indicating a square grid
    v: float
        The Poisson's ratio of the material
    cuda: bool, optional (False)
        If True kernels will be made on the GPU

    Returns
    -------
    dict with keys: sxx, syy, szz, sxy, syz, sxz
        The influence matrices for each of the stress components, numpy arrays if cuda is set to false or cupy could not
        be imported else cupy arrays

    References
    ----------
    Lee, M. J., Gu, Y. P., & Jo, Y. J. (2000). The Stress Field in the Body by Tangential Loading of a Rectangular Patch
    on a Semi-Infinite Solid. Transactions of the Korean Society of Mechanical Engineers A, 24(4), 1032-1038.
    https://doi.org/10.22634/KSME-A.2000.24.4.1032 (In korean)

    Notes
    -----
    Several changes from the original paper have been made for this implementation, these were assumed to be typos,
    they are indicated in the code.
    """
    if not isinstance(grid_spacing, Sequence):
        grid_spacing = (grid_spacing, grid_spacing)
    if len(grid_spacing) == 1:
        grid_spacing = (grid_spacing[0], grid_spacing[0])
    if len(grid_spacing) != 2:
        raise ValueError("Grid spacing should be a number or two element sequence")

    if cuda:
        xp = slippy.xp
    else:
        xp = np
    x = grid_spacing[1] * (xp.arange(span[1]) - span[1] // 2 + (1 - span[1] % 2))
    y = grid_spacing[0] * (xp.arange(span[0]) - span[0] // 2 + (1 - span[0] % 2))
    x = x.reshape((1, 1, -1))
    y = y.reshape((1, -1, 1))
    z = xp.array(z).reshape((-1, 1, 1))
    f_x_z, f_x_x_x, f1_x_x_x, f1_x_y_y, f1_x_x_y, f_x_y_y, f_x_x_y, f_x_z_z, f_x_x_z, f_z_y, \
        f_z_z, f_x_y_z = tangential_derivative_terms(x, y, z, grid_spacing, cuda)
    pi = xp.pi
    sxx = (v + 1) / pi * f_x_z + 1 / (2 * pi) * ((2 * v * f1_x_x_x) - z * f_x_x_x)
    syy = v / pi * f_x_z + 1 / (2 * pi) * ((2 * v * f1_x_y_y) - z * f_x_y_y)
    szz = -z / (2 * pi) * f_x_z_z
    sxy = 1 / (2 * pi) * (f_z_y + 2 * v * f1_x_x_y - z * f_x_x_y)
    syz = -z / (2 * pi) * f_x_y_z
    sxz = 1 / (2 * pi) * (f_z_z - z * f_x_x_z)
    return {'xx': sxx, 'yy': syy, 'zz': szz, 'xy': sxy, 'yz': syz, 'xz': sxz}


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
        b = tensor_components['xx'] + tensor_components['yy'] + tensor_components['zz']
        c = (tensor_components['xx'] * tensor_components['yy'] +
             tensor_components['yy'] * tensor_components['zz'] +
             tensor_components['xx'] * tensor_components['zz'] -
             tensor_components['xy'] ** 2 - tensor_components['xz'] ** 2 - tensor_components[
                 'yz'] ** 2)
        d = (tensor_components['xx'] * tensor_components['yy'] * tensor_components['zz'] +
             2 * tensor_components['xy'] * tensor_components['xz'] * tensor_components['yz'] -
             tensor_components['xx'] * tensor_components['yz'] ** 2 -
             tensor_components['yy'] * tensor_components['xz'] ** 2 -
             tensor_components['zz'] * tensor_components['xy'] ** 2)

        p = c - (b ** 2) / 3
        q = ((2 / 27) * b ** 3 - (1 / 3) * b * c + d)
        del c
        del d
        if delete:
            for key in list(tensor_components.keys()):
                if key not in required_components:
                    del tensor_components[key]
        principals = xp.zeros((3,) + b.shape)
        for i in range(3):
            principals[i] = 2 * xp.sqrt((-1 / 3) * p) * xp.cos(
                1 / 3 * xp.arccos(3 * q / (2 * p) * xp.sqrt(-3 / p)) - 2 * xp.pi * i / 3) - b / 3
            #                ^ real roots from cubic equation for depressed cubic                          ^
            #                                                                               change of variable
        rtn_dict['1'] = xp.max(principals, 0)
        rtn_dict['3'] = xp.min(principals, 0)
        rtn_dict['2'] = b - rtn_dict['1'] - rtn_dict['3']
    return rtn_dict
