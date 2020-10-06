import typing

import numpy as np

from slippy.abcs import _ContactModelABC

__all__ = ['get_gap_from_model', 'non_dimentional_height']


def non_dimentional_height(height: float, youngs: float, v: float, load: float, gs_x: float, gs_y: float = None,
                           inverse: bool = False, return_uz: bool = False):
    """Gives the non dimentional height from a dimentional height

    Parameters
    ----------
    height: float
        The height to be dimentionalised
    youngs: float
        The Young's modulus of the material
    v: float
        The Poisson's ratio of the material
    load: float
        The total load on the contact
    gs_x: float
        The grid spacing of the discrete grid in the x direction
    gs_y: float, optional (None)
        The grid spacing of the descretisation grid in the y direction, if None it is assumed that the grid is square
    inverse: bool, optional (False)
        If set to True the height will be re dimentionalised, else it will be non dimentionalised
    return_uz: bool, optional (False)
        If True the descriptive height uz will be returned

    Returns
    -------
    non_dimentional_height: float
        or the dimentionalised height if inverse is set to True

    Notes
    -----
    The height is non dimentionalised by dividing by the displacement caused by a the load on a single grid square:
    H = h/u_z
    u_z found according to equation 3.25 in the reference

    References
    ----------
    """
    load = load or 1
    a = gs_x
    b = gs_x if gs_y is None else gs_y
    c = (a**2+b**2)**0.5
    big_a = 2*a*np.log((b+c)/(c-b)) + 2*b*np.log((a+c)/(c-a))
    uz = big_a * load * (1-v**2) / np.pi / youngs / a / b
    if return_uz:
        return uz
    if inverse:
        return height*uz
    else:
        return height/uz


# noinspection PyUnresolvedReferences
def get_gap_from_model(model: _ContactModelABC, interference: float = 0,
                       off_set: typing.Sequence = (0, 0), mode: str = 'nearest',
                       periodic: bool = False):
    """

    Parameters
    ----------
    model : ContactModel
        An instance of a contact model containing two surfaces
    interference :
        The interference between the surfaces, from the point of first contact, positive is into the surface
    off_set
        The off set in the x and y directions between the origin of the first and second surface
    mode : str {'nearest', 'linear', 'cubic'} optional, 'nearest'
        The mode of interpolation used to generate the points on the second surface, see surface.interpolate for more
        information
    periodic : bool, optional (False)
        If True the second surface is considered periodic, the result will be the same shape and size as the first
        surface

    Returns
    -------
    point_wise_interference : np.ndarray
        point wise interference between the surfaces with the same grid spacing as the first surface in the contact
        model
    contact_points_1 : tuple[np.ndarray, np.ndarray]
        The x and y locations of the interference array in the same coordinates as the first surface
    contact_points_2 : tuple[np.ndarray, np.ndarray]
        The x and y locations of the interference array in the same coordinates as the second surface

    See Also
    --------
    slippy.surface._Surface.interpolate
    """
    if not isinstance(model, _ContactModelABC):
        raise ValueError("Model must be a contact model object")
    # Type checking
    if len(off_set) != 2:
        raise ValueError("off_set should be a two element sequence")
    if model.surface_2 is None:
        raise ValueError("Second surface must be set for this contact type")
    if not model.surface_1.is_discrete:
        raise ValueError("The master surface (surface 1) must be discrete to find the interference")

    if model.surface_2.is_discrete:
        # find overlap
        if periodic:
            contact_points_1 = model.surface_1.get_points_from_extent()
            contact_points_2y, contact_points_2x = contact_points_1[0] - off_set[0], contact_points_1[1] - off_set[1]
            contact_points_2 = (np.remainder(contact_points_2y, model.surface_2.extent[0]),
                                np.remainder(contact_points_2x, model.surface_2.extent[1]))
            sub_1 = model.surface_1.profile
            assert sub_1.shape == contact_points_1[0].shape == contact_points_2[0].shape
        else:  # not periodic
            extent_1y = (max(off_set[0], 0), min(model.surface_1.extent[0], off_set[0] + model.surface_2.extent[0]))
            slice_y = [ex / model.surface_1.grid_spacing for ex in extent_1y]
            slice_y[1] += 1
            extent_1x = (max(off_set[0], 0), min(model.surface_1.extent[0], off_set[0] + model.surface_2.extent[0]))
            slice_x = [ex / model.surface_1.grid_spacing for ex in extent_1x]
            slice_x[1] += 1
            contact_points_1 = np.meshgrid(np.arange(extent_1x[0], extent_1x[1] + model.surface_1.grid_spacing,
                                                     model.surface_1.grid_spacing),
                                           np.arange(extent_1y[0], extent_1y[1] + model.surface_1.grid_spacing,
                                                     model.surface_1.grid_spacing))
            contact_points_1 = (contact_points_1[1], contact_points_1[0])
            contact_points_2 = np.array(contact_points_1[0]) - off_set[0], np.array(contact_points_1[1]) - off_set[1]
            sub_1 = model.surface_1.profile[slice_x[0]:slice_x[1], slice_y[0]:slice_y[1]]
            assert sub_1.shape == contact_points_1.shape == contact_points_2.shape
        # interpolate using the required technique
        sub_2 = model.surface_2.interpolate(*contact_points_2, mode=mode)
        point_wise_interference = -sub_2 - sub_1
        point_wise_interference -= min(point_wise_interference.flatten()) + interference

    else:  # model.surface_2 is not descrete
        if not model.surface_2.is_analytic:
            raise ValueError("The second surface is not descretised or an analytical surface the interference "
                             "between the surfaces cannot be found")
        # find overlap extents (same as periodic)
        contact_points_1 = model.surface_1.get_points_from_extent()
        contact_points_2 = contact_points_1[0] - off_set[0], contact_points_1[1] - off_set[1]
        # call the height function on the second surface
        point_wise_interference = -model.surface_2.height(*contact_points_2) - model.surface_1.profile
        point_wise_interference -= min(point_wise_interference.flatten()) + interference

    return point_wise_interference, contact_points_1, contact_points_2
