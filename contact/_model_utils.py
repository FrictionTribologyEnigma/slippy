from .models import ContactModel
import typing
import numpy as np
from slippy.abcs import _SurfaceABC

__all__ = ['get_gap_from_model']


# noinspection PyUnresolvedReferences
def get_gap_from_model(model: ContactModel, interferance: float,
                       off_set: typing.Sequence = (0, 0), mode: str = 'nearest',
                       periodic: bool = False):
    """

    Parameters
    ----------
    model : ContactModel
        An instance of a contact model containing two surfaces
    interferance :
        The interferance between the surfaces, from the point of first contact, positive is into the surface
    off_set
        The off set in the x and y directions between the origin of the first and second surface
    mode : str {'nearest', 'linear', 'cubic'} optional, 'nearest'
        The mode of interpolaion used to generate the points on the second surface, see surface.interpolate for more
        information
    periodic : bool, optioanl (False)
        If True the second surface is considered periodic, the result will be the same shape and size as the first
        surface

    Returns
    -------
    point_wise_interferance : np.ndarray
        pointwise interferance between the surfaces with the same grid spacing as the first surface in the contact
        model
    contact_points_1 : tuple[np.ndarray, np.ndarray]
        The x and y locations of the interferance array in the same coordinates as the first surface
    contact_points_2 : tuple[np.ndarray, np.ndarray]
        The x and y loactions of the interferance array in the same coordinates as the first surface

    See Also
    --------
    slippy.surface._Surface.interpolate
    """
    # Type checking
    if len(off_set) != 2:
        raise ValueError("off_set should be a two element sequence")
    if model.surface_2 is None:
        raise ValueError("Second surface must be set for this contact type")
    if not model.surface_1.is_descrete:
        raise ValueError("The master surface (surface 1) must be descretised to find the interferance")

    if model.surface_2.is_descrete:
        # find overlap
        if periodic:
            contact_points_1 = model.surface_1.get_points_from_extent()
            contact_points_2x, contact_points_2y = contact_points_1[0] - off_set[0], contact_points_1[1] - off_set[1]
            contact_points_2 = (np.fmod(contact_points_2x, model.surface_2.extent[0]),
                                np.fmod(contact_points_2y, model.surface_2.extent[1]))
            sub_1 = model.surface_1.profile
            assert sub_1.shape == contact_points_1.shape == contact_points_2.shape
        else:  # not periodic
            extent_1x = (max(off_set[0], 0), min(model.surface_1.extent[0], off_set[0] + model.surface_2.extent[0]))
            slice_x = [ex / model.surface_1.grid_spacing for ex in extent_1x]
            slice_x[1] += 1
            extent_1y = (max(off_set[0], 0), min(model.surface_1.extent[0], off_set[0] + model.surface_2.extent[0]))
            slice_y = [ex / model.surface_1.grid_spacing for ex in extent_1y]
            slice_y[1] += 1
            contact_points_1 = np.meshgrid(np.arange(extent_1x[0], extent_1x[1] + model.surface_1.grid_spacing,
                                                     model.surface_1.grid_spacing),
                                           np.arange(extent_1y[0], extent_1y[1] + model.surface_1.grid_spacing,
                                                     model.surface_1.grid_spacing))
            contact_points_2 = np.array(contact_points_1[0]) - off_set[0], np.array(contact_points_1[1]) - off_set[1]
            sub_1 = model.surface_1.profile[slice_x[0]:slice_x[1], slice_y[0]:slice_y[1]]
            assert sub_1.shape == contact_points_1.shape == contact_points_2.shape
        # interpolate using the required technique
        sub_2 = model.surface_2.interpolate(*contact_points_2, mode=mode)
        point_wise_interferance = sub_1 - sub_2
        point_wise_interferance -= min(point_wise_interferance.flatten()) + interferance

    else:  # model.surface_2 is not descrete
        if not model.surface_2.is_analytical:
            raise ValueError("The second surface is not descretised or an analytical surface the interferance "
                             "between the surfaces cannot be found")
        # find overlap extents (same as periodic)
        contact_points_1 = model.surface_1.get_points_from_extent()
        contact_points_2 = contact_points_1[0] - off_set[0], contact_points_1[1] - off_set[1]
        # call the height function on the second surface
        point_wise_interferance = model.surface_1.profile - model.surface_2.height(*contact_points_2)
        point_wise_interferance -= min(point_wise_interferance.flatten()) + interferance

    return point_wise_interferance, contact_points_1, contact_points_2
