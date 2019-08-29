"""Elastic BEM extended to contact between one rigid body and one elastic body"""

# TODO do rigid first then extend to full

import slippy.surface as surface
from collections import namedtuple
from typing import Sequence, Tuple
import numpy as np
from contact.materials import rigid

__all__ = ['rigid_defined_load', 'rigid_defined_interferance', 'rigid_closure_plot']

BEMResult = namedtuple('BEMResult', ['surface_loads', 'total_deflection', 'surface_1_deflection',
                                     'surface_2_deflection'], defaults=(None,)*4)


def rigid_defined_load(load: float, surface_1: surface.Surface, surface_2: surface.Surface = None,
                       invert_surface: str = '', full_results: bool = True, full_analysis: bool = False,
                       influence_matrix_span: Sequence[int] = None,
                       inner_loop_tolerance: float = 1e-4, inner_loop_max_iter: int = 100,
                       outer_loop_tolerance: float = 1e-5, outer_loop_max_iter: int = None,
                       starting_position: float = None):
    """
    Solves the rigid contact problem for a specified load between two rough surfaces

    Parameters
    ----------
    load: float
        The load applied between the surfaces
    surface_1: slippy.surface.Surface
        The non rigid surface, should have a material defined
    surface_2: slippy.surface.Surface, optional (flat surface (z=0))
        The rigid surface, if not given defaults to a flat plane at z=0
    invert_surface: str, optional ('') {'','1','2','12'}
        If surfaces need to be inverted before analysis, forinstance if both surfaces come from profilometer scans,
        to invert surface_1 include '1' in the string to invert surface_2 include '2' in the string
    full_results: bool, optional (True)
        If True the full results will be returned including loads and displacements at each point on the surface, if
        false only the displacement between the surfaces will be returned
    full_analysis: bool (False)
        If False loads on the surface will only casue displacemetns in the direction of the load, other wise loads on
        the surface will cause displacements in all directions
    influence_matrix_span: Sequence[int], optional (None)
        The span of the influence matrix used in the BEM calculation in grid points, defaults to the same as the size of
        the surface in grid points
    inner_loop_tolerance: float, optional (1e-4)
        The tollerance on the norm of the resuidual for the optimisation loop used to find the loads that cause a
        specified displacement in the BEM calculation
    inner_loop_max_iter: int, optional (100)
        The maximum number of iterations used in one run of the inner optimisation loop which finds the loads needed to
        cause a specified displacement
    outer_loop_tolerance: float, optional (1e-5)
        The tolerance on the total load used to define convergence of the outer loop which optimises the interferance
        between the surfaces to find the load
    outer_loop_max_iter: int, optional (1000)
        The maximum number of iterations used in the outer loop, which optimises the interfereance between the surfaces
        to give the specified load

    Returns
    -------
    results : BEMResult
        A named tuple of results
    """
    pass


def rigid_defined_interferance(interferance: float, surface_1: surface.Surface, surface_2: surface.Surface = None,
                               absolute_interferance: bool = False, invert_surface: str = '',
                               offset: tuple = (0, 0), cyclic: bool = False, interpolate: bool = False,
                               full_results: bool = True, simple_analysis: bool = True,
                               influence_matrix_span: Sequence[int] = None,
                               inner_loop_tolerance: float = 1e-4, inner_loop_max_iter: int = 100):
    """
    Solves the rigid contact problem for a specified load between two rough surfaces

    Parameters
    ----------
    interferance: float
        The interferance between the two surfaces from the point of first touching, or as an offset from the heights if
        the absolute_interferance keyword argument is set to True, see notes for example
    surface_1: slippy.surface.Surface
        The non rigid surface, should have a material defined
    surface_2: slippy.surface.Surface, optional (flat surface (z=0))
        The rigid surface, if not given defaults to a flat plane at z=0
    absolute_interferance: bool, optional (False
        If set to true the interferance is used as an absolute value, ofsetting the surfaces from their current position
        can be useful to compare randome surfaces
    invert_surface: str, optional ('') {'','1','2','12'}
        If surfaces need to be inverted before analysis, forinstance if both surfaces come from profilometer scans,
        to invert surface_1 include '1' in the string to invert surface_2 include '2' in the string
    offset: tuple optional ((0, 0))
        The offset between the surfaces, in the same units as the grid_spacing properties of the surfaces, the second
        surface is moved in the positive x and y directions by this amount
    cyclic: bool optional (False)
        Set to True if the surfaces are periodic and the offset can be applied by wrapping the second surface.
    interpolate: bool optional (False)
        Set to True to interpolate the surface values from the offset, otherwise the nearest value is taken for discrete
        surfaces, point heights for analytical surfaces are generated regardless of this setting
    full_results: bool, optional (True)
        If True the full results will be returned including loads and displacements at each point on the surface, if
        false only the displacement between the surfaces will be returned
    simple_analysis: bool (True)
        If True loads on the surface will only casue displacemetns in the direction of the load, other wise loads on
        the surface will cause displacements in all directions
    influence_matrix_span: Sequence[int], optional (None)
        The span of the influence matrix used in the BEM calculation in grid points, defaults to the same as the size of
        the surface in grid points
    inner_loop_tolerance: float, optional (1e-4)
        The tollerance on the norm of the resuidual for the optimisation loop used to find the loads that cause a
        specified displacement in the BEM calculation
    inner_loop_max_iter: int, optional (100)
        The maximum number of iterations used in one run of the inner optimisation loop which finds the loads needed to
        cause a specified displacement

    Returns
    -------
    results : BEMResult
        A named tuple of results
    """
    # check if both have materials and both are elastic or one is rigid and the other elastic
    if not surface_1.material.material_type == 'Elastic':
        raise ValueError("First surface must be elastic")
    if surface_2 is None:
        surface_2 = surface.FlatSurface((0, 0))
        surface_2.material = rigid

    # Invert surfaces if they need it
    if '1' in invert_surface:
        surface_1.invert_surface = True
    if '2' in invert_surface:
        surface_2.invert_surface = True

    # workout overlap
    grid_spacing = surface_1.grid_spacing
    if surface_2.is_analytic and not surface_2.is_descrete:
        # call the surface height function to give the surface heights, remember to mod on the extent if one is there
        # and it is cyclic
        if cyclic or surface_2.extent is None:
            worn_sub_profile_1 = surface_1.worn_profile
            x_mesh, y_mesh = surface_1._get_points_from_extent(surface_1.extent, surface_1.grid_spacing)
            x_mesh -= offset[0]
            y_mesh -= offset[1]
            if surface_2.extent is not None:
                x_mesh = x_mesh % surface_2.extent[0]
                y_mesh = y_mesh % surface_2.extent[1]

            worn_sub_profile_2 = surface_2.height(x_mesh, y_mesh)
        else:
            # is analytic but not cyclic, and surface 2 has an extent
            start = [max([0, os]) for os in offset]
            end = [min(s1_ex, s2_ex + os) for s1_ex, s2_ex, os in zip(surface_1.extent, surface_2.extent, offset)]
            # TODO need worn_sub_profile_1 and 2, wear is not supported for analytic surfaces, should make a subclass
            #  for analytic surfaces

    elif surface_2.is_descrete:
        # TODO need worn_sub_profile_1 and 2
        offset_grid_pts = [os/surface_2.grid_spacing for os in offset]
    else:
        raise ValueError('Second surface must be discrete or analytic')

    # Work out element wise interferance (using worn profiles and total int)
    if absolute_interferance:
        interferance_each_point = worn_sub_profile_2 - worn_sub_profile_1 - interferance
    else:
        min_gap = min((worn_sub_profile_2 - worn_sub_profile_1).flatten())
        interferance_each_point = worn_sub_profile_2 - worn_sub_profile_1 - min_gap - interferance
    interferance_each_point = np.clip(interferance_each_point, None, 0)
    interferance_each_point[interferance_each_point == 0] = np.nan
    displacement = Displacements(z=interferance_each_point)
    # solve problem
    loads, full_displacement = elastic_displacement(displacement, grid_spacing=grid_spacing,
                                                    shear_modulus=surface_1.material.G, v=surface_1.material.v,
                                                    shear_modulus_2=surface_2.material.G, v_2=surface_2.material.v,
                                                    span=influence_matrix_span, simple=simple_analysis,
                                                    tol=inner_loop_tolerance, max_it=inner_loop_max_iter)

    # TODO update solve given displacement to give option of no negative loads or negative loads allowed/ check if
    #  negative loads are produced

    # move results round to undo the offset or wrapping

    if not full_results:
        return BEMResult(surface_loads=loads, total_deflection=full_displacement, surface_1_deflection=None,
                         surface_2_deflection=None)
    if full_results:
        if simple_analysis:
            disp_1, disp_2 = disambiguate_displacement(displacement=full_displacement,
                                                       shear_modulus_1=surface_1.material.G, v_1=surface_1.material.v,
                                                       shear_modulus_2=surface_2.material.G, v_2=surface_2.material.v)
        else:
            disp_1 = elastic_loading(loads, grid_spacing=grid_spacing, v=surface_1.material.v,
                                     shear_modulus=surface_1.material.G, span=influence_matrix_span, simple=False,
                                     deflections='xyz')
            disp_2 = elastic_loading(loads, grid_spacing=grid_spacing, v=surface_2.material.v,
                                     shear_modulus=surface_2.material.G, span=influence_matrix_span, simple=False,
                                     deflections='xyz')
    else:
        disp_1, disp_2 = None, None
    # reinterpolate the results on the second surface if needed

    # return results


def rigid_closure_plot():
    """
    Finds a displacement vs load plot for two rough surfaces if one of the surfaces is rigid

    """
    pass


if __name__ == '__main__':
    # test basics, errors if no surfaces etc

    # test defined load
    # basic results
    # full results

    # test defined interferance
    # basic results
    # full results

    # test both work as inverse of eachother

    # test closure plot results line up with individual solutions
    # test closure plot covers a reasnoble range (0 to full contact)
    pass
