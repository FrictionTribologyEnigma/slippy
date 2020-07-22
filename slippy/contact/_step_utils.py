"""
Helper functions for steps
"""
import os
import typing
import warnings
from collections import namedtuple
from numbers import Number

import numpy as np
from scipy.signal import fftconvolve

import slippy
from slippy.abcs import _ContactModelABC
from ._material_utils import Loads, Displacements
from .influence_matrix_utils import bccg, plan_convolve, guess_loads_from_displacement
from .materials import _IMMaterial

__all__ = ['solve_normal_interference', 'get_next_file_num', 'OffSetOptions', 'solve_normal_loading']

OffSetOptions = namedtuple('off_set_options', ['off_set', 'abs_off_set', 'periodic', 'interpolation_mode'])


def solve_normal_loading(loads: Loads, model: _ContactModelABC, current_state: dict,
                         deflections: str = 'xyz', material_options: list = None,
                         reverse_loads_on_second_surface: str = ''):
    """

    Parameters
    ----------
    loads: Loads
        The loads on the surface in the same units as the material object
    model: _ContactModelABC
        A contact model object containing the surfaces and materials to be used
    current_state: dict
        The state dict for the model before this step is called
    deflections: str, optional ('xyz')
        The directions of the deflections to be calculated for each surface
    material_options: dict, optional (None)
        list of Dicts of options to be passed to the displacement_from_surface_loads method of the surface
    reverse_loads_on_second_surface: str, optional ('')
        string containing the components of the loads to be reversed for the second surface for example 'x' will reverse
        loads in the x direction only

    Returns
    -------
    total_displacement: Displacements
        A named tuple of the total displacement
    surface_1_displacement: Displacements
        A named tuple of the displacement on surface 1
    surface_2_displacement: Displacements
        A named tuple of the displacement on surface 2

    """
    surf_1 = model.surface_1
    surf_2 = model.surface_2
    if material_options is None:
        material_options = [dict(), dict()]
    else:
        material_options = [mo or dict() for mo in material_options]

    surface_1_displacement = surf_1.material.displacement_from_surface_loads(loads=loads,
                                                                             grid_spacing=surf_1.grid_spacing,
                                                                             deflections=deflections,
                                                                             current_state=current_state,
                                                                             **material_options[0])

    if reverse_loads_on_second_surface:
        loads_2 = Loads(*[-1 * loads.__getattribute__(l) if l in reverse_loads_on_second_surface
                          else loads.__getattribute__(l) for l in 'xyz'])  # noqa: E741
    else:
        loads_2 = loads

    surface_2_displacement = surf_2.material.displacement_from_surface_loads(loads=loads_2,
                                                                             grid_spacing=surf_1.grid_spacing,
                                                                             deflections=deflections,
                                                                             current_state=current_state,
                                                                             **material_options[1])
    total_displacement = Displacements(*(s1 + s2 for s1, s2 in zip(surface_1_displacement, surface_2_displacement)))

    return total_displacement, surface_1_displacement, surface_2_displacement


# noinspection PyArgumentList
def solve_normal_interference(interference: float, gap: np.ndarray, model: _ContactModelABC, current_state: dict,
                              adhesive_pressure: typing.Union[float, typing.Callable] = None,
                              contact_nodes: np.ndarray = None, max_iter: int = 100, tol: float = 1e-4,
                              initial_guess_loads: np.ndarray = None,
                              material_options: dict = None, remove_percent: float = 0.5,
                              node_thresh_percent: int = 0.01):
    """Solves contact with set normal interference

    Parameters
    ----------
    interference: float
        The interference between the surfaces measured from the point of first contact
    gap: np.ndarray
        The undeformed gap between the surfaces at the moment of first contact
    model: _ContactModelABC
        A contact model object containing the surfaces
    current_state: dict
        The state dict for the model before this step is solved
    adhesive_pressure: {float, Callable}, optional (None)
        The maximum adhesive force between the two surfaces, or a callable which wll be called as following:
        adhesive_force(surface_loads, deformed_gap, contact_nodes, model) and must return two boolean arrays containing
        the nodes to be removed and the nodes to be added in the iteration.
    contact_nodes: np.ndarray
        Boolean array of the surface nodes in contact at the start of the calculation
    material_options: dict
        Dict of options to be passed to the loads_from_surface_displacement method of the first surface
    max_iter: int
        The maximum number of iterations to find a stable set of contact nodes
    tol: float
        The tolerance on the solution
    initial_guess_loads: np.ndarray
        The initial guess for the loads, used in the optimisation step, must be the same shape as the gap array
        if this is not supplied the materials are used to generate an initial guess, this is often less accurate than
        using the previous time step, especially when the time step is short
    remove_percent: float
        The percentage of the current contact nodes which can be removed in a single iteration
    node_thresh_percent: float
        Percentage of contact nodes which can need to be added before the solution is converged

    Returns
    -------
    loads: Loads
        A named tuple of surface loads
    total_displacement: Displacements
        A named tuple of the total displacement
    surface_1_displacement: Displacements
        A named tuple of the displacement on surface 1
    surface_2_displacement: Displacements
        A named tuple of the displacement on surface 2
    contact_nodes: np.ndarray
        A boolean array of nodes in contact

    Notes
    -----

    """

    surf_1 = model.surface_1
    surf_2 = model.surface_2

    material_options = material_options or dict()

    if contact_nodes is None and adhesive_pressure is not None:
        warnings.warn('Contact nodes not set from previous step results may show unphysical adhesion force, use a '
                      'no adhesion step to initialise the contact nodes to avoid this behaviour')

    if adhesive_pressure is None:
        adhesive_pressure = 0

    z = interference - gap  # necessary displacement for completely touching, positive is into surfaces

    if contact_nodes is None:
        contact_nodes = z > 0

    if not any(contact_nodes.flatten()):
        zeros = np.zeros_like(z)
        return Loads(z=zeros), Displacements(z=zeros), Displacements(z=zeros), Displacements(z=zeros), contact_nodes

    if isinstance(surf_1.material, _IMMaterial) and isinstance(surf_2.material, _IMMaterial):
        if 'span' in material_options:
            span = material_options['span']
        else:
            span = tuple(gs * 2 for gs in gap.shape)

        im1 = surf_1.material.influence_matrix(span=span, grid_spacing=[surf_1.grid_spacing] * 2, components=['zz'])[
              'zz']
        im2 = surf_2.material.influence_matrix(span=span, grid_spacing=[surf_1.grid_spacing] * 2, components=['zz'])[
              'zz']
        total_im = im1 + im2

        convolution_func = plan_convolve(gap, total_im, contact_nodes)

        if initial_guess_loads is None:
            initial_guess_loads = guess_loads_from_displacement(Displacements(z=z), {'zz': total_im})

        max_pressure = min(surf_1.material.max_load, surf_2.material.max_load)

        loads_in_domain = bccg(convolution_func, z[contact_nodes], tol, max_iter, initial_guess_loads.z[contact_nodes],
                               min_pressure=adhesive_pressure, max_pressure=max_pressure)
        loads_full = np.zeros_like(z)
        loads_full[contact_nodes] = loads_in_domain
        loads = Loads(z=loads_full)

        total_disp_in_domain = convolution_func(loads_in_domain)
        if slippy.CUDA:
            import cupy as cp
            total_disp_in_domain = cp.asnumpy(total_disp_in_domain)
        total_disp_full = np.zeros_like(z)
        total_disp_full[contact_nodes] = total_disp_in_domain

        total_disp = Displacements(z=total_disp_full)
        disp_1 = Displacements(z=fftconvolve(loads_full, im1, 'same'))
        disp_2 = Displacements(z=fftconvolve(loads_full, im2, 'same'))

        contact_nodes = np.logical_and(loads_full > adhesive_pressure, loads_full != 0)

        return loads, total_disp, disp_1, disp_2, contact_nodes

    displacements = Displacements(z=z.copy(), x=None, y=None)
    displacements.z[np.logical_not(contact_nodes)] = np.nan

    it_num = 0
    added_nodes_last_it = np.inf

    while True:

        loads, disp_tup = surf_1.material.loads_from_surface_displacement(displacements=displacements,
                                                                          grid_spacing=surf_1.grid_spacing,
                                                                          other=surf_2.material,
                                                                          current_state=current_state,
                                                                          **material_options)

        # find deformed nd_gap and add contacting nodes to the contact nodes
        deformed_gap = gap - interference + disp_tup[0].z  # the nd_gap minus the interference plus the displacement

        force_another_iteration = False
        n_contact_nodes = sum(contact_nodes.flatten())

        print('Total contact nodes:', n_contact_nodes)

        if isinstance(adhesive_pressure, Number):
            nodes_to_remove = np.logical_and(loads.z < adhesive_pressure, contact_nodes)
            nodes_to_add = np.logical_and(deformed_gap < 0, np.logical_not(contact_nodes))
            print('Nodes to add: ', sum(nodes_to_add.flatten()))
            # noinspection PyUnresolvedReferences
            print('Nodes to remove raw: ', sum(nodes_to_remove.flatten()))

            max_remove = int(min(n_contact_nodes * remove_percent, 0.5 * added_nodes_last_it))
            # noinspection PyUnresolvedReferences
            if sum(nodes_to_remove.flatten()) > max_remove:
                nodes_to_remove = np.argpartition(-loads.z.flatten(), -max_remove)[-max_remove:]
                nodes_to_remove = np.unravel_index(nodes_to_remove, contact_nodes.shape)
                print('Nodes to remove treated: ', len(nodes_to_remove[0]))
                print('Forcing another iteration')
                force_another_iteration = True
        else:
            nodes_to_remove, nodes_to_add, force_another_iteration = adhesive_pressure(loads, deformed_gap,
                                                                                       contact_nodes, model)

        node_thresh = n_contact_nodes * node_thresh_percent

        if force_another_iteration or any(nodes_to_remove.flatten()) or sum(nodes_to_add.flatten()) > node_thresh:
            contact_nodes[nodes_to_add] = True
            contact_nodes[nodes_to_remove] = False
            n_nodes_added = sum(nodes_to_add.flatten())
            added_nodes_last_it = n_nodes_added if n_nodes_added else added_nodes_last_it  # if any nodes then update
            displacements = Displacements(z=z.copy(), x=None, y=None)
            displacements.z[np.logical_not(contact_nodes)] = np.nan

        else:
            break

        it_num += 1

        if it_num > max_iter:
            warnings.warn('Solution failed to converge on a set of contact nodes while solving for normal interference')
            loads.z[:] = np.nan
            break

    return (loads,) + disp_tup + (contact_nodes,)


def get_next_file_num(output_folder):
    highest_num = 0
    for f in os.listdir(output_folder):
        if os.path.isfile(os.path.join(output_folder, f)):
            file_name = os.path.splitext(f)[0]
            try:
                file_num = int(file_name)
                if file_num > highest_num:
                    highest_num = file_num
            except ValueError:
                pass

    output_file_num = str(highest_num + 1)
    return output_file_num
