"""
Helper functions for steps
"""
import numpy as np
from slippy.abcs import _ContactModelABC
import typing
from slippy.contact import Displacements
from numbers import Number
import warnings
import os

__all__ = ['solve_normal_interference', 'get_next_file_num']


def solve_normal_interference(interference: float, gap: np.ndarray, model: _ContactModelABC,
                              adhesive_force: typing.Union[float, typing.Callable] = None,
                              contact_nodes: np.ndarray = None, max_iter: int = 50,
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
    adhesive_force: {float, Callable}, optional (None)
        The maximum adhesive force between the two surfaces, or a callable which wll be called as following:
        adhesive_force(surface_loads, deformed_gap, contact_nodes, model) and must return two boolean arrays containing
        the nodes to be removed and the nodes to be added in the iteration.
    contact_nodes: np.ndarray
        Boolean array of the surface nodes in contact at the start of the calculation
    material_options: dict
        Dict of options to be passed to the loads_from_surface_displacement method of the first surface
    max_iter: int
        The maximum number of iterations to find a stable set of contact nodes
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
    if adhesive_force is None:
        adhesive_force = 0

    # z = -1 * np.clip(gap - interference, None, 0)
    # z[z == 0] = np.nan
    z = interference - gap  # necessary displacement for completely touching, positive is into surfaces

    if contact_nodes is None:
        contact_nodes = z > 0
    elif not any(contact_nodes.flatten()) and adhesive_force:
        contact_nodes = z > 0
        warnings.warn('Contact nodes not set from previous step results may show unphysical adhesion force, use a '
                      'no adhesion step to initialise the contact nodes to avoid this behaviour')

    displacements = Displacements(z=z.copy(), x=None, y=None)
    displacements.z[np.logical_not(contact_nodes)] = np.nan

    it_num = 0
    added_nodes_last_it = np.inf

    while True:

        loads, disp_tup = surf_1.material.loads_from_surface_displacement(displacements=displacements,
                                                                          grid_spacing=surf_1.grid_spacing,
                                                                          other=surf_2.material,
                                                                          **material_options)

        # find deformed gap and add contacting nodes to the contact nodes
        deformed_gap = gap - interference + disp_tup[0].z  # the gap minus the interference plus the displacement

        force_another_iteration = False
        n_contact_nodes = sum(contact_nodes.flatten())

        print('Total contact nodes:', sum(contact_nodes.flatten()))

        if isinstance(adhesive_force, Number):
            nodes_to_remove = loads.z < adhesive_force
            nodes_to_add = np.logical_and(deformed_gap < 0, np.logical_not(contact_nodes))
            print('Nodes to add: ', sum(nodes_to_add.flatten()))
            print('Nodes to remove raw: ', sum(nodes_to_remove.flatten()))

            max_remove = int(min(n_contact_nodes * remove_percent, 0.5*added_nodes_last_it))
            # noinspection PyUnresolvedReferences
            if sum(nodes_to_remove.flatten()) > max_remove:
                if not max_remove:
                    raise ValueError()
                nodes_to_remove = np.argpartition(-loads.z.flatten(), -max_remove)[-max_remove:]
                nodes_to_remove = np.unravel_index(nodes_to_remove, contact_nodes.shape)
                print('Nodes to remove treated: ', len(nodes_to_remove[0]))
                print('Forcing another iteration')
                force_another_iteration = True
        else:
            nodes_to_remove, nodes_to_add, force_another_iteration = adhesive_force(loads, deformed_gap,
                                                                                    contact_nodes, model)

        node_thresh = n_contact_nodes*node_thresh_percent

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
