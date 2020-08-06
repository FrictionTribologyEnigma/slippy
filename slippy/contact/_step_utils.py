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
from scipy.interpolate import interp1d

import slippy
from slippy.abcs import _ContactModelABC
from ._material_utils import Loads, Displacements
from ._model_utils import non_dimentional_height
from .influence_matrix_utils import bccg, plan_convolve, guess_loads_from_displacement
from .materials import _IMMaterial

__all__ = ['solve_normal_interference', 'get_next_file_num', 'OffSetOptions', 'solve_normal_loading',
           'HeightOptimisationFunction']

OffSetOptions = namedtuple('off_set_options', ['off_set', 'abs_off_set', 'periodic', 'interpolation_mode'])


class HeightOptimisationFunction:
    """ A class to make a memorised function to be used in a height optimisation loop
    Ths doesn't do the optimising it just gives a a callable class that can be used in one of scipy's methods

    Parameters
    ----------
    just_touching_gap: np.ndarray
        The just touching gap between the surfaces, should be found by get_gap_from function
    model: _ContactModelABC
        The contact model this will be solved in
    adhesion_model: float
        The maximum adhesive pressure between the surfaces
    initial_contact_nodes:
        If set the solution will be constrained to these nodes, if this is not wanted use None
    max_it_inner: int
        The maximum number of iterations for the inner loops
    tol_inner: float
        The tolerance used to declare convergence for the inner loops
    material_options: dict, list of dict
        Material options for the materials in the model
    max_set_load: float
        The target total load in the normal direction, this must be kept up to date if the same function is being reused
    tolerance: float
        The tolerance used in the outer loop, the loop that this functions is optimised over, this is necessary to
        ensure a full solution is always returned (the final solution will not be returned from the cache)
    use_cache: bool, optional (True)
        If False the cache won't be used
    cache_loads: bool, optional (True)
        If False the full loads result will not be cached (otherwise this will be used to generate an initial guess of
        the loads for each iteration)
    """
    def __init__(self, just_touching_gap: np.ndarray, model: _ContactModelABC,
                 adhesion_model: float, initial_contact_nodes: np.ndarray,
                 max_it_inner: int, tol_inner: float, material_options: typing.Union[typing.Sequence[dict], dict],
                 max_set_load: float, tolerance: float, use_cache: bool = True, cache_loads=True):
        self._grid_spacing = model.surface_1.grid_spacing
        self._just_touching_gap = just_touching_gap
        self._model = model
        self._adhesion_model = adhesion_model
        self.initial_contact_nodes = initial_contact_nodes
        self.contact_nodes = initial_contact_nodes
        self._max_it_inner = max_it_inner
        self._tol_inner = tol_inner
        self._material_options = material_options
        self._original_set_load = max_set_load
        self._set_load = max_set_load
        self.results = dict()
        # these should really be a sorted dict but it seems like the added dependencies are not worth it atm
        self.cache_heights = [0]
        self.cache_total_load = [0]
        self.cache_surface_loads = [np.zeros_like(just_touching_gap)]
        self.tolerance = tolerance
        self.it = 0
        self.use_cache = use_cache
        self.use_loads_cache = cache_loads
        surf_1 = model.surface_1
        surf_2 = model.surface_2

        self.cache_max = False

        try:
            if max_set_load:
                self.uz = non_dimentional_height(1, surf_1.material.E, surf_1.material.v, max_set_load,
                                                 surf_1.grid_spacing, return_uz=True)
        except AttributeError:
            self.uz = 1
        if isinstance(surf_1.material, _IMMaterial) and isinstance(surf_2.material, _IMMaterial):
            max_load = min([surf_1.material.max_load, surf_2.material.max_load])

            if use_cache and max_load != np.inf:
                max_loads = max_load * np.ones(just_touching_gap.shape)
                self.cache_surface_loads.append(max_loads)
                self.cache_total_load.append(max_load*just_touching_gap.size*surf_1.grid_spacing**2)
                span = [2*s for s in just_touching_gap.shape]
                im1 = surf_1.material.influence_matrix(span=span, grid_spacing=[surf_1.grid_spacing] * 2,
                                                       components=['zz'])['zz']
                im2 = surf_2.material.influence_matrix(span=span, grid_spacing=[surf_1.grid_spacing] * 2,
                                                       components=['zz'])['zz']
                total_im = im1 + im2
                max_elastic_disp = fftconvolve(max_loads, total_im, 'same')
                self.cache_heights.append(np.max(max_elastic_disp+just_touching_gap))

    def clear_cache(self):
        if self.cache_max:
            self.cache_heights = [self.cache_heights[0:2]]
            self.cache_total_load = [self.cache_total_load[0:2]]
            self.cache_surface_loads = [self.cache_surface_loads[0:2]]
        else:
            self.cache_heights = []
            self.cache_total_load = []
            self.cache_surface_loads = []

    def change_load(self, new_load, contact_nodes):
        # if you change the load... need to change the set load,
        self.contact_nodes = contact_nodes
        self._set_load = new_load
        self.results = dict()
        surf_1 = self._model.surface_1
        try:
            self.uz = non_dimentional_height(1, surf_1.material.E, surf_1.material.v, self._set_load,
                                             surf_1.grid_spacing, return_uz=True)
        except AttributeError:
            self.uz = 1
        self.it = 0

    def get_bounds_from_cache(self, lower, upper):
        # want to find the closest HEIGHT above and below the set load, if none above or below return None for that one
        if len(self.cache_heights) < 2:
            return lower, upper

        lower *= self.uz
        upper *= self.uz

        try:
            load_diff_above = [load - self._set_load if (load - self._set_load > 0) else float('inf')
                               for load in self.cache_total_load]
            index_above = min(range(len(load_diff_above)), key=load_diff_above.__getitem__)
            upper_est = self.cache_heights[index_above] if load_diff_above[index_above] != float('inf') else float('inf')
            upper_bound = min(upper_est, upper)
        except ValueError:
            upper_bound = upper

        try:
            load_diff_below = [self._set_load - load if (self._set_load - load) > 0 else float('inf')
                               for load in self.cache_total_load]
            index_below = min(range(len(load_diff_below)), key=load_diff_below.__getitem__)
            lower_est = self.cache_heights[index_below] if load_diff_below[index_below] != float('inf') \
                else -float('inf')
            lower_bound = max(lower_est, lower)
        except ValueError:
            lower_bound = lower
        return lower_bound/self.uz, upper_bound/self.uz

    def __call__(self, height, current_state):
        # If the height guess is in the cache that can be out load guess
        pressure_initial_guess = None  # overwritten in one case where it should be
        height = float(height)
        height *= self.uz  # make height dimentional
        if len(self.cache_heights) > 1:
            in_between = None
            try:
                total_load_guess = self.cache_total_load[self.cache_heights.index(height)]
                in_between = True
            except ValueError:
                # it wasn't so we should check that there is at least one point above and below our target load
                # interp1d will only work if this is right
                try:
                    # linear is safest here, get negative guesses with cubic
                    load_interpolator = interp1d(self.cache_heights, self.cache_total_load, 'linear')
                    total_load_guess = float(load_interpolator(height))
                except ValueError:
                    total_load_guess = None
            if total_load_guess is not None:
                if total_load_guess < 0:
                    raise ValueError()
                # now we should check that there is at least one point between our guess and the target:
                if in_between is None:
                    in_between = [load for load in self.cache_total_load if self._set_load > load > total_load_guess or
                                  self._set_load < load < total_load_guess]
                if in_between and (abs(total_load_guess - self._set_load) / self._set_load) > self.tolerance*5:
                    print(f"Cache: Returning load guess from cache, height: {height}, "
                          f"load guess: {total_load_guess}, set load {self._set_load}")
                    return total_load_guess - self._set_load
                elif self.use_loads_cache:
                    pressure_initial_guess = Loads(z=interp1d(self.cache_heights,
                                                              self.cache_surface_loads, axis=0)(height))

        # if it is not available in the cache then work it out properly:
        # make height dimensional
        self.it += 1

        loads, total_disp, disp_1, disp_2, contact_nodes, failed = \
            solve_normal_interference(height, gap=self._just_touching_gap,
                                      model=self._model,
                                      current_state=current_state,
                                      adhesive_pressure=self._adhesion_model,
                                      contact_nodes=self.contact_nodes,
                                      max_iter=self._max_it_inner,
                                      material_options=self._material_options,
                                      tol=self._tol_inner,
                                      initial_guess_loads=pressure_initial_guess)

        self.results['loads'] = loads
        self.results['total_displacement'] = total_disp
        self.results['surface_1_displacement'] = disp_1
        self.results['surface_2_displacement'] = disp_2
        self.results['contact_nodes'] = contact_nodes
        print(f'**********************************\nIteration {self.it}:')
        print(f'Interference is: {height}')
        print(f'Percentage of nodes in contact: {sum(contact_nodes.flatten()) / contact_nodes.size}')
        print(f'Target load is: {self._set_load}')

        total_load = np.sum(loads.z.flatten()) * self._grid_spacing ** 2
        self.results['total_normal_load'] = total_load

        if failed:
            print(f'Failed: total load: {total_load}, height {height}, max_load {np.max(loads.z.flatten())}')

        if self.use_cache and height not in self.cache_heights and not failed:
            print(f'Cache: adding value: total load: {total_load}, height {height}, max_load {np.max(loads.z.flatten())}')
            self.cache_total_load.append(total_load)
            self.cache_heights.append(height)
            if self.use_loads_cache:
                self.cache_surface_loads.append(loads.z)
        return total_load - self._set_load


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
        Boolean array of the surface nodes in contact at the start of the calculation, if set loading will be confined
        to these nodes
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
    failed: bool
        False if the solution converged

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
        print('no_contact_nodes')
        zeros = np.zeros_like(z)
        return (Loads(z=zeros), Displacements(z=zeros), Displacements(z=zeros), Displacements(z=zeros),
                contact_nodes, False)

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

        loads_in_domain, failed = bccg(convolution_func, z[contact_nodes], tol, max_iter,
                                       initial_guess_loads.z[contact_nodes],
                                       min_pressure=adhesive_pressure, max_pressure=max_pressure)
        loads_full = np.zeros_like(z)
        loads_full[contact_nodes] = loads_in_domain
        loads = Loads(z=loads_full)

        total_disp = convolution_func(loads_in_domain, ignore_domain=True)
        if slippy.CUDA:
            import cupy as cp
            total_disp = cp.asnumpy(total_disp)

        total_disp = Displacements(z=total_disp)
        disp_1 = Displacements(z=fftconvolve(loads_full, im1, 'same'))
        disp_2 = Displacements(z=fftconvolve(loads_full, im2, 'same'))

        contact_nodes = np.logical_and(loads_full > adhesive_pressure, loads_full != 0)

        return loads, total_disp, disp_1, disp_2, contact_nodes, failed

    displacements = Displacements(z=z.copy(), x=None, y=None)
    displacements.z[np.logical_not(contact_nodes)] = np.nan

    it_num = 0
    added_nodes_last_it = np.inf
    failed = False

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
            failed = True
            break

    return (loads,) + disp_tup + (contact_nodes, failed)


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
