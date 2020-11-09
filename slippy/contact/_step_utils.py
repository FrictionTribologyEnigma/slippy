"""
Helper functions for steps
"""
import os
import typing
import warnings
import bisect
from collections import namedtuple
from numbers import Number
from scipy.interpolate import interp1d

import numpy as np
from scipy.signal import fftconvolve

import slippy
if slippy.CUDA:
    import cupy as cp
else:
    cp = None

from slippy.abcs import _ContactModelABC  # noqa: E402
from ._material_utils import Loads, Displacements  # noqa: E402
from .influence_matrix_utils import bccg, plan_convolve, guess_loads_from_displacement  # noqa: E402
from .materials import _IMMaterial  # noqa: E402

__all__ = ['solve_normal_interference', 'get_next_file_num', 'OffSetOptions', 'solve_normal_loading',
           'HeightOptimisationFunction', 'make_interpolation_func']

OffSetOptions = namedtuple('off_set_options', ['off_set', 'abs_off_set', 'periodic', 'interpolation_mode'])


def make_interpolation_func(values, kind, name: str):
    """

    Parameters
    ----------
    values: sequence of floats
        Either [start, finish] or [position, time] where position and time are equal length sequences of floats.
    kind: any kind compatible with scipy.interpolate.interp1d
    name: str
        The name of the parameter being interpolated used for errors

    Returns
    -------
    interpolation_function: callable
    """
    try:
        values = np.asarray(values, dtype=float)
        assert not np.any(np.isnan(values))
    except ValueError:
        raise ValueError(f"Could not convert values for {name} to an valid format")
    except AssertionError:
        raise ValueError(f"Could not convert values for {name} to an valid format")

    if values.size == 2:
        position = values
        time = np.array([0, 1])
    elif values.shape[0] == 2:
        position = values[0]
        time = values[1]
    else:
        raise ValueError(f"Values for {name} are an invalid shape, should be 2 values (start, finish) or two equally "
                         f"sized sequences of values (position, time: shape 2 by n). Input shape was {values.shape}")
    return interp1d(time, position, kind, bounds_error=True)


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
    _contact_nodes = None
    _last_loads = None
    _results: typing.Optional[dict] = None
    set_contact_nodes = False

    def __init__(self, just_touching_gap: np.ndarray, model: _ContactModelABC,
                 adhesion_model: float, initial_contact_nodes: np.ndarray,
                 max_it_inner: int, tol_inner: float, material_options: typing.Union[typing.Sequence[dict], dict],
                 max_set_load: float, tolerance: float, use_cache: bool = True, cache_loads=True,
                 periodic_axes: typing.Tuple[bool] = (False, False)):
        if slippy.CUDA:
            xp = cp
            cache_loads = False
        else:
            xp = np

        self._grid_spacing = model.surface_1.grid_spacing

        self._just_touching_gap = xp.asarray(just_touching_gap)

        self._model = model
        self._adhesion_model = adhesion_model
        self.initial_contact_nodes = initial_contact_nodes
        self.contact_nodes = initial_contact_nodes
        self._max_it_inner = max_it_inner
        self._tol_inner = tol_inner
        self._material_options = material_options
        self._original_set_load = max_set_load
        self._set_load = float(max_set_load)
        self._periodic_axes = periodic_axes
        # these should really be a sorted dict but it seems like the added dependencies are not worth it atm
        self.cache_heights = [0.0]
        self.cache_total_load = [0.0]
        self.cache_surface_loads = [xp.zeros(just_touching_gap.shape)]
        self.tolerance = tolerance
        self.it = 0
        self.use_cache = use_cache
        self.use_loads_cache = cache_loads
        surf_1 = model.surface_1
        surf_2 = model.surface_2

        self.im_mats = False
        self.conv_func = None
        self.cache_max = False
        self.last_call_failed = False

        if isinstance(surf_1.material, _IMMaterial) and isinstance(surf_2.material, _IMMaterial):
            self.im_mats = True
            span = just_touching_gap.shape
            max_pressure = min([surf_1.material.max_load, surf_2.material.max_load])
            self._max_pressure = max_pressure
            im1 = surf_1.material.influence_matrix(span=span, grid_spacing=[surf_1.grid_spacing] * 2,
                                                   components=['zz'])['zz']
            im2 = surf_2.material.influence_matrix(span=span, grid_spacing=[surf_1.grid_spacing] * 2,
                                                   components=['zz'])['zz']
            total_im = im1 + im2
            self.total_im = xp.asarray(total_im)

            self.conv_func = plan_convolve(just_touching_gap, self.total_im, self.contact_nodes, circular=periodic_axes)

            if use_cache and max_pressure != np.inf:
                max_loads = max_pressure * xp.ones(just_touching_gap.shape)
                self.cache_surface_loads.append(max_loads)
                self.cache_total_load.append(max_pressure * just_touching_gap.size * surf_1.grid_spacing ** 2)
                if cache_loads:
                    max_elastic_disp = fftconvolve(max_loads, total_im, 'same')
                    self.cache_heights.append(np.max(max_elastic_disp + just_touching_gap))

    @property
    def contact_nodes(self):
        return self._contact_nodes

    @contact_nodes.setter
    def contact_nodes(self, value):
        if slippy.CUDA:
            xp = cp
        else:
            xp = np
        if value is None:
            self._contact_nodes = None
        else:
            value = xp.array(value, dtype=bool)
            self._contact_nodes = value
            if self.im_mats:
                self.conv_func = plan_convolve(self._just_touching_gap, self.total_im, self._contact_nodes,
                                               circular=self._periodic_axes)

    @property
    def results(self):
        if self._results is None:
            print("No results found in height opt func")
        if slippy.CUDA:
            xp = cp
        else:
            xp = np
        if self.im_mats:
            # need to put the loads into an np array of right shape
            # find the full displacements (and convert to np array)
            # find disp on surface 1 and surface 2
            surf_1 = self._model.surface_1
            surf_2 = self._model.surface_2
            span = self._just_touching_gap.shape
            # noinspection PyUnresolvedReferences
            im1 = surf_1.material.influence_matrix(span=span, grid_spacing=[surf_1.grid_spacing] * 2,
                                                   components=['zz'])['zz']
            # noinspection PyUnresolvedReferences
            im2 = surf_2.material.influence_matrix(span=span, grid_spacing=[surf_1.grid_spacing] * 2,
                                                   components=['zz'])['zz']
            full_loads = np.zeros(self._just_touching_gap.shape)
            if slippy.CUDA:
                full_loads[xp.asnumpy(self._results['domain'])] = xp.asnumpy(self._results['loads_in_domain'])
                full_disp = xp.asnumpy(self.conv_func(self._results['loads_in_domain'], ignore_domain=True))
            else:
                full_loads[self._results['domain']] = self._results['loads_in_domain']
                full_disp = self.conv_func(self._results['loads_in_domain'], ignore_domain=True)

            conv_func_1 = plan_convolve(full_loads, im1, None, circular=self._periodic_axes)
            conv_func_2 = plan_convolve(full_loads, im2, None, circular=self._periodic_axes)

            disp_1 = Displacements(z=conv_func_1(full_loads))
            disp_2 = Displacements(z=conv_func_2(full_loads))

            total_load = float(xp.sum(self._results['loads_in_domain']) * self._grid_spacing ** 2)

            results = {'loads': Loads(z=full_loads), 'total_displacement': Displacements(z=full_disp),
                       'surface_1_displacement': disp_1, 'surface_2_displacement': disp_2,
                       'contact_nodes': full_loads > 0, 'total_normal_load': total_load,
                       'interference': self._results['interference']}
            return results
        else:
            return self._results

    def clear_cache(self):
        if self.cache_max:
            self.cache_heights = [self.cache_heights[0], self.cache_heights[-1]]
            self.cache_total_load = [self.cache_total_load[0], self.cache_total_load[-1]]
            self.cache_surface_loads = [self.cache_surface_loads[0], self.cache_surface_loads[-1]]
        else:
            self.cache_heights = []
            self.cache_total_load = []
            self.cache_surface_loads = []

    def change_load(self, new_load, contact_nodes):
        # if you change the load... need to change the set load,
        self.contact_nodes = contact_nodes
        if contact_nodes is None:
            self.set_contact_nodes = False
        else:
            self.set_contact_nodes = True
        self._set_load = float(new_load)
        self._last_loads = None
        self._results = None
        self.it = 0

    def get_bounds_from_cache(self, lower, upper):
        # want to find the closest HEIGHT above and below the set load, if none above or below return None for that one
        if len(self.cache_heights) < 2:
            return lower, upper
        index = bisect.bisect_left(self.cache_total_load, self._set_load)
        try:
            upper_est = self.cache_heights[index]
            upper_bound = min(upper_est, upper)
        except IndexError:
            upper_bound = upper

        if index > 0:
            lower_est = self.cache_heights[index-1]
            lower_bound = max(lower_est, lower)
        else:
            lower_bound = lower

        return lower_bound, upper_bound

    def __call__(self, height, current_state):
        if slippy.CUDA:
            xp = cp
        else:
            xp = np
        # If the height guess is in the cache that can be out load guess
        pressure_initial_guess = None  # overwritten in one case where it should be (cached)
        height = float(height)
        if height in self.cache_heights:
            total_load = self.cache_total_load[self.cache_heights.index(height)]
            print(f"Returning bound value from cache: height: {height:.4}, total_load {total_load:.4}")
            return total_load - self._set_load
        self.it += 1
        # if im mats we can save some time here mostly by not moving data to and from the gpu
        if self.im_mats:
            z = - self._just_touching_gap + height
            if not self.set_contact_nodes:
                self.contact_nodes = z > 0  # this will remake the conv function as a side effect
            contact_nodes = self.contact_nodes
            if not xp.any(contact_nodes):
                print('no contact nodes')
                total_load = 0
                full_loads = xp.zeros(contact_nodes.shape, dtype=xp.float32)
                failed = False
            else:
                if pressure_initial_guess is None:
                    pressure_initial_guess = xp.zeros_like(z)
                else:
                    pressure_initial_guess = pressure_initial_guess.z
                z_in = z[contact_nodes]
                pressure_guess_in = pressure_initial_guess[contact_nodes]
                loads_in_domain, failed = bccg(self.conv_func, z_in, self._tol_inner,
                                               self._max_it_inner, pressure_guess_in,
                                               self._adhesion_model, self._max_pressure)
                self._results = {'loads_in_domain': loads_in_domain, 'domain': self.contact_nodes,
                                 'interference': height}
                total_load = float(xp.sum(loads_in_domain) * self._grid_spacing ** 2)
                if self.use_loads_cache:
                    full_loads = xp.zeros(contact_nodes.shape)
                    full_loads[contact_nodes] = loads_in_domain
                else:
                    full_loads = None

            self.add_to_cache(height, total_load, full_loads, failed)
            if failed:
                # noinspection PyUnboundLocalVariable
                print(f'Failed: total load: {total_load}, height {height}, max_load {xp.max(loads_in_domain)}')
                self.last_call_failed = True
            else:
                print(f'Solved: interference: {height}\tTotal load: {total_load}\tTarget load: {self._set_load}')
                self.last_call_failed = False
            return total_load - self._set_load

        # else use the basic form

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

        self._results['loads'] = loads
        self._results['total_displacement'] = total_disp
        self._results['surface_1_displacement'] = disp_1
        self._results['surface_2_displacement'] = disp_2
        self._results['contact_nodes'] = contact_nodes
        self._results['interference'] = height

        total_load = np.sum(loads.z.flatten()) * self._grid_spacing ** 2

        self._results['total_normal_load'] = total_load

        if failed:
            self.last_call_failed = True
            print(f'Failed: total load: {total_load}, height {height}, max_load {np.max(loads.z.flatten())}')
        else:
            self.last_call_failed = False
            print(f'Interference is: {height}\tTotal load is: {total_load}\tTarget load is: {self._set_load}')

        self.add_to_cache(height, total_load, loads.z, failed)

        return total_load - self._set_load

    def add_to_cache(self, height, total_load, loads, failed):
        if self.use_cache and height not in self.cache_heights and not failed:
            print(f"Inserting height: {height}, total_load: {total_load} into cache, len = {1+len(self.cache_heights)}")
            index = bisect.bisect_left(self.cache_heights, height)
            self.cache_heights.insert(index, height)
            self.cache_total_load.insert(index, total_load)
            # print(f'Cache: adding value: total load: {total_load},
            # height {height}, cache len:{len(self.cache_heights)}')
            if self.use_loads_cache:
                self.cache_surface_loads.insert(index, loads)


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

    if not np.any(contact_nodes.flatten()):
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
        if slippy.CUDA:
            import cupy as cp
            loads_in_domain = cp.asnumpy(loads_in_domain)
        loads_full = np.zeros_like(z)
        loads_full[contact_nodes] = loads_in_domain
        loads = Loads(z=loads_full)

        total_disp = convolution_func(loads_in_domain, ignore_domain=True)
        if slippy.CUDA:
            total_disp = cp.asnumpy(total_disp)

        total_disp = Displacements(z=total_disp)
        disp_1 = Displacements(z=fftconvolve(loads_full, im1, 'same'))
        disp_2 = Displacements(z=fftconvolve(loads_full, im2, 'same'))

        contact_nodes = np.logical_and(loads_full > adhesive_pressure, loads_full != 0)

        return loads, total_disp, disp_1, disp_2, contact_nodes, failed
    # if not both influence matrix based materials
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
