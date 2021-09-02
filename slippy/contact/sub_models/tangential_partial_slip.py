import typing
from scipy.interpolate import interp1d
import numpy as np
import slippy
from slippy.core import _SubModelABC
from slippy.core.materials import _IMMaterial
from slippy.core.influence_matrix_utils import bccg, plan_convolve
# TODO add from_offset option to get the displacement from the offset


class TangentialPartialSlip(_SubModelABC):
    """ Solves the partial slip problem

    Parameters
    ----------
    name: str
        The name of the sub model, used for debugging
    direction: {'x', 'y'}
        The direction of applied load or displacement, only 'x' and  'y' are currently supported
    load, displacement: float or sequence of floats
        Up to one can be supplied, either the total load or the rigid body displacement. Suitable values are:
            - float: indicating a constant load/ displacement
            - 2 by n array: of time points and load/ displacement values
        If an array is supplied and it is too short it is extrapolated by repeating the final value, this produces a
        warning. If neither are supplied this sub-model requires rigid_body_displacement to be provided by a further
        sub-model
    periodic_axes: 2 element sequence of bool, optional (False, False)
        True for each axis which the solution should be periodic in, should match solving step
    tol: float, optional (1e-7)
        The tolerance used to declare convergence for the bccg iterations
    max_it: int, optional (None)
        The maximum number of iterations for the bccg iterations, defaults to the same as the number of contact nodes
    """

    def __init__(self, name: str, direction: str,
                 load: typing.Union[float, typing.Sequence] = None,
                 displacement: typing.Union[float, typing.Sequence] = None,
                 periodic_axes: typing.Sequence[bool] = (False, False),
                 tol: float = 1e-7, max_it: int = None):

        requires = {'maximum_tangential_force', 'contact_nodes', 'time'}

        if load is None and displacement is None:
            self.displacement_from_sub_model = True
            requires.add('rigid_body_displacement_' + direction)
            self.update_displacement = False
        else:
            self.displacement_from_sub_model = False
        provides = {'slip_distance', 'stick_nodes', 'loads_x', 'loads_y', 'total_displacement_x',
                    'total_displacement_y'}
        super().__init__(name, requires, provides)

        self.load_controlled = False

        if load is not None:
            if displacement is not None:
                raise ValueError("Either the load or the displacement can be set, not both")
            try:
                self.load = float(load)
                self.update_load = False
                self.load_upd = None
            except TypeError:
                self.load = None
                self.load_upd = interp1d(load[0, :], load[1, :], fill_value='extrapolate')
                self.update_load = True
            self.load_controlled = True

        if displacement is not None:
            try:
                self.displacement = float(displacement)
                self.update_displacement = False
                self.displacement_upd = None
            except TypeError:
                self.displacement = None
                self.displacement_upd = interp1d(displacement[0, :], displacement[1, :], fill_value='extrapolate')
                self.update_displacement = True

        self.component = direction * 2
        self._last_span = None
        self._pre_solve_checks = False
        self._im_1 = None
        self._im_2 = None
        self._im_total = None
        self._periodic_axes = periodic_axes
        self._tol = tol
        self._max_it = max_it
        self.previous_result = None

    def _check(self, span):
        # check that both are im materials and store ims
        if isinstance(self.model.surface_1.material, _IMMaterial) and \
           isinstance(self.model.surface_2.material, _IMMaterial):
            im_1 = self.model.surface_1.material.influence_matrix([self.component],
                                                                  [self.model.surface_1.grid_spacing] * 2,
                                                                  span)[self.component]
            im_2 = self.model.surface_2.material.influence_matrix([self.component],
                                                                  [self.model.surface_1.grid_spacing] * 2,
                                                                  span)[self.component]
            self._im_1 = im_1
            self._im_2 = im_2
            self._im_total = im_1 + im_2
            self._pre_solve_checks = True
        else:
            raise ValueError("This sub model only supports influence matrix based materials")

    def solve(self, current_state: dict) -> dict:
        span = current_state['maximum_tangential_force'].shape

        if not self._pre_solve_checks or span != self._last_span:
            self._check(span)
            self._last_span = span

        domain = current_state['contact_nodes']

        conv_func = plan_convolve(self._im_total, self._im_total, domain,
                                  circular=self._periodic_axes)
        # if the displacements are provided by another sub model or we have a set displacement we just have one set
        # of bccg iterations:
        if not self.load_controlled:
            if self.update_displacement:
                set_displacement = self.displacement_upd(current_state['time'])
            elif self.displacement_from_sub_model:
                set_displacement = current_state['rigid_body_displacement_' + self.component[0]]
            else:
                set_displacement = self.displacement
            try:
                set_displacement = float(set_displacement)*np.ones_like(current_state['maximum_tangential_force'])
            except TypeError:
                pass
            x0 = self.previous_result if self.previous_result is not None else \
                current_state['maximum_tangential_force']/2
            min_pressure = np.array(-1*current_state['maximum_tangential_force'][domain])
            loads_in_domain, failed = bccg(conv_func, set_displacement[domain], self._tol,
                                           self._max_it, x0[domain],
                                           min_pressure,
                                           current_state['maximum_tangential_force'][domain])
            loads_in_domain = slippy.asnumpy(loads_in_domain)
            full_loads = np.zeros_like(current_state['maximum_tangential_force'])
            full_loads[domain] = loads_in_domain
            stick_nodes = np.logical_and(domain, full_loads < (0.99 * current_state['maximum_tangential_force']))
            current_state['stick_nodes'] = stick_nodes
            tangential_deformation = slippy.asnumpy(conv_func(loads_in_domain, True))
            current_state['loads_' + self.component[0]] = full_loads

            if 'total_displacement_' + self.component[0] in current_state:
                current_state['total_displacement_' + self.component[0]] += tangential_deformation
            else:
                current_state['total_displacement_' + self.component[0]] = tangential_deformation

            slip_distance = set_displacement-tangential_deformation
            slip_distance[stick_nodes] = 0
            slip_distance[np.logical_not(domain)] = 0
            current_state['slip_distance'] = slip_distance
            return current_state
        else:
            raise NotImplementedError('Load controlled partial slip is not yet implemented')
