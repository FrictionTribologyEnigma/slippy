import numpy as np
import typing
import slippy
from ._TransientSubModelABC import _TransientSubModelABC
from slippy.core import _IMMaterial, plan_convolve, gmres

__all__ = ['RigidBodyDisplacementSliding', 'RollingSliding1D']


class RigidBodyDisplacementSliding(_TransientSubModelABC):
    """Displacement from pure sliding

    """

    def __init__(self, distance_x: typing.Union[typing.Sequence[float], float] = 0,
                 distance_y: typing.Union[typing.Sequence[float], float] = 0, name: str = 'Sliding',
                 interpolation_mode: str = 'linear'):

        super().__init__(name, {'contact_nodes'}, {'rigid_body_displacement_x', 'rigid_body_displacement_y'},
                         [distance_x, distance_y],
                         ['distance_x', 'distance_y'], interpolation_mode)

    def _solve(self, current_state: dict, distance_x=None, distance_y=None, **kwargs) -> dict:
        if distance_x is None or distance_y is None:
            raise ValueError("Transient items not properly set")
        if kwargs:
            raise ValueError("Unexpected transient items")
        current_state['rigid_body_displacement_x'] = distance_x
        current_state['rigid_body_displacement_y'] = distance_y
        return current_state


class RollingSliding1D(_TransientSubModelABC):
    """Solve the one dimensional rolling sliding problem

    Parameters
    ----------
    creep_or_speed_difference: float, 2 element sequence of floats
        Either a float indicating constant 1D value, or a two element sequence of floats giving the value at the start
        and end of the model step, or a 2 by n array of n values and n times respectively. The interpretation of this
        parameter depends on the from_contact_time parameter. If the rigid body displacement is calculated from the
        contact time this parameter represents the speed difference, otherwise the rigid body displacement is calculated
        from the distance to the leading edge and this parameter represents the creep.
    from_contact_time: bool, optional (False)
        If true the creep will be calculated from the contact time for each element, in this case the
        creep_or_speed_difference parameter is interpreted as the difference in speed between the bodies, otherwise it
        is interpreted as the creep (difference in speed divided by the mean speed)
    direction: {'x', 'y'}, optional ('x')
        The axis along which the creep is applied
    name: str, optional ('RollingSliding1D')
        The name of the sub model used for logging and debugging
    interpolation_mode: str, optional (None)
        The kind of interpolation to use for interpolating the creep values, and mode compatible with
        scipy.interpolate.interp1d can be used
    periodic_axes: tuple, optional ((False, False))
        For each True value the corresponding axis will be solved by circular convolution, meaning the result is
        periodic in that direction
    tol: float, optional (1e-7)
        The tolerance used for convergence of the GMRES iterations
    max_inner_it: int, optional (None)
        The maximum number of iterations for the GMRES solver, defaults to the problem size
    restart: int, optional (20)
        The number of iterations between restarts of the GMRES solver, a higher number generally gives faster
        convergence but each iteration is more computationally costly
    max_outer_it: int, optional (100)
        The maximum number of iterations used for the outer loop (slip area determination)
    add_to_existing:bool = True
        If True the rigid body displacement will be added to the existing displacements in the current state, not
        needed for matched materials

    Notes
    -----
    This sub model will only run on the CPU

    References
    ----------

    Examples
    --------

    """

    def __init__(self, creep_or_speed_difference: typing.Union[typing.Sequence[float], float],
                 from_contact_time: bool = False,
                 direction: str = 'x',
                 name: str = 'RollingSliding1D',
                 interpolation_mode: str = 'linear',
                 periodic_axes: typing.Sequence[bool] = (False, False),
                 tol: float = 1e-7, max_inner_it: int = None, restart: int = 20, max_outer_it: int = 100,
                 add_to_existing: bool = True):
        if direction not in {'x', 'y'}:
            raise ValueError("Creep direction should be 'x' or 'y'")
        self.component = direction * 2
        self.axis = int(direction == 'x')  # 1 if x 0 if y
        self.from_contact_time = from_contact_time
        self._last_shape = None
        self._pre_solve_checks = False
        self._im_1 = None
        self._im_2 = None
        self._im_total = None
        self._periodic_axes = periodic_axes
        self._tol = tol
        self._max_it = max_inner_it
        self._restart = restart
        self._max_outer_it = max_outer_it
        self._base_conv_func = None
        self.previous_result = None
        self.previous_domain = -10
        self.multiplier = 1.0
        self._add_to_existing = add_to_existing
        requires = {'maximum_tangential_force', 'contact_nodes', 'time'}
        if from_contact_time:
            requires.add('contact_time_1')
            requires.add('contact_time_2')

        super().__init__(name, requires,
                         {f'rigid_body_displacement_{direction}',
                          f'loads_{direction}',
                          f'total_displacement_{direction}',
                          f'total_tangential_force_{direction}',
                          'slip_nodes', f'{name}_failed'},
                         [creep_or_speed_difference, ],
                         ['creep', ], interpolation_mode)

    def _check(self, shape):
        # check that both are im materials and store ims
        if not self._pre_solve_checks or shape != self._last_shape:
            if isinstance(self.model.surface_1.material, _IMMaterial) and \
               isinstance(self.model.surface_2.material, _IMMaterial):
                span = tuple([s * (2 - pa) for s, pa in zip(shape, self._periodic_axes)])
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
                self._base_conv_func = plan_convolve(np.zeros(shape), self._im_total,
                                                     None, self._periodic_axes)
                self._last_shape = shape
            else:
                raise ValueError("This sub model only supports influence matrix based materials")

    def _solve(self, current_state: dict, creep=None) -> dict:
        # get rigid body displacement (minus off set at first node)
        if creep is None:
            raise ValueError("Transient items not properly set")

        if slippy.CUDA:
            xp = slippy.xp
        else:
            xp = np

        domain = xp.array(current_state['contact_nodes'], dtype=bool)

        if self.from_contact_time:
            rbd = (current_state['contact_time_1']+current_state['contact_time_2']) * creep/2
            first = xp.argmax(xp.logical_and(xp.array(rbd) == 0, domain))
            if self.axis:
                first_index = xp.arange(domain.shape[not self.axis], dtype=int), first
            else:
                first_index = first, xp.arange(domain.shape[not self.axis], dtype=int)
        else:
            x_1 = xp.asarray(self.model.surface_1.get_points_from_extent()[self.axis])
            first = xp.argmax(domain, axis=self.axis)
            if self.axis:
                first_index = xp.arange(domain.shape[not self.axis], dtype=int), first
            else:
                first_index = first, xp.arange(domain.shape[not self.axis], dtype=int)
            off_sets = xp.expand_dims(x_1[first_index], self.axis)
            x_1 -= off_sets
            rbd = x_1 * creep * domain
        rbd = xp.array(rbd)
        # checks for im materials makes self._base_conv_function if needed
        self._check(domain.shape)

        # prepare limits etc
        limits_full = xp.array(current_state['maximum_tangential_force'])
        limits_full[first_index] = xp.inf
        max_it = self._max_it or xp.sum(domain)
        max_loads = xp.array(current_state['maximum_tangential_force'])
        max_loads[first_index] = 0
        disp_guess = self._base_conv_func(max_loads)
        disp_offsets = xp.expand_dims(disp_guess[first_index], self.axis)
        max_def = disp_guess - disp_offsets
        self.multiplier = xp.mean(disp_guess[first_index]) / np.mean(max_loads)

        try:
            if self.previous_result is None or xp.any(domain ^ self.previous_domain):
                initial_guess = max_loads[domain]
            else:
                initial_guess = self.previous_result[domain]
        except ValueError:
            initial_guess = max_loads[domain]

        # noinspection PyTypeChecker
        if xp.all(max_def < rbd):
            # full slip
            print('full slip')
            sub_loads, failed = max_loads[domain], False
            full_loads = max_loads
            full_disp = disp_guess
            disp_offsets = max_def
            slip_nodes = domain
        else:
            conv_func = ConvFuncWrapper(self._base_conv_func, limits_full, domain,
                                        self.axis, self.multiplier, xp)
            n = xp.sum(domain)
            n = int(n)
            # noinspection PyArgumentList

            domain_unsat = domain
            failed = True
            sub_loads = None
            i = 0
            while i < self._max_outer_it and n > 0:
                sub_loads, failed = gmres(conv_func, initial_guess, rbd[domain_unsat] - conv_func.conv_sat(),
                                          self._restart, max_it, self._tol)
                if failed:
                    raise ValueError(f"Sub model {self.name} failed to converge")
                initial_guess, still_going, n = conv_func.update_saturated(sub_loads)
                n = int(n)
                if not still_going:
                    break
                # noinspection PyArgumentList
                domain_unsat = conv_func.domain_unsat
                print('iteration:', i, 'domain_unsat:', xp.sum(domain_unsat))
                i += 1
            full_loads, disp_offsets, full_disp = conv_func.get_full_pressures(sub_loads)
            slip_nodes = conv_func.saturated_nodes

        # cache result
        if not failed:
            self.previous_result = full_loads
            self.previous_domain = domain

        if failed:
            print(f"Sub model {self.name} failed to converge")
        else:
            print(f"Sub model {self.name} converged successfully")

        # if f'total_displacement_{self.component[0]}' in current_state and self._add_to_existing:
        #     full_disp -= current_state[f'total_displacement_{self.component[0]}']
        rbd -= xp.expand_dims(disp_offsets, self.axis)
        gs = self.model.surface_1.grid_spacing

        return {f'loads_{self.component[0]}': full_loads,
                f'total_displacement_{self.component[0]}': full_disp,
                f'rigid_body_displacement_{self.component[0]}': rbd,
                f'total_tangential_force_{self.component[0]}': gs ** 2 * np.sum(full_loads),
                f'{self.name}_failed': failed,
                'slip_nodes': slip_nodes}


class ConvFuncWrapper:
    def __init__(self, conv_func, max_loads, domain, direction: int, multiplier, xp):
        self.first = xp.argmax(domain, axis=direction)
        if direction:
            self.first_index = xp.arange(domain.shape[not direction], dtype=int), self.first
        else:
            self.first_index = self.first, xp.arange(domain.shape[not direction], dtype=int)
        self.conv_func = conv_func
        self.max_loads = max_loads.copy()
        self.max_loads[self.first_index] = xp.inf
        self.domain = domain
        self.multiplier = multiplier
        self.saturated_nodes = xp.zeros_like(domain)
        self.domain_unsat = domain
        self.sat_pressures = []
        self.direction = direction
        self._xp = xp

    def update_saturated(self, pressures):
        new_sat = pressures >= self.max_loads[self.domain_unsat]
        self.saturated_nodes[self.domain_unsat] = new_sat
        condition = self._xp.all(self._xp.logical_not(self.saturated_nodes[self.first_index]))
        if not condition:
            print("Assertion failed")
            assert condition
        self.sat_pressures = self.max_loads[self.saturated_nodes]
        self.domain_unsat = self._xp.logical_and(self.domain, self._xp.logical_not(self.saturated_nodes))
        return pressures[self._xp.logical_not(new_sat)], self._xp.sum(new_sat), len(pressures) - self._xp.sum(new_sat)

    def unsat_all(self):
        self.saturated_nodes[:] = False
        self.domain_unsat = self.domain
        self.sat_pressures = []

    def conv_sat(self):
        full_loads = self._xp.zeros(self.domain.shape)
        full_loads[self.saturated_nodes] = self.sat_pressures
        full_disp = self.conv_func(full_loads)
        return full_disp[self.domain_unsat]

    def get_full_pressures(self, sub_loads, include_sat=True):
        full_loads = self._xp.zeros(self.domain.shape)
        full_loads[self.domain_unsat] = sub_loads.flatten()
        if include_sat:
            full_loads[self.saturated_nodes] = self.sat_pressures
        disp_offsets = full_loads[self.first_index]
        full_loads[self.first_index] = 0
        full_disp = self.conv_func(full_loads)
        return full_loads, disp_offsets, full_disp

    def __call__(self, sub_loads):
        sub_shape = sub_loads.shape
        full_loads, disp_offsets, full_disp = self.get_full_pressures(sub_loads, False)
        stripey = self._xp.expand_dims(disp_offsets, self.direction) * self.multiplier
        full_disp += stripey
        return full_disp[self.domain_unsat].reshape(sub_shape)
