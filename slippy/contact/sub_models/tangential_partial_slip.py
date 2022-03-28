import typing
import numpy as np
import slippy
from ._TransientSubModelABC import _TransientSubModelABC
from slippy.core.materials import _IMMaterial
from slippy.core.influence_matrix_utils import bccg, plan_convolve
# TODO add from_offset option to get the displacement from the offset


class TangentialPartialSlip(_TransientSubModelABC):
    """ Solves the partial slip problem

    Parameters
    ----------
    axis: int
        The axis along which the displacement will be applied (0 or 1).
    displacement: float or sequence of floats
        The rigid body displacement between the parts. Suitable values are:
            - float: indicating a constant displacement
            - sequence of 2 values indicating the values at the start and end of the model step, intermediate values
              will be linearly interpolated.
            - 2 by n array: of time points and displacement values, intermediate values will be interpolated by the
              method specified by the interpolation_mode parameter (defaults to linear)
            - None
        If an array is supplied and it is too short it is extrapolated by repeating the final value, this produces a
        warning. If neither are supplied this sub-model requires rigid_body_displacement to be provided by a further
        sub-model
    periodic_axes: 2 element sequence of bool, optional (False, False)
        True for each axis which the solution should be periodic in, should match solving step
    tol: float, optional (1e-7)
        The tolerance used to declare convergence for the bccg iterations
    max_it: int, optional (None)
        The maximum number of iterations for the bccg iterations, defaults to the same as the number of contact nodes
    interpolation_mode: str, optional ('linear')
        The interpolation mode used when a 2 by n array of values, can be any method compatible with scipy.interp1d
    name: str, optional ("TangentialPartialSlip")
        The name of the sub model, used for debugging and in the log file
    """

    def __init__(self, axis: int,
                 displacement: typing.Optional[typing.Union[float, typing.Sequence]],
                 periodic_axes: typing.Sequence[bool] = (False, False),
                 tol: float = 1e-7, max_it: int = None, interpolation_mode: str = 'linear',
                 name: str = "TangentialPartialSlip"):

        requires = {'maximum_tangential_force', 'contact_nodes', 'time'}
        self.direction = 'x' if axis else 'y'
        if displacement is None:
            self.displacement_from_sub_model = True
            displacement = 0.0
            requires.add('rigid_body_displacement_' + self.direction)
        else:
            self.displacement_from_sub_model = False
        provides = {'slip_distance', 'stick_nodes', f'loads_{self.direction}', f'total_displacement_{self.direction}'}
        super().__init__(name, requires, provides, transient_names=[f'rigid_body_displacement_{self.direction}'],
                         transient_values=[displacement],
                         interpolation_mode=interpolation_mode)

        self.component = self.direction * 2
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

    def _solve(self, current_state: dict, **kwargs) -> dict:
        span = [(2-pa) * s for pa, s in zip(self._periodic_axes, current_state['maximum_tangential_force'].shape)]

        if not self._pre_solve_checks or span != self._last_span:
            self._check(span)
            self._last_span = span

        domain = current_state['contact_nodes']

        conv_func = plan_convolve(current_state['maximum_tangential_force'], self._im_total, domain,
                                  circular=self._periodic_axes)
        # if the displacements are provided by another sub model or we have a set displacement we just have one set
        # of bccg iterations:
        if self.displacement_from_sub_model:
            displacement = current_state['rigid_body_displacement_' + self.direction]
        else:
            displacement = kwargs[f'rigid_body_displacement_{self.direction}']

        set_displacement = float(displacement)*np.ones(current_state['maximum_tangential_force'].shape)

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
        rtn_dict = dict()
        rtn_dict['stick_nodes'] = stick_nodes
        tangential_deformation = slippy.asnumpy(conv_func(loads_in_domain, True))
        rtn_dict['loads_' + self.component[0]] = full_loads

        if 'total_displacement_' + self.component[0] in current_state:
            rtn_dict['total_displacement_' + self.component[0]] += tangential_deformation
        else:
            rtn_dict['total_displacement_' + self.component[0]] = tangential_deformation

        slip_distance = set_displacement-tangential_deformation
        slip_distance[stick_nodes] = 0
        slip_distance[np.logical_not(domain)] = 0
        rtn_dict['slip_distance'] = slip_distance
        return rtn_dict
