import numpy as np
from numbers import Number
import typing
import warnings

from .steps import _ModelStep
from ._model_utils import get_gap_from_model
from ._step_utils import HeightOptimisationFunction, make_interpolation_func

__all__ = ['QuasiStaticStep']


class QuasiStaticStep(_ModelStep):
    """
    A model step for quasi static relative movement

    To be used for loading, unloading and sliding as well as quasi static impacts, can be used for rolling-sliding
    with appropriate sub models for tangential behaviour.

    Parameters
    ----------

    step_name: str
        An identifying name for the step used for errors and outputs
    number_of_steps: int
        The number of sub steps the problem will be split into, together with the time_period this controls the duration
        of the time steps used
    no_time: bool, optional (False)
        Set to true if there is no time dependence and the time steps can be solved in any order (no permanent changes
        between steps such as plastic deformation or heat generation), if True the model will be solved more efficiently
    time_period: float, optional (1.0)
        The total time period of this model step, used for solving sub-models and writing outputs
    off_set_x, off_set_y: float or Sequence of float, optional (0.0)
        The off set between the surfaces in the x and y directions, this can be a relative off set or an absolute off
        set, controlled by the relative_loading parameter:

        * A constant (float) value, indicating a constant offset between the surfaces (no relative movement of profiles)
        * A two element sequence of floats, indicating the the start and finish offsets, if this is used, the
          movement_interpolation_mode will be used to generate intermediate values
        * A 2 by n array of n absolute position values and n time values normalised to a 0-1 scale. array[0] should be
          position values and array[1] should be time values, time values must be between 0 and 1. The
          movement_interpolation_mode will be used to generate intermediate values
    interference, normal_load, mean_gap: float or Sequence of float, optional (None)
        The interference, normal load and mean gap between the surfaces, only one of these can be set, setting neither
        keeps the interference as it is at the start of this model step. As above for the off sets, either of these
        parameters can be:

        * A constant (float) value, indicating a constant load/ interference between the surfaces.
        * A two element sequence of floats, indicating the the start and finish load. interference, if this is used, the
          movement_interpolation_mode will be used to generate intermediate values
        * A 2 by n array of n absolute position values and n time values normalised to a 0-1 scale. array[0] should be
          position values and array[1] should be time values, time values must be between 0 and 1. The
          movement_interpolation_mode will be used to generate intermediate values
    relative_loading: bool, optional (False)
        If True the load or displacement and off set will be applied relative to the value at the start of the step,
        otherwise the absolute value will be used. eg, if the previous step ended with a load of 10N and this step ramps
        from 0 to 10N setting relative_loading to True will ramp the total load form 10 to 20N over this step.
    adhesion: bool, optional (True)
        If True the adhesion model set for the contact model will be used, If set to false this step will ignore the
        adhesion model (typically used for loading steps)
    unloading: bool, optional (False)
        If True the contact nodes will be constrained to be a sub set of those found in the previous time step.
    fast_ld: bool, optional (False)
        If True the load and displacements can be swapped to give a faster simulation, eg: if 100 equally spaced load
        steps are specified by a start and finish load, the displacement at the maximum load will be found and instead
        100 equally spaced displacement controlled steps will solved between just touching and the maximum deflection.
        This swapping makes the computation much faster as it removes an optimisation loop.
    impact_properties: list, optional (None)
        This is currently not supported, providing a value will cause an error
    movement_interpolation_mode: str or int, optional ('linear')
        Any valid input to scipy.interpolate.interp1d as the 'kind' parameter, using 'nearest', 'previous' or 'next'
        will cause a warning. This parameter controls how the offset and loading is interpolated over the step
    profile_interpolation_mode: {'nearest', 'linear'}, optional ('nearest')
        Used to generate the grid points for the second surface at the location of the grid points for the first
        surface, nearest ensures compatibility with sub models which change the profile, if the grid spacings of the
        surfaces match
    periodic_geometry: bool, optional (False)
        If True the surface profile will warp when applying the off set between the surfaces
    periodic_axes: tuple, optional ((False, False))
        For each True value the corresponding axis will be solved by circular convolution, meaning the result is
        periodic in that direction
    method: {'auto', 'pk', 'double', 'rey'}, optional ('auto')
        The method by which the normal contact is solved, only used for load controlled contact.
        'pk' uses the Polonsky and Keer algorithm for elastic contact.
        'double' uses a double iteration procedure, suitable for elastic contact with a maximum pressure.
        'auto' automatically selects 'pk' if there is no maximum pressure and 'double' if there is.
    max_it: int, optional (1000)
        The maximum number of iterations used in the main loop
    tolerance: float, optional (1e-8)
        The relative tolerance used for convergnece of the main loop
    max_it_outer: int, optional (100)
        Only used for the double iteration method
    tolerance_outer: float, optional (1e-4)
        The norm of the residual used to declare convergence of the bccg iterations
    tolerance: float, optional (1e-4)
        The norm of the residual used to declare convergence of the bccg iterations
    no_update_warning: bool, optional (True)
        Change to False to suppress warning given when no movement or loading changes are specified
    upper: float, optional (4.0)
        For load controlled contact the upper bound for the interference between the bodies will be this factor
        multiplied by the largest just touching gap. 4 is suitable for flat on flat contacts but for ball on flat
        contacts a lower value will give faster converging solutions

    Examples
    --------
    In the following example we model the contact between a rough cylinder and a flat plane.
    Both surface are elastic. This code could be used to generate load displacement curves.

    >>> import slippy.surface as s
    >>> import slippy.contact as c
    >>> # define contact geometry
    >>> cylinder = s.RoundSurface((1 ,np.inf, 1), shape=(256, 256), grid_spacing=0.001)
    >>> roughness = s.HurstFractalSurface(1, 0.2, 1000, shape=(256, 256), grid_spacing=0.001,
    >>>                                   generate = True)
    >>> combined = cylinder + roughness * 0.00001
    >>> flat = s.FlatSurface(shape=(256, 256), grid_spacing=0.001, generate = True)
    >>> # define material behaviour and assign to surfaces
    >>> material = c.Elastic('steel', properties = {'E':200e9, 'v':0.3})
    >>> combined.material = material
    >>> flat.material = material
    >>>
    >>> # make a contact model
    >>> my_model = c.ContactModel('qss_test', combined, flat)
    >>>
    >>> # make a modelling step to describe the problem
    >>> max_int = 0.002
    >>> n_time_steps = 20
    >>> my_step = c.QuasiStaticStep('loading', n_time_steps, no_time=True,
    >>>                             interference = [max_int*0.001, max_int],
    >>>                             periodic_geometry=True, periodic_axes = (False, True))
    >>> # add the steps to the model
    >>> my_model.add_step(my_step)
    >>> # add output requests
    >>> output_request = c.OutputRequest('Output-1',
    >>>                                  ['interference', 'total_normal_load',
    >>>                                   'loads_z', 'total_displacement',
    >>>                                   'converged'])
    >>> my_step.add_output(output_request)
    >>> # solve the model
    >>> final_result = my_model.solve()
    """
    _just_touching_gap = None
    _adhesion_model = None
    _initial_contact_nodes = None
    _upper = None

    def __init__(self, step_name: str, number_of_steps: int, no_time: bool = False,
                 time_period: float = 1.0,
                 off_set_x: typing.Union[float, typing.Sequence[float]] = 0.0,
                 off_set_y: typing.Union[float, typing.Sequence[float]] = 0.0,
                 interference: typing.Union[float, typing.Sequence[float]] = None,
                 normal_load: typing.Union[float, typing.Sequence[float]] = None,
                 mean_gap: typing.Union[float, typing.Sequence[float]] = None,
                 relative_loading: bool = False,
                 adhesion: bool = True,
                 unloading: bool = False,
                 fast_ld: bool = False,
                 impact_properties: dict = None,
                 movement_interpolation_mode: str = 'linear',
                 profile_interpolation_mode: str = 'nearest',
                 periodic_geometry: bool = False, periodic_axes: tuple = (False, False),
                 method: str = 'auto',
                 max_it: int = 1000, tolerance=1e-8,
                 max_it_outer: int = 100, tolerance_outer=1e-4,
                 no_update_warning: bool = True,
                 upper: float = 4.0):

        # movement interpolation mode sort out movement interpolation mode make array of values
        if impact_properties is not None:
            raise NotImplementedError("Impacts are not yet implemented")
        # work out the time step if needed:
        self._no_time = no_time
        self.total_time = time_period
        if not no_time and fast_ld:
            raise ValueError("Cannot have time dependence and fast_ld, either set no_time True or set fast_ld False")
        self._fast_ld = fast_ld
        self._relative_loading = relative_loading
        self.profile_interpolation_mode = profile_interpolation_mode
        self._periodic_profile = periodic_geometry
        self._periodic_axes = periodic_axes
        self._max_it_outer = max_it_outer
        self._r_tol_outer = tolerance_outer
        self._max_it = max_it
        self._r_tol = tolerance
        self.number_of_steps = number_of_steps

        if method not in {'auto', 'pk', 'double', 'rey'}:
            raise ValueError(f"Unrecognised method for step {step_name}: {method}")
        sum_param = (normal_load is not None) + (interference is not None) + (mean_gap is not None)
        if sum_param > 1 or sum_param == 0:
            raise ValueError(f"Exactly one of normal_load, interference and mean_gap must be set, {sum_param} were set")
        if mean_gap is not None:
            if method not in {'auto', 'rey'}:
                raise ValueError("pk and double methods don't support mean gap")
            else:
                method = 'rey'
        if interference is not None and method == 'rey':
            raise ValueError("Rey method doesn't support interference")

        self._method = method
        self._height_optimisation_func = None
        self._adhesion = adhesion
        self._unloading = unloading
        self._upper_factor = upper

        self.time_step = time_period / number_of_steps

        # check that something is actually changing
        self.update = set()

        if not isinstance(off_set_x, Number) or not isinstance(off_set_y, Number):
            if no_time:
                raise ValueError("Can not have no time dependence and sliding contact")
            off_set_x = [off_set_x] * 2 if isinstance(off_set_x, Number) else off_set_x
            off_set_y = [off_set_y] * 2 if isinstance(off_set_y, Number) else off_set_y
            off_set_x_func = make_interpolation_func(off_set_x, movement_interpolation_mode, 'relative_off_set_x')
            off_set_y_func = make_interpolation_func(off_set_y, movement_interpolation_mode, 'relative_off_set_y')
            self._off_set_upd = lambda time: np.array([off_set_x_func(time), off_set_y_func(time)])
            self.update.add('off_set')
            self.off_set = None
        else:
            self.off_set = np.array([off_set_x, off_set_y])

        if normal_load is not None and interference is not None:
            raise ValueError("Both normal_load and interference are set, only one of these can be set")
        if normal_load is None and interference is None:
            if relative_loading:
                interference = 0
            else:
                raise ValueError("Cannot have no set load or interference and not relative loading, set either the"
                                 "normal load, normal interference or change relative_loading to True")

        self.load_controlled = False

        if normal_load is not None:
            if isinstance(normal_load, Number):
                self.normal_load = normal_load
            else:
                self.normal_load = None
                self._normal_load_upd = make_interpolation_func(normal_load, movement_interpolation_mode,
                                                                'normal_load')
                self.update.add('normal_load')
            self.load_controlled = True
        else:
            self.normal_load = None

        if mean_gap is not None:
            if isinstance(mean_gap, Number):
                self.mean_gap = mean_gap
            else:
                self.mean_gap = None
                self._mean_gap_upd = make_interpolation_func(mean_gap, movement_interpolation_mode,
                                                             'mean_gap')
                self.update.add('mean_gap')
        else:
            self.mean_gap = None

        if interference is not None:
            if isinstance(interference, Number):
                self.interference = interference
            else:
                self.interference = None
                self._interference_upd = make_interpolation_func(interference, movement_interpolation_mode,
                                                                 'interference')
                self.update.add('interference')
        else:
            self.interference = None

        if not self.update and no_update_warning:
            warnings.warn("Nothing set to update")

        provides = {'off_set', 'loads_z', 'surface_1_displacement_z', 'surface_2_displacement_z',
                    'total_displacement_z', 'interference', 'just_touching_gap', 'surface_1_points', 'contact_nodes',
                    'surface_2_points', 'time', 'time_step', 'new_step', 'converged', 'gap', 'total_normal_load'}
        super().__init__(step_name, time_period, provides)

    def solve(self, previous_state, output_file):
        current_time = previous_state['time']
        if self._fast_ld:
            # solve the normal contact problem
            raise NotImplementedError("TODO")
            # TODO
            # change to disp controlled, set the displacement variable

        for s in self.sub_models:
            s.no_time = self._no_time

        if self.load_controlled:
            update_func = self._solve_load_controlled
        else:  # displacement controlled
            update_func = self._solve_displacement_controlled

        relative_time = np.linspace(0, 1, self.number_of_steps + 1)[1:]
        just_touching_gap = None

        original = dict()

        if self._relative_loading:
            original['normal_load'] = previous_state['total_normal_load'] if 'total_normal_load' in previous_state \
                else 0
            original['interference'] = previous_state['interference'] if 'interference' in previous_state else 0
            original['off_set'] = np.array(previous_state['off_set']) if 'off_set' in previous_state else \
                np.array([0, 0])

        self._adhesion_model = self.model.adhesion if self._adhesion else None
        max_pressure = min(self.model.surface_1.material.max_load, self.model.surface_2.material.max_load)

        if self._method == 'auto':
            if not np.isinf(max_pressure):
                if self._adhesion_model:
                    raise ValueError("Adhesion and maximum load not allowed")
                self._method = 'double'
            elif self._adhesion_model is not None:
                self._method = 'rey'
            else:
                self._method = 'pk'

        current_state = dict()

        for i in range(self.number_of_steps):
            self.update_movement(relative_time[i], original)
            # find overlapping nodes
            if 'off_set' in self.update or just_touching_gap is None or not self._no_time:
                just_touching_gap, surface_1_points, surface_2_points \
                    = get_gap_from_model(self.model, interference=0, off_set=self.off_set,
                                         mode=self.profile_interpolation_mode, periodic=self._periodic_profile)
                self._just_touching_gap = just_touching_gap
                self._upper = None
            current_state = dict(just_touching_gap=just_touching_gap, surface_1_points=surface_1_points,
                                 surface_2_points=surface_2_points, off_set=self.off_set,
                                 time_step=self.time_step)
            if i == 0:
                current_state['new_step'] = True
            else:
                current_state['new_step'] = False
            # solve contact
            if self._unloading and 'contact_nodes' in previous_state:
                initial_contact_nodes = previous_state['contact_nodes']
            else:
                initial_contact_nodes = None

            self._initial_contact_nodes = initial_contact_nodes
            print('#####################################################\nTime step:', i,
                  '\n#####################################################')
            print('Set load:', self.normal_load)

            results = update_func(current_state)
            current_state.update(results)
            current_time += self.time_step
            current_state['time'] = current_time
            # solve sub models
            self.solve_sub_models(current_state)
            self.save_outputs(current_state, output_file)

        return current_state

    @property
    def upper(self):
        if self._upper is None:
            self._upper = np.max(self._just_touching_gap) * self._upper_factor
        return self._upper

    def update_movement(self, relative_time, original):
        for name in self.update:
            if self._relative_loading:
                self.__setattr__(name, original[name] + self.__getattribute__(f'_{name}_upd')(relative_time))
            else:
                self.__setattr__(name, self.__getattribute__(f'_{name}_upd')(relative_time))

    def _solve_load_controlled(self, current_state) -> dict:
        # if there is time dependence or we don't already have one, make a new height optimiser
        if not self._no_time or self._height_optimisation_func is None:
            opt_func = HeightOptimisationFunction(just_touching_gap=self._just_touching_gap,
                                                  model=self.model,
                                                  adhesion_model=self._adhesion_model,
                                                  initial_contact_nodes=self._initial_contact_nodes,
                                                  max_it=self._max_it,
                                                  rtol=self._r_tol,
                                                  material_options=dict(),
                                                  max_set_load=self.normal_load,
                                                  rtol_outer=self._r_tol_outer,
                                                  max_it_outer=self._max_it_outer,
                                                  periodic_axes=self._periodic_axes)
            self._height_optimisation_func = opt_func
        else:
            opt_func = self._height_optimisation_func

        if self._unloading and 'contact_nodes' in current_state:
            contact_nodes = current_state['contact_nodes']
        else:
            contact_nodes = None
            # contact_nodes = np.ones(self._just_touching_gap.shape, dtype=np.bool)

        if self._method == 'pk':
            opt_func.contact_nodes = None
            opt_func.p_and_k(self.normal_load)
        elif self._method == 'rey':
            opt_func.contact_nodes = None
            opt_func.rey(target_load=self.normal_load)
        else:
            opt_func.change_load(self.normal_load, contact_nodes)
            # need to set bounds and pick a sensible starting point
            upper = self.upper
            print(f'upper bound set at: {upper}')
            if self._no_time:
                brackets = opt_func.get_bounds_from_cache(0, upper)
            else:
                brackets = (0, upper)
            print(f'Bounds adjusted using cache to: {brackets}')
            print(f'Interference tolerance set to {self._r_tol_outer} Relative')
            opt_func.brent(0, upper)

        # noinspection PyProtectedMember

        results = opt_func.results
        load_conv = (np.abs(results['total_normal_load'] - self.normal_load) / self.normal_load) < 0.05
        results['converged'] = bool(load_conv) and not opt_func.last_call_failed
        return results

    def _solve_displacement_controlled(self, current_state):
        # Note also includes gap control for rey
        if not self._no_time or self._height_optimisation_func is None:
            opt_func = HeightOptimisationFunction(just_touching_gap=self._just_touching_gap,
                                                  model=self.model,
                                                  adhesion_model=self._adhesion_model,
                                                  initial_contact_nodes=self._initial_contact_nodes,
                                                  max_it=self._max_it,
                                                  rtol=self._r_tol,
                                                  material_options=dict(),
                                                  max_set_load=1,
                                                  rtol_outer=self._r_tol_outer,
                                                  max_it_outer=self._max_it_outer,
                                                  periodic_axes=self._periodic_axes)
            self._height_optimisation_func = opt_func
        else:
            opt_func = self._height_optimisation_func

        if self._method == 'rey':
            opt_func.rey(target_mean_gap=self.mean_gap)
            return opt_func.results

        if self._unloading and 'contact_nodes' in current_state:
            contact_nodes = current_state['contact_nodes']
        else:
            contact_nodes = None
            # contact_nodes = np.ones(self._just_touching_gap.shape, dtype=np.bool)

        opt_func.change_load(1, contact_nodes)
        _ = opt_func(self.interference, current_state)
        results = opt_func.results
        results['interference'] = self.interference
        results['converged'] = not opt_func.last_call_failed
        return results

    def __repr__(self):
        return f'{self.name}: QuasiStaticStep'
