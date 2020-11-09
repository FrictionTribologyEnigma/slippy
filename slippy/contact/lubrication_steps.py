"""
Model steps for lubricated contacts
"""
import typing
import warnings
from collections.abc import Sequence
from numbers import Number

import numpy as np
import slippy

from slippy.abcs import _NonDimensionalReynoldSolverABC
from ._material_utils import Loads, Displacements
from ._model_utils import get_gap_from_model
from ._step_utils import make_interpolation_func, solve_normal_loading
from .influence_matrix_utils import plan_convolve
from .steps import _ModelStep
from .materials import _IMMaterial

__all__ = ['IterSemiSystem']


class IterSemiSystem(_ModelStep):
    """
    Parameters
    ----------
    step_name: str
        An identifying name for the step used for errors and outputs
    reynolds_solver: _NonDimensionalReynoldSolverABC
        A reynolds solver object which will be used to solve for pressures
    rolling_speed: float or Sequence of float, optional (None)
        The mean speed of the surfaces (u1+u2)/2 parameters can be:
        - A constant (float) value, indicating a constant rolling speed.
        - A two element sequence of floats, indicating the the start and finish rolling speed, if this is used, the
          movement_interpolation_mode will be used to generate intermediate values
        - A 2 by n array of n rolling speed and n time values normalised to a 0-1 scale. array[0] should be
          position values and array[1] should be time values, time values must be between 0 and 1. The
          movement_interpolation_mode will be used to generate intermediate values
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
        - A constant (float) value, indicating a constant offset between the surfaces (no relative movement of profiles)
        - A two element sequence of floats, indicating the the start and finish offsets, if this is used, the
          movement_interpolation_mode will be used to generate intermediate values
        - A 2 by n array of n absolute position values and n time values normalised to a 0-1 scale. array[0] should be
          position values and array[1] should be time values, time values must be between 0 and 1. The
          movement_interpolation_mode will be used to generate intermediate values
    interference, normal_load: float or Sequence of float, optional (None)
        The interference and normal load between the surfaces, only one of these can be set (the other will be solved
        for) setting neither keeps the interference as it is at the start of this model step. As above for the off sets,
        either of these parameters can be:
        - A constant (float) value, indicating a constant load/ interference between the surfaces.
        - A two element sequence of floats, indicating the the start and finish load. interference, if this is used, the
          movement_interpolation_mode will be used to generate intermediate values
        - A 2 by n array of n absolute position values and n time values normalised to a 0-1 scale. array[0] should be
          position values and array[1] should be time values, time values must be between 0 and 1. The
          movement_interpolation_mode will be used to generate intermediate values
    relative_loading: bool, optional (False)
        If True the load or displacement and off set will be applied relative to the value at the start of the step,
        otherwise the absolute value will be used. eg, if the previous step ended with a load of 10N and this step ramps
        from 0 to 10N setting relative_loading to True will ramp the total load form 10 to 20N over this step.
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
    max_it_pressure: int, optional (100)
        The maximum number of iterations in the fluid pressure calculation loop
    rtol_pressure: float, optional (1e-7)
        The relative tolerance for the fluid pressure calculation loop
    max_it_interference: int, optional (100)
        The maximum number of iterations in the loop that finds the interference between the surfaces
    rtol_interference: float, optional (1e-7)
        The relative tolerance on the total load (integral of the pressure solution minus the applied load)
    initial_guess: {callable, 'previous', list}, optional ('previous')
        The initial guess for the interference, and/ or pressure profile between the surfaces, any callable will be
        called with the contact model and the undeformed nd_gap as positional arguments, it must return the interference
        and the pressure profile. 'previous' will use the result(s) from the previous step, if results are not found the
        interference and pressure profile will be set to 0. Can also be a 2 element list, the first element being the
        interference and the second being the pressure profile as an array, if this is the wrong shape zeros wil be
        used.
    no_update_warning: bool, optional (True)
        Change to False to suppress warning given when no movement or loading changes are specified

    Attributes
    ----------

    Methods
    -------

    Notes
    -----
    The solver iterates through a 'pressure loop' until the solution has converged to a set of pressure values, then the
    loading is checked, if the total pressure is too low the surfaces are brought closer together. This is continued
    until the total load has converged to the set value. This outer loop is referred to as the interference loop.
    """
    "The minimum number of iterations in the reynolds solving loop"
    _dh = 0
    "The base change in height"
    _interferences = list()
    _load_errors = list()

    _reynolds: typing.Optional[_NonDimensionalReynoldSolverABC] = None
    initial_guess: typing.Optional[typing.Union[typing.Callable, list, str]]

    def __init__(self, step_name: str, reynolds_solver: _NonDimensionalReynoldSolverABC,
                 rolling_speed: typing.Union[float, typing.Sequence[float]],
                 number_of_steps: int = 1,
                 no_time: bool = False, time_period: float = 1.0,
                 off_set_x: typing.Union[float, typing.Sequence[float]] = 0.0,
                 off_set_y: typing.Union[float, typing.Sequence[float]] = 0.0,
                 interference: typing.Union[float, typing.Sequence[float]] = None,
                 normal_load: typing.Union[float, typing.Sequence[float]] = None,
                 relative_loading: bool = False,
                 movement_interpolation_mode: str = 'linear',
                 profile_interpolation_mode: str = 'nearest',
                 periodic_geometry: bool = False, periodic_axes: tuple = (False, False),
                 max_it_pressure: int = 5000, rtol_pressure: float = 2e-6,
                 max_it_interference: int = 5000,
                 rtol_interference: float = 1e-4,
                 relaxation_factor: float = 0.1,
                 initial_guess: typing.Union[typing.Callable, str, typing.Sequence] = 'previous',
                 no_update_warning: bool = True):

        self._adjust_height_every_step = True
        self._initial_guess = initial_guess
        self._no_time = no_time
        self.total_time = time_period
        self._relative_loading = relative_loading
        self.profile_interpolation_mode = profile_interpolation_mode
        self._periodic_profile = periodic_geometry
        self._periodic_axes = periodic_axes

        self._max_it_pressure = max_it_pressure
        self._max_it_interference = max_it_interference
        self._rtol_pressure = rtol_pressure
        self._rtol_interference = rtol_interference
        self._max_pressure = np.inf

        self.reynolds = reynolds_solver

        if relaxation_factor <= 0 or relaxation_factor > 1:
            raise ValueError("Relaxation factor must be greater than 0 and less than or equal to 1")
        self._relaxation_factor = relaxation_factor

        self.time_step = time_period / number_of_steps
        self.number_of_steps = number_of_steps

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

        if isinstance(rolling_speed, Number):
            self.rolling_speed = rolling_speed
        else:
            self.rolling_speed = None
            self._rolling_speed_upd = make_interpolation_func(rolling_speed, movement_interpolation_mode,
                                                              'rolling_speed')
            self.update.add('rolling_speed')

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

        if interference is not None:
            if isinstance(interference, Number):
                self.interference = interference
            else:
                self.interference = None
                self._interference_upd = make_interpolation_func(interference, movement_interpolation_mode,
                                                                 'interference')
                self.update.add('interference')
            self.load_controlled = False

        if not self.update and no_update_warning:
            warnings.warn("Nothing set to update")

        self._provides = None

        base_provides = {'just_touching_gap', 'surface_1_points', 'surface_2_points', 'off_set', 'time_step', 'time',
                         'interference', 'total_normal_load', 'pressure', 'nd_pressure', 'loads',
                         'surface_1_displacement', 'surface_2_displacement', 'total_displacement', 'converged',
                         'gap', 'nd_gap'}

        provides = base_provides.union(reynolds_solver.provides).union(reynolds_solver.requires)

        super().__init__(step_name, time_period, provides)

    @property
    def provides(self):
        results_set = self._provides
        if self.model is None:
            return results_set
        if self.model.lubricant_model is None:
            return results_set
        return results_set.union(set(self.model.lubricant_model.sub_models.keys()))

    @provides.setter
    def provides(self, value):
        if self._provides is None:
            self._provides = value
        else:
            raise ValueError("The provides property can only be set during instantiation")

    @property
    def reynolds(self):
        return self._reynolds

    @reynolds.setter
    def reynolds(self, value):
        if isinstance(value, _NonDimensionalReynoldSolverABC):
            self._reynolds = value
        else:
            raise ValueError("Cannot set a non reynolds solver object as the reynolds solver, to use custom solvers"
                             f"first subclass _NonDimensionalReynoldSolverABC from slippy.abcs, received "
                             f"type was {type(value)}")

    @reynolds.deleter
    def reynolds(self):
        self._reynolds = None

    def data_check(self, previous_state: set):
        pass

    def update_movement(self, relative_time, original):
        for name in self.update:
            if self._relative_loading:
                self.__setattr__(name, original[name] + self.__getattribute__(f'_{name}_upd')(relative_time))
            else:
                self.__setattr__(name, self.__getattribute__(f'_{name}_upd')(relative_time))

    def solve(self, previous_state: dict, output_file):
        cuda = slippy.CUDA
        slippy.CUDA = False
        start_time = previous_state['time']
        gs = self.model.surface_1.grid_spacing

        im_mats = (isinstance(self.model.surface_1.material, _IMMaterial) and
                   isinstance(self.model.surface_2.material, _IMMaterial))
        surf_1_material = self.model.surface_1.material
        surf_2_material = self.model.surface_2.material

        for s in self.sub_models:
            s.no_time = self._no_time

        relative_time = np.linspace(0, 1, self.number_of_steps)
        just_touching_gap = None

        original = dict()

        if self._relative_loading:
            original['normal_load'] = previous_state['total_normal_load'] if 'total_normal_load' in previous_state \
                else 0
            original['interference'] = previous_state['interference'] if 'interference' in previous_state else 0
            original['off_set'] = np.array(previous_state['off_set']) if 'off_set' in previous_state else \
                np.array([0, 0])

        previous_gap_shape = None  # shape of just touching gap array

        for i in range(self.number_of_steps):
            self.update_movement(relative_time[i], original)
            self.reynolds.rolling_speed = self.rolling_speed
            # find overlapping nodes
            if 'off_set' in self.update or just_touching_gap is None or not self._no_time:
                just_touching_gap, surface_1_points, surface_2_points \
                    = get_gap_from_model(self.model, interference=0, off_set=self.off_set,
                                         mode=self.profile_interpolation_mode, periodic=self._periodic_profile)

            time_step_current_state = dict(just_touching_gap=just_touching_gap, surface_1_points=surface_1_points,
                                           surface_2_points=surface_2_points, off_set=self.off_set,
                                           time_step=self.time_step, time=start_time+(i+1)*self.time_step)

            # make a new loads function if we need it
            if (previous_gap_shape is None or previous_gap_shape != just_touching_gap.shape) and im_mats:
                span = just_touching_gap.shape
                max_pressure = min([surf_1_material.max_load, surf_2_material.max_load])
                self._max_pressure = max_pressure
                im1 = surf_1_material.influence_matrix(span=span, grid_spacing=[gs] * 2,
                                                       components=['zz'])['zz']
                im2 = surf_2_material.influence_matrix(span=span, grid_spacing=[gs] * 2,
                                                       components=['zz'])['zz']
                total_im = im1 + im2
                loads_func = plan_convolve(just_touching_gap, total_im, circular=self._periodic_axes)
                previous_gap_shape = just_touching_gap.shape

            elif not im_mats:
                def loads_func(loads):
                    return solve_normal_loading(loads=Loads(z=loads, x=None, y=None), model=self.model,
                                                deflections='z', current_state=time_step_current_state)[0].z
            # sort out initial guess
            if i >= 0:
                initial_guess = 'previous'
            else:
                initial_guess = self.initial_guess

            if initial_guess is None:
                initial_guess = [self.reynolds.dimensionalise_gap(0.01),
                                 self.reynolds.dimensionalise_pressure(0.05)]
            if isinstance(initial_guess, str) and initial_guess.lower() == 'previous':
                pressure = np.zeros_like(just_touching_gap) if 'pressure' not in previous_state else \
                    previous_state['pressure']
                interference = 0.0 if 'interference' not in previous_state else previous_state['interference']
            elif isinstance(initial_guess, Sequence):
                interference = initial_guess[0]
                if isinstance(initial_guess[1], Number):
                    pressure = initial_guess[1] * np.ones_like(just_touching_gap)
                else:
                    try:
                        pressure = np.asarray(initial_guess[1], dtype=np.float)
                        assert (pressure.shape == just_touching_gap.shape)
                    except ValueError:
                        raise ValueError('Initial guess for pressure could not be converted to a numeric array')
                    except AssertionError:
                        # noinspection PyUnboundLocalVariable
                        raise ValueError("Initial guess for pressure produced an array of the wrong size:"
                                         f"expected {just_touching_gap.shape}, got: {pressure.shape}")
            elif hasattr(initial_guess, '__call__'):
                interference, pressure = initial_guess(self.model, just_touching_gap)
            else:
                raise ValueError('Unsupported type for initial guess')

            results_last_it = {'nd_pressure': self.reynolds.dimensionalise_pressure(pressure, True),
                               'just_touching_gap': just_touching_gap,
                               'interference': interference,
                               'pressure': pressure}

            # we have the interference, and the pressure initial guesses, find the initial displacement before solving

            if not (results_last_it['nd_pressure'] == 0).all():
                # noinspection PyUnboundLocalVariable
                results_last_it['total_displacement_z'] = loads_func(previous_state['pressure'])

            else:
                results_last_it['total_displacement_z'] = np.zeros_like(just_touching_gap)
            results_last_it = self.model.lubricant_model.solve_sub_models(results_last_it)
            # main loops
            it_num = 0
            # Find the gap and non denationalise it
            gap = just_touching_gap + results_last_it['total_displacement_z'] - results_last_it['interference']
            results_last_it['nd_interference'] = self.reynolds.dimensionalise_gap(results_last_it['interference'], True)
            results_last_it['gap'] = gap

            while True:
                nd_gap = self.reynolds.dimensionalise_gap(results_last_it['gap'], True)
                results_last_it['nd_gap'] = nd_gap
                # if flag:
                #     return locals()
                # else:
                #     flag = True
                # solve reynolds equation
                results_this_it = self.reynolds.solve(results_last_it, self._max_pressure)

                # add just touching gap, needed for sub models
                results_this_it['just_touching_gap'] = just_touching_gap

                # check for pressure convergence
                change_in_pressures = results_this_it['nd_pressure'] - results_last_it['nd_pressure']
                total_nd_pressure = np.sum(results_last_it['nd_pressure'])  # use previous state here ... more stable
                if total_nd_pressure > 0:
                    pressure_relative_error = np.sum(np.abs(change_in_pressures)) / total_nd_pressure
                else:
                    pressure_relative_error = 1
                pressure_converged = pressure_relative_error < self._rtol_pressure

                # apply the relaxation factor to the pressure result
                results_this_it['nd_pressure'] = (results_last_it['nd_pressure'] +
                                                  self._relaxation_factor * change_in_pressures)

                # solve contact geometry
                results_this_it['pressure'] = self.reynolds.dimensionalise_pressure(results_this_it['nd_pressure'])
                results_this_it['total_displacement_z'] = loads_func(results_this_it['pressure'])

                # find gap
                gap = just_touching_gap + results_this_it['total_displacement_z'] - results_last_it['interference']
                results_this_it['gap'] = gap

                # solve lubricant sub models
                results_this_it = self.model.lubricant_model.solve_sub_models(results_this_it)

                # check for load convergence
                total_load = np.sum(results_this_it['pressure']) * gs ** 2

                if self.load_controlled:
                    load_relative_error = (total_load / self.normal_load) - 1
                    load_converged = abs(load_relative_error) < self._rtol_pressure
                else:
                    load_converged = True
                    load_relative_error = 0.0

                results_this_it['nd_gap'] = self.reynolds.dimensionalise_gap(results_this_it['gap'], True)
                results_this_it['interference'] = results_last_it['interference']

                # escape the loop if it converged
                if pressure_converged and load_converged:
                    converged = True
                    print(f"Step {self.name} converged successfully after {it_num} iterations.")
                    print(f"Converged load is {total_load}, last change in pressure was {pressure_relative_error}\n")
                    break

                # escape the loop it if failed
                if it_num > self._max_it_pressure:  # this logic has changed used to just check the error
                    converged = False
                    print(f"Step {self.name} failed to converge after {it_num} iterations.\n")
                    print("Consider increasing the maximum number of iterations or reducing the relaxation factor")
                    print(f"Converged load is {total_load}, last change in pressure was {pressure_relative_error}")
                    break

                # adjust height for load balance

                if self._adjust_height_every_step and self.load_controlled:
                    # adjust height based on load balance
                    new_nd_interference = self.update_interference(it_num, pressure_relative_error,
                                                                   load_relative_error,
                                                                   results_last_it['nd_interference'],
                                                                   np.mean(results_this_it['nd_gap']),
                                                                   np.min(results_this_it['nd_gap']))
                    interference_updated = True

                elif pressure_converged:
                    # adjust height based on load balance
                    new_nd_interference = self.update_interference(it_num, pressure_relative_error,
                                                                   load_relative_error,
                                                                   results_last_it['nd_interference'],
                                                                   np.mean(results_this_it['nd_gap']),
                                                                   np.min(results_this_it['nd_gap']))
                    interference_updated = True
                else:
                    new_nd_interference = 0
                    interference_updated = False

                if interference_updated:
                    results_this_it['nd_interference'] = new_nd_interference
                    results_this_it['interference'] = self.reynolds.dimensionalise_gap(new_nd_interference)
                    gap = (just_touching_gap + results_this_it['total_displacement_z'] - results_this_it['interference'])
                    results_this_it['gap'] = gap
                    # print summary of iteration to log file
                    old_int = results_last_it['nd_interference']
                    print(f'{it_num}\ter_load: {load_relative_error:.4g}\t'
                          f'er_press: {pressure_relative_error:.4g}\t'
                          f'old_int: {old_int:.6g}\t'
                          f'new_int: {new_nd_interference:.6g}')
                else:
                    results_this_it['nd_interference'] = results_last_it['nd_interference']
                    results_this_it['interference'] = results_last_it['interference']

                it_num += 1
                results_last_it = results_this_it

            # clean up after it has converged
            current_state = {**time_step_current_state, **results_this_it}
            pressure = current_state['pressure']
            if im_mats:
                current_state['surface_1_displacement'] = \
                    Displacements(z=plan_convolve(pressure, im1, circular=self._periodic_axes)(pressure))
                current_state['surface_2_displacement'] = \
                    Displacements(z=plan_convolve(pressure, im2, circular=self._periodic_axes)(pressure))
                current_state['total_displacement'] = Displacements(z=current_state['total_displacement_z'])

            else:
                all_disp = solve_normal_loading(Loads(z=pressure), self.model, current_state, 'z')
                current_state['total_displacement'] = all_disp[0]
                current_state['surface_1_displacement'] = all_disp[1]
                current_state['surface_2_displacement'] = all_disp[2]

            del current_state['total_displacement_z']
            current_state['total_normal_load'] = total_load
            current_state['loads'] = Loads(z=current_state['pressure'])
            current_state['converged'] = converged
            current_state = self.solve_sub_models(current_state)
            self.save_outputs(current_state, output_file)

            previous_state = current_state

        slippy.CUDA = cuda

        return current_state

    def __repr__(self):
        return "Lubrication step"

    def update_interference(self, it_num, pressure_error_rel, load_error_rel, current_interference, mean_gap, min_gap):
        """This method updates the interference between the 2 surfaces during solution

        Parameters
        ----------
        it_num: int
            The current iteration number of the solution
        pressure_error_rel
            The current relative pressure error on the solution
        load_error_rel: float
            The current relative load error
        current_interference: float
            The current interference between the two bodies (the maximum overlap between the undeformed profiles)
        mean_gap
            The average nd_gap between the surfaces
        min_gap
            The minimum nd_gap between the surfaces

        Returns
        -------
        new_interference: float
            The non dimensional interference between the surfaces

        Notes
        -----
        This method is quite basic at the moment and can definitely be improved, if executed on every loop, the height
        is just updated by a fixed proportion of the minimum nd_gap size

        If updated only when the solver converges, the Regula-Falsi method is used
        """
        if self._adjust_height_every_step:
            new_interference = current_interference - 0.1 * load_error_rel
            return new_interference

        # else:  # height only adjusted when the load has converged, in this case use the Regula-Falsi method
        self._interferences.append(current_interference)
        self._load_errors.append(load_error_rel)

        if len(self._interferences) == 1:  # if this is the first guess give a value that will probably bound it
            new_interference = current_interference + abs(mean_gap) * -np.sign(load_error_rel)
            return new_interference

        if len(self._interferences) > 2 and abs(sum(np.sign(self._load_errors))) == 1:
            # if this is true we must have bound a root

            if self._load_errors[0] * self._load_errors[2] < 0:
                del (self._interferences[1])
                del (self._load_errors[1])
            else:
                del (self._interferences[0])
                del (self._load_errors[0])

        else:  # we have not bound a root, continue with the secant method but del the first item so we progress
            del (self._interferences[0])
            del (self._load_errors[0])

        new_interference = (self._interferences[0] - self._load_errors[0] * (self._interferences[1] -
                                                                             self._interferences[0]) /
                            (self._load_errors[1] - self._load_errors[0]))

        print(f'Adjusting interference, new interference is {new_interference}')

        return new_interference
