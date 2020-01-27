"""
Model steps for lubricated contacts
"""
import typing
from collections import namedtuple, Sequence
from numbers import Number

import numpy as np

from slippy.abcs import _ReynoldsSolverABC
from slippy.contact._model_utils import get_gap_from_model
from ._material_utils import Loads
from ._step_utils import OffSetOptions, solve_normal_loading
from .steps import _ModelStep

IterSemiSystemOptions = namedtuple('IterSemiSystemOptions', ['pressure_it', 'pressure_rtol', 'load_it', 'load_rtol',
                                                             'relaxation_factor'])


class IterSemiSystemLoad(_ModelStep):
    """
    Parameters
    ----------
    step_name: str
        The name of the step, used for outputs
    reynolds_solver: _ReynoldsSolverABC
        A reynolds solver object that will be used to find pressures
    load_z: float
        The normal load on the contact
    relative_off_set: tuple, optional (None)
        The relative off set between the surface origins in the x and y directions, relative to the last step, only one
        off set method can be used
    absolute_off_set: tuple, optional (None)
        The absolute off set between the surface origins in the x and y directions, only one off set method can be used
    max_pressure_it: int, optional (100)
        The maximum number of iterations in the fluid pressure calculation loop
    pressure_tol: float, optional (1e-7)
        The relative tolerance for the fluid pressure calculation loop
    max_interference_it: int, optional (100)
        The maximum number of iterations in the loop that finds the interference between the surfaces
    load_tol: float, optional (1e-7)
        The relative tolerance on the total load (integral of the pressure solution minus the applied load)
    interpolation_mode: str, optional ('nearest')
        The interpolation mode used to find the points on the second surface, only used if an off set is applied between
        the surfaces or if they have incompatible grid_spacings
    periodic: bool, optional (False)
        If true the surfaces are treated as periodic for the off set application, note this has no effect on the
        solver. To apply periodicity to the solver, this should be specified in the material options and the reynolds
        solver object.
    material_options: list, optional (None)
        a list of material options dicts which will be passed to the materials displacement_from_surface_loading method
        the first item in the list will be passed to the first material the second item will be passed to the second
        material. If no options are specified the default for the material is used.
    initial_guess: {callable, 'previous', list}, optional (None)
        The initial guess for the interference, and/or pressure profile between the surfaces, any callable will be
        called with the contact model and the undeformed nd_gap as positional arguments, it must return the interference
        and the pressure profile. 'previous' will use the result(s) from the previous step, if results are not found the
        interference and pressure profile will be set to 0. Can also be a 2 element list, the first element being the
        interference and the second being the pressure profile as an array, if this is the wrong shape zeros wil be
        used.

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
    min_iterations = 70
    "The minimum number of iterations in the reynolds solving loop"
    _dh = 0
    "The base change in height"
    _interferences = list()
    _load_errors = list()

    _reynolds: typing.Optional[_ReynoldsSolverABC] = None
    initial_guess: typing.Optional[typing.Union[typing.Callable, list, str]]

    def __init__(self, step_name: str, reynolds_solver: _ReynoldsSolverABC, load_z: float,
                 surface_1_speed: float, surface_2_speed: float,
                 radius_in_rolling_direction: float, hertzian_half_width: float,
                 relative_off_set: tuple = None, absolute_off_set: typing.Optional[tuple] = None,
                 max_pressure_it: int = 100, pressure_tol: float = 1e-7,
                 max_interference_it: int = 100, load_tol: float = 1e-7,
                 relaxation_factor: float = 0.5,
                 interpolation_mode: str = 'nearest', periodic: bool = False,
                 material_options: list = None,
                 initial_guess: typing.Union[typing.Callable, list, str] = None):

        self.adjust_height_every_step = True
        self.initial_guess = initial_guess

        self.load = load_z
        self.reynolds = reynolds_solver

        self._material_options = [{}, {}] or material_options

        self.rolling_speed = (surface_1_speed + surface_2_speed) / 2
        self.srr = 2 * (surface_1_speed - surface_2_speed) / (surface_1_speed + surface_2_speed)

        if relative_off_set is None:
            if absolute_off_set is None:
                off_set = (0, 0)
                abs_off_set = False
            else:
                off_set = absolute_off_set
                abs_off_set = True
        else:
            if absolute_off_set is not None:
                raise ValueError("Only one mode of off set can be specified, both the absolute and relative off set "
                                 "were given")
            off_set = relative_off_set
            abs_off_set = False

        if relaxation_factor <= 0 or relaxation_factor > 1:
            raise ValueError("Relaxation factor must be greater than 0 and less than or equal to 1")

        self._off_set_options = OffSetOptions(off_set=off_set, abs_off_set=abs_off_set,
                                              periodic=periodic, interpolation_mode=interpolation_mode)

        self._solver_options = IterSemiSystemOptions(pressure_it=max_pressure_it, pressure_rtol=pressure_tol,
                                                     load_it=max_interference_it, load_rtol=load_tol,
                                                     relaxation_factor=relaxation_factor)

        super().__init__(step_name)

    @property
    def reynolds(self):
        return self._reynolds

    @reynolds.setter
    def reynolds(self, value):
        if isinstance(value, _ReynoldsSolverABC):
            self._reynolds = value
        else:
            raise ValueError("Cannot set a non reynolds solver object as the reynolds solver, to use custom solvers"
                             f"first subclass _ReynoldsSolverABC from slippy.abcs, received type was {type(value)}")

    @reynolds.deleter
    def reynolds(self):
        self._reynolds = None

    def data_check(self, previous_state: set):
        # check if there is a lubricant defined for the model
        if self.model.lubricant_model is None:
            print('Error: No lubricant model set for the contact model, lubrication based steps will not solve')
        pass

    def solve(self, previous_state: dict, output_file):
        if self._off_set_options.abs_off_set:
            off_set = self._off_set_options.off_set
        else:
            off_set = tuple(current + change for current, change in zip(previous_state['off_set'],
                                                                        self._off_set_options.off_set))

        just_touching_gap, surf_1_pts, surf_2_pts = get_gap_from_model(self.model, interferance=0,
                                                                       off_set=off_set,
                                                                       mode=self._off_set_options.interpolation_mode,
                                                                       periodic=self._off_set_options.periodic)

        # Sorting out the initial guess:
        initial_guess = self.initial_guess

        if initial_guess is None:
            initial_guess = [0, 0]
        if isinstance(initial_guess, Sequence):
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
        elif isinstance(initial_guess, str) and initial_guess.lower() == 'previous':
            pressure = np.zeros_like(just_touching_gap) if 'pressure' not in previous_state else \
                previous_state['pressure']
            interference = 0.0 if 'interference' not in previous_state else previous_state['interference']
        elif hasattr(initial_guess, '__call__'):
            interference, pressure = initial_guess(self.model, just_touching_gap)
        else:
            raise ValueError('Unsupported type for initial guess')

        previous_state = {'pressure': pressure, 'just_touching_gap': just_touching_gap}

        # we have the interference, and the pressure initial guesses
        # initiate all sub models
        gs = self.model.surface_1.grid_spacing

        if not all(previous_state['pressure'] == 0):
            disp = solve_normal_loading(loads=Loads(z=previous_state['pressure'], x=None, y=None), model=self.model,
                                        deflections='z', material_options=self._material_options)
            previous_state['displacement'] = disp
        else:
            previous_state['displacement'] = (np.zeros_like(just_touching_gap),  # total
                                              np.zeros_like(just_touching_gap),  # surface_1
                                              np.zeros_like(just_touching_gap))  # surface_2
        previous_state = self.model.lubricant_model.solve_sub_models(previous_state)
        # main loops
        it_num = 0
        while True:
            gap_array = just_touching_gap + previous_state['displacement'][0].z - interference

            # solve reynolds
            current_state = self.reynolds.solve(previous_state, gap_array)

            # check for pressure convergence
            change_in_pressures = current_state['pressure'] - previous_state['pressure']
            pressure_relative_error = np.sum(np.abs(change_in_pressures) / current_state['pressure'])
            pressure_converged = pressure_relative_error < self._solver_options.pressure_rtol

            # apply the relaxation factor to the pressure result
            current_state['pressure'] = (previous_state['pressure'] +
                                         self._solver_options.relaxation_factor * change_in_pressures)

            # solve contact geometry
            disp = solve_normal_loading(current_state['pressure'], self.model)
            current_state['displacement'] = disp
            current_state['nd_gap'] = just_touching_gap - current_state['interference'] - disp[0].z

            # solve lubricant sub models
            current_state = self.model.lubricant_model.solve_sub_models(current_state)

            # check for load convergence
            total_load = np.sum(current_state['pressure']) * gs ** 2
            load_relative_error = (total_load / self.load - 1)
            load_converged = abs(load_relative_error) < self._solver_options.load_rtol

            # escape the loop if it converged
            if pressure_converged and load_converged:
                print(f"step {self.name} converged successfully after {it_num} iterations.\n")
                print(f"Converged load is {total_load}, last change in pressure was {pressure_relative_error}")
                break

            # escape the loop it if failed
            if it_num > self._solver_options.pressure_it:  # this logic has changed used to just check the error
                print(f"step {self.name} failed to converge after {it_num} iterations.\n")
                print("Consider increasing the maximum number of iterations or reducing the relaxation factor")
                print(f"Converged load is {total_load}, last change in pressure was {pressure_relative_error}")
                break

            # adjust height for load balance
            if self.adjust_height_every_step:
                # adjust height based on load balance
                new_interference = self.update_interference(it_num, pressure_relative_error, load_relative_error,
                                                            previous_state['interference'],
                                                            np.mean(current_state['nd_gap']),
                                                            np.min(current_state['nd_gap']))
                current_state['interference'] = new_interference
            elif pressure_converged:
                # adjust height based on load balance
                new_interference = self.update_interference(it_num, pressure_relative_error, load_relative_error,
                                                            previous_state['interference'],
                                                            np.mean(current_state['nd_gap']),
                                                            np.min(current_state['nd_gap']))
                current_state['interference'] = new_interference

            it_num += 1
            previous_state = current_state

        self.solve_sub_models(current_state)
        self.save_outputs(current_state, output_file)
        return current_state

    def __repr__(self):
        pass

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
        new_interference

        Notes
        -----
        This method is quite basic at the moment and can definitely be improved, if executed on every loop, the height
        is just updated by a fixed proportion of the minimum nd_gap size

        If updated only when the solver converges, the regula falsi method is used
        """
        if self.adjust_height_every_step:
            if not it_num % 15:
                self._dh = abs(min_gap) * 0.004
            else:
                self._dh *= 0.9
            return current_interference + self._dh * -np.sign(load_error_rel)  # if the error is negative-> more overlap

        # else:  # height only adjusted when the load has converged, in this case use the Regula-Falsi method
        self._interferences.append(current_interference)
        self._load_errors.append(load_error_rel)

        if len(self._interferences) == 1:  # if this is the first guess give a value that will probably bound it
            return current_interference + abs(mean_gap) * -np.sign(load_error_rel)

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

        return (self._interferences[0] - self._load_errors[0] * (self._interferences[1] - self._interferences[0]) /
                (self._load_errors[1] - self._load_errors[0]))


    @classmethod
    def new_step(cls, model):
        pass
