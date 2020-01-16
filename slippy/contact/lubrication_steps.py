"""
Model steps for lubricated contacts
"""
import typing
import numpy as np
from collections import namedtuple
from slippy.abcs import _ReynoldsSolverABC
from slippy.contact._model_utils import get_gap_from_model
from .steps import _ModelStep
from ._step_utils import OffSetOptions


IterSemiSystemOptions = namedtuple('IterSemiSystemOptions', ['pressure_it', 'pressure_rtol', 'load_it', 'load_rtol'],
                                   defaults=[None, ] * 2)


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
    initial_guess: {callable, 'previous', list}, optional (None)
        The initial guess for the interference, and/or pressure profile between the surfaces, any callable will be
        called with the contact model and the undeformed gap as positional arguments, it must return the interference
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
    _reynolds: typing.Optional[_ReynoldsSolverABC] = None
    initial_guess: typing.Optional[typing.Union[typing.Callable, list, str]]

    def __init__(self, step_name: str, reynolds_solver: _ReynoldsSolverABC, load_z: float,
                 relative_off_set: tuple = None, absolute_off_set: typing.Optional[tuple] = None,
                 max_pressure_it: int = 100, pressure_tol: float = 1e-7,
                 max_interference_it: int = 100, load_tol: float = 1e-7,
                 interpolation_mode: str = 'nearest', periodic: bool = False,
                 material_options: dict = None,
                 initial_guess: typing.Union[typing.Callable, list, str] = None):

        self.initial_guess = initial_guess

        self.load = load_z
        self.reynolds = reynolds_solver

        self._material_options = {} or material_options

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

        self._off_set_options = OffSetOptions(off_set=off_set, abs_off_set=abs_off_set,
                                              periodic=periodic, interpolation_mode=interpolation_mode)
        self._solver_options = IterSemiSystemOptions(pressure_it=max_pressure_it, pressure_rtol=pressure_tol,
                                                     load_it=max_interference_it, load_rtol=load_tol)

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

    def _data_check(self, previous_state: set):
        # check if there is a lubricant defined for the model
        if self.model.lubricant_model is None:
            print('Error: No lubricant model set for the contact model, lubrication based steps will not solve')
        pass

    def _solve(self, previous_state: dict, output_file):
        if self._off_set_options.abs_off_set:
            off_set = self._off_set_options.off_set
        else:
            off_set = tuple(current + change for current, change in zip(previous_state['off_set'],
                                                                        self._off_set_options.off_set))
        # TODO carry on from here
        if self.initial_guess is None:
            interferance = 0
            undeformed_gap, surf_1_pts, surf_2_pts = get_gap_from_model(self.model, interferance=0,
                                                                        off_set=off_set,
                                                                        mode=self._off_set_options.interpolation_mode,
                                                                        periodic=self._off_set_options.periodic)
        if 'pressure' not in previous_state:
            previous_state['pressure'] = np.zeros_like(gap)
        if 'temperature' not in previous_state:
            previous_state['temperature'] = np.zeros_like(gap)
        while load != set_load:
            while not_converged:
                # solve lubricant models
                current_state = self.model.lubricant.solve_pre_models(previous_state)
                # solve reynolds
                current_state = self.reynolds_solver.solve(previous_state, current_state)
                # solve post lubricant models
                current_state = self.model.lubricant.solve_post_models(current_state)
                # solve contact geometry
                disp, deformed_gap = solve_normal_pressure(undeformed_gap, )
        self.solve_sub_models(current_state)
        self.save_outputs(current_state, output_file)
        return current_state

    def __repr__(self):
        pass

    @classmethod
    def new_step(cls, model):
        pass
