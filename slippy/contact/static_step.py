r"""
======================
Static Modelling steps
======================

Steps for modelling static or quasi static situations:
should include:
specified global interference
specified global loading
specified surface loading
specified surface displacement
closure plot generator/ adhesive pull of tester (the same but backwards)

No models should have to do wear or other time varying stuff.

All should return a current state dict
"""

import numpy as np
import scipy.optimize as optimize

from ._model_utils import get_gap_from_model
from ._step_utils import HeightOptimisationFunction
from .steps import _ModelStep

__all__ = ['StaticStep']


class StaticStep(_ModelStep):
    """
    Static loading between two bodies

    Parameters
    ----------
    step_name: str
        An identifying name for the step used for errors and outputs
    time_period: float, optional (1.0)
        The total time period of this model step, used for solving sub-models and writing outputs
    off_set_x, off_set_y: float
        The off set between the surfaces origins, in the same units as the grid spacings of the surfaces.
    normal_load, interference: float
        The total compressive load and the interference between the two surfaces (measured from the point of first
        contact). Exactly one of these must be set.
    relative_loading: bool, optional (False)
        If True the load or displacement will be applied relative to the value at the start of the step,
        otherwise the absolute value will be used. eg, if the previous step ended with a load of 10N and this step ramps
        from 0 to 10N setting relative_loading to True will ramp the total load form 10 to 20N over this step.
    adhesion: bool, optional (True)
        If True the adhesion model set for the contact model will be used, If set to false this step will ignore the
        adhesion model (typically used for loading steps)
    unloading: bool, optional (False)
        If True the contact nodes will be constrained to be a sub set of those found in the previous time step.
    profile_interpolation_mode: {'nearest', 'linear'}, optional ('nearest')
        Used to generate the grid points for the second surface at the location of the grid points for the first
        surface, nearest ensures compatibility with sub models which change the profile, if the grid spacings of the
        surfaces match
    periodic_geometry: bool, optional (False)
        If True the surface profile will warp when applying the off set between the surfaces
    periodic_axes: tuple, optional ((False, False))
        For each True value the corresponding axis will be solved by circular convolution, meaning the result is
        periodic in that direction
    max_it_interference: int, optional (100)
        The maximum number of iterations used to find the interference between the surfaces, only used if
        a total normal load is specified (Not used if contact is displacement controlled)
    rtol_interference: float, optional (1e-3)
        The relative tolerance on the load used as the convergence criteria for the interference optimisation loop, only
        used if a total normal load is specified (Not used if contact is displacement controlled)
    max_it_displacement: int, optional (100)
        The maximum number of iterations used to find the surface pressures from the interference, used for all IM
        based materials
    rtol_displacement: float, optional (1e-4)
        The norm of the residual used to declare convergence of the bccg iterations

    Examples
    --------

    """

    def __init__(self, step_name: str, time_period: float = 1.0,
                 off_set_x: float = 0.0, off_set_y: float = 0.0,
                 normal_load: float = None, interference: float = None,
                 relative_loading: bool = False, adhesion: bool = True,
                 unloading: bool = False, profile_interpolation_mode: str = 'nearest',
                 periodic_geometry: bool = False, periodic_axes: tuple = (False, False),
                 max_it_interference: int = 100, rtol_interference=1e-3,
                 max_it_displacement: int = None, rtol_displacement=1e-4):

        self._off_set = (off_set_x, off_set_y)
        self._relative_loading = bool(relative_loading)
        self.profile_interpolation_mode = profile_interpolation_mode
        self._periodic_profile = periodic_geometry
        self._periodic_axes = periodic_axes
        self._max_it_interference = max_it_interference
        self._rtol_interference = rtol_interference
        self._max_it_displacement = max_it_displacement
        self._rtol_displacement = rtol_displacement
        self._height_optimisation_func = None
        self._adhesion = adhesion
        self._unloading = unloading

        if normal_load is not None and interference is not None:
            raise ValueError("Both normal_load and interference are set, only one of these can be set")
        if normal_load is None and interference is None:
            if relative_loading:
                interference = 0
            else:
                raise ValueError("Cannot have no set load or interference and not relative loading, set either the"
                                 "normal load, normal interference or change relative_loading to True")

        self.interference = interference
        # noinspection PyTypeChecker
        self.normal_load = normal_load

        self.load_controlled = interference is None

        provides = {'off_set', 'loads', 'surface_1_displacement', 'surface_2_displacement', 'total_displacement',
                    'interference', 'just_touching_gap', 'surface_1_points', 'contact_nodes', 'total_normal_load',
                    'surface_2_points', 'time', 'time_step', 'new_step', 'converged', 'gap'}

        super().__init__(step_name, time_period, provides)

    def data_check(self, previous_state: set):
        # check that both surfaces are defined, both materials are defined, if there is a tangential load check that
        # there is a friction model defined, print all of this to console, update the previous_state set, delete the
        # previous state?
        current_state = self.provides
        return current_state

    def solve(self, previous_state: dict, output_file) -> dict:
        """
        Solve this model step

        Parameters
        ----------
        previous_state: dict
            The previous state of the model
        output_file: file buffer
            The file in which the outputs will be written

        Returns
        -------
        current_state: dict
            The current state of the model

        """
        # just in case the displacement finder in a scipy optimise block should be a continuous function, no special
        # treatment required
        for s in self.sub_models:
            s.no_time = False

        if self._unloading and 'contact_nodes' in previous_state:
            initial_contact_nodes = previous_state['contact_nodes']
        else:
            initial_contact_nodes = None

        # noinspection PyTypeChecker
        just_touching_gap, surface_1_points, surface_2_points = get_gap_from_model(self.model, interference=0,
                                                                                   off_set=self._off_set,
                                                                                   mode=self.profile_interpolation_mode,
                                                                                   periodic=self._periodic_profile)

        current_state = {'just_touching_gap': just_touching_gap, 'surface_1_points': surface_1_points,
                         'surface_2_points': surface_2_points, 'time': previous_state['time'] + self.max_time,
                         'time_step': self.max_time, 'new_step': True, 'off_set': self._off_set}

        adhesion_model = self.model.adhesion if self._adhesion else None
        # for some reason the type checker messes up here, types are actually correct
        # noinspection PyTypeChecker
        max_load = self.normal_load if self.load_controlled else 1.0
        opt_func = HeightOptimisationFunction(just_touching_gap=just_touching_gap, model=self.model,
                                              adhesion_model=adhesion_model,
                                              initial_contact_nodes=initial_contact_nodes,
                                              max_it_inner=self._max_it_displacement, tol_inner=self._rtol_displacement,
                                              max_set_load=max_load,
                                              tolerance=self._rtol_interference, material_options=None)

        if self._unloading and 'contact_nodes' in current_state:
            contact_nodes = current_state['contact_nodes']
        else:
            contact_nodes = None

        if self.load_controlled:
            if self._relative_loading:
                load = previous_state['total_normal_load'] + self.normal_load
            else:
                load = self.normal_load
            upper = 3 * max(just_touching_gap.flatten())
            print(f'upper bound set at: {upper}')
            print(f'Interference tolerance set to {self._rtol_displacement} Relative')
            opt_func.change_load(load, contact_nodes)
            _ = optimize.root_scalar(opt_func, bracket=(0, upper), rtol=self._rtol_interference,
                                     maxiter=self._max_it_interference, args=(current_state,))
            converged = (np.abs(opt_func.results['total_normal_load']-self.normal_load) /
                         self.normal_load < 0.05 and not opt_func.last_call_failed)
        else:
            if self._relative_loading:
                interference = previous_state['interference'] + self.interference
            else:
                interference = self.interference
            opt_func.change_load(1, contact_nodes)
            _ = opt_func(interference, current_state)
            converged = not opt_func.last_call_failed

        current_state.update(opt_func.results)
        current_state['converged'] = converged
        current_state['gap'] = (just_touching_gap - current_state['interference'] +
                                opt_func.results['total_displacement'].z)
        self.solve_sub_models(current_state)
        self.save_outputs(current_state, output_file)

        return current_state

    def __repr__(self):
        string = (f'StaticStep({self.name}, time_period={self.max_time},'
                  f'off_set_x={self._off_set[0]}, off_set_y={self._off_set[1]},'
                  f'normal_load={self.normal_load}, interference={self.interference}'
                  f'relative_loading:={self._relative_loading}, adhesion={self._adhesion},'
                  f'unloading={self._unloading}, profile_interpolation_mode={self.profile_interpolation_mode},'
                  f'periodic_geometry={self._periodic_profile}, periodic_axes{self._periodic_axes},'
                  f'max_it_interference={self._max_it_interference}, rtol_interference:{self._rtol_interference},'
                  f'max_it_displacement={self._max_it_displacement}, rtol_displacement={self._rtol_displacement})')
        return string
