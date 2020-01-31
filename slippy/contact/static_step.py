r"""
======================
Static Modelling steps
======================

Steps for modelling static or quasi static situations:
should include:
specified global interference
specified gloabal loading
specified surface loading
specified surface displacement
closure plot generator
pull off test (for testing adhesion models)
friciton limited tangential loading (should work generically with friction models

No models should have to do wear or other time varying stuff.

All should return a current state dict...?
"""
import typing
from collections import namedtuple

import numpy as np
import scipy.optimize as optimize

from slippy.contact import Loads, _ModelStep
from slippy.contact._model_utils import get_gap_from_model, non_dimentional_height
from slippy.contact._step_utils import solve_normal_interference  # , get_next_file_num

__all__ = ['StaticNormalLoad', 'StaticNormalInterference', 'ClosurePlot', 'PullOff', 'SurfaceDisplacement',
           'SurfaceLoading']

data_path = r'D:\slippy_data'

StaticNormalLoadOptions = namedtuple('StaticNormalLoadOptions',
                                     ['maxit_load_loop', 'atol_load_loop',
                                      'rtol_load_loop', 'periodic', 'interpolation_mode', 'max_it_find_contact_nodes',
                                      'material_options'],
                                     defaults=(None,) * 7)

StaticNormalInterferenceOptions = namedtuple('StaticNormalInterferenceOptions',
                                             ['periodic', 'interpolation_mode',
                                              'max_it_find_contact_nodes', 'material_options'],
                                             defaults=(None,) * 4)


class StaticNormalLoad(_ModelStep):
    """
    Static loading between two bodies

    Parameters
    ----------
    step_name: str
        The name of the step
    load_z: float
        Loads in each of the principal directions.
    relative_off_set: tuple, optional (None)
        The relative off set betwee nsurface 1 and surface 2 (relative to the offset at the start of the step) only one
        off set can be specified
    absolute_off_set: tuple, optional (None)
        The absolute off set between surface 1 and surface 2 (off set between the origins of the surfaces)
        only one offset can be specified
    max_it_load_loop: int, optional (100)
        The maximum number of iterations before the loading loop exits. For definitions of the optimisation loops see
        the notes section
    rtol_load_loop: float, optional (1e-3)
        The relative tolerance on the height of the second surface, when this tolerance is reached the loop will
        converge.
    atol_load_loop: float, optional (None)
        The absolute tolerance on the load loop
    interpolation_mode: str {'nearest', 'linear', 'quadratic'}, optional ('nearest')
        The interpolation mode to be used on the second surface, only used if the second surface is not analytical
    periodic: bool, optional (False)
        If True the second surface is considered periodic for the interference detection step, surfaces are always
        considered periodic for the loading step (This is a fundamental assumption of the FFT BEM technique)
    adhesion: bool, optional (True)
        If True the parent model's adhesion model is used when solving the step. Setting to false solves the step
        as if there is no adhesion present. This is necessary to initialise adhesive contacts
    max_it_find_contact_nodes: int, optional (100)
        The maximum number of iterations used while finding the contact nodes
    material_options: dict
        Dict of options to be passed to the loads_from_surface_displacement method of the first surface in the model

    Notes
    -----
    For each iteration of the loading loop the height of the second surface is adjusted, the interference between the
    surfaces found and the loads on the surfaces needed to cause deformation equal to the interference are found.

    The displacement is solved for each iteration of the load loop, this is the step in the computation in which the
    loads on the surface needed to cause a specified displacement are found.
    """
    _load: Loads = None
    _off_set: tuple = (0, 0)
    _abs_off_set: bool = False

    def __init__(self, step_name: str, load_z: float, relative_off_set: tuple = None,
                 absolute_off_set: typing.Optional[tuple] = None, max_it_load_loop: int = 100,
                 rtol_load_loop: float = None, atol_load_loop: float = None, interpolation_mode: str = 'nearest',
                 periodic: bool = False, adhesion: bool = True, max_it_find_contact_nodes: int = 100,
                 material_options: dict = None):

        material_options = {} or material_options
        if rtol_load_loop is None and atol_load_loop is None:
            rtol_load_loop = 1e-3
        self._load = Loads(z=load_z, x=None, y=None)
        self._options = StaticNormalLoadOptions(maxit_load_loop=max_it_load_loop, atol_load_loop=atol_load_loop,
                                                periodic=periodic, interpolation_mode=interpolation_mode,
                                                rtol_load_loop=rtol_load_loop, material_options=material_options,
                                                max_it_find_contact_nodes=max_it_find_contact_nodes)

        self._adhesion = adhesion

        if relative_off_set is None:
            if absolute_off_set is None:
                self._off_set = (0, 0)
                self._abs_off_set = False
            else:
                self._off_set = absolute_off_set
                self._abs_off_set = True
        else:
            if absolute_off_set is not None:
                raise ValueError("Only one mode of off set can be specified, both the absolute and relative off set "
                                 "were given")
            self._off_set = relative_off_set
            self._abs_off_set = False

        super().__init__(step_name)

    def data_check(self, previous_state: set):
        # check that both surfaces are defined, both materials are defined, if there is a tangential load check that
        # there is a friction model defined, print all of this to console, update the previous_state set, delete the
        # previous state?

        current_state = {'off_set', 'loads', 'surface_1_disp', 'surface_2_disp', 'total_disp', 'undeformed_gap',
                         'interference'}  # TODO

        return current_state

    def solve(self, previous_state: dict, output_file):
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
        initial_contact_nodes = previous_state['contact_nodes'] if 'contact_nodes' in previous_state else None

        opt = self._options

        if self._abs_off_set:
            off_set = self._off_set
        else:
            off_set = tuple(current + change for current, change in zip(previous_state['off_set'], self._off_set))

        # noinspection PyTypeChecker
        just_touching_gap, surface_1_points, surface_2_points = get_gap_from_model(self.model, interferance=0,
                                                                                   off_set=off_set,
                                                                                   mode=opt.interpolation_mode,
                                                                                   periodic=opt.periodic)

        current_state = dict(just_touching_gap=just_touching_gap, surface_1_points=surface_1_points,
                             surface_2_points=surface_2_points)

        adhesion_model = self._adhesion if self._adhesion else None

        results = dict()
        it = 0

        surf_1 = self.model.surface_1
        try:
            uz = non_dimentional_height(1, surf_1.material.E, surf_1.material.v, self._load.z, surf_1.grid_spacing,
                                        return_uz=True)
        except AttributeError:
            uz = 1

        print(f'uz = {uz}')

        def opt_func(height):
            nonlocal results, it

            # make height dimensional
            height *= uz
            it += 1

            # noinspection PyTypeChecker
            loads, *disp_tup, contact_nodes = solve_normal_interference(height, gap=just_touching_gap, model=self.model,
                                                                        adhesive_force=adhesion_model,
                                                                        contact_nodes=initial_contact_nodes,
                                                                        max_iter=opt.max_it_find_contact_nodes,
                                                                        material_options=opt.material_options)

            results['loads'] = loads
            results['total_displacement'] = disp_tup[0]
            results['surface_1_displacement'] = disp_tup[1]
            results['surface_2_displacement'] = disp_tup[2]
            results['contact_nodes'] = contact_nodes
            print(f'Iteration {it}:')
            print(f'Interference is: {height}')
            print(f'Percentage of nodes in contact: {sum(contact_nodes.flatten()) / contact_nodes.size}')
            set_load = self._load.z
            print(f'Target load is: {set_load}')

            total_load = np.sum(loads.z.flatten()) * self.model.surface_1.grid_spacing ** 2
            print(f'Total load is: {total_load}')

            return total_load - set_load

        # need to set bounds and pick a sensible starting point
        upper = 5 * max(just_touching_gap.flatten()) / uz

        print(f'upper bound set at: {upper}')
        print(f'Interference tolerance set to {opt.atol_load_loop} A, {opt.rtol_load_loop} R')

        # TODO implement initial guesses

        opt_result = optimize.root_scalar(opt_func, bracket=(0, upper), rtol=opt.rtol_load_loop,
                                          xtol=opt.atol_load_loop, maxiter=opt.maxit_load_loop)

        current_state.update(results)
        current_state['interference'] = opt_result.root * uz
        current_state['gap'] = just_touching_gap - current_state['interference'] + results['total_displacement'].z

        # check out put requests/ sub models
        self.solve_sub_models(current_state)
        self.save_outputs(current_state, output_file)
        return current_state

    def __repr__(self):
        return 'Not yet sorry'

    @classmethod
    def new_step(cls, model):
        pass


class StaticNormalInterference(_ModelStep):
    """
    Static interference between two surfaces, found from point of first touching

    Parameters
    ----------
    step_name: str
        The name of the step, can be any string, used to define output request
    absolute_interference: float, optional (None)
        The absolute interference between the surfaces from the point of first contact, only one type of interference
        can be set
    relative_interference: float, optional (None)
        The realtive interference between the surfaces from the current position, only one type of interference can be
        set
    relative_off_set: tuple, optional (None)
        The relative off set betwee nsurface 1 and surface 2 (relative to the offset at the start of the step) only one
        off set can be specified
    absolute_off_set: tuple, optional (None)
        The absolute off set between surface 1 and surface 2 (off set between the origins of the surfaces)
        only one offset can be specified
    interpolation_mode: str {'nearest', 'linear', 'quadratic'}, optional ('nearest')
        The interpolation mode to be used on the second surface, only used if the second surface is not analytical
    periodic: bool, optional (False)
        If True the second surface is considered periodic for the interference detection step, surfaces are always
        considered periodic for the loading step (This is a fundamental assumption of the FFT BEM technique)
    adhesion: bool, optional (True)
        If True the parent model's adhesion model is used when solving the step. Setting to false solves the step
        as if there is no adhesion present. This is necessary to initialise adhesive contacts
    max_it_find_contact_nodes: int, optional (100)
        The maximum number of iterations used while finding the contact nodes
    material_options: dict
        Dict of options to be passed to the loads_from_surface_displacement method of the first surface in the model

    """
    relative_interference: bool = False
    'Bool true if the interference is relative'
    interference: float = 0.0
    'The specified interference'
    _adhesion: bool = True
    'If false the step is solved with no adhesion'

    def __init__(self, step_name: str, absolute_interference: float = None, relative_interference: float = None,
                 relative_off_set: tuple = None, absolute_off_set: typing.Optional[tuple] = None,
                 interpolation_mode: str = 'nearest', periodic: bool = False, adhesion: bool = True,
                 max_it_find_contact_nodes: int = 100, material_options: dict = None):

        if absolute_interference is not None and relative_interference is not None:
            raise ValueError('Only one type of interference can be set, both the absolute and the relative interference'
                             ' have been set.')
        elif absolute_interference is None and relative_interference is None:
            raise ValueError('Either the relative interference or the absolute interference must be set')

        # noinspection PyTypeChecker
        self.interference = absolute_interference or relative_interference
        self.relative_interference = absolute_interference is None

        material_options = dict() or material_options

        self._adhesion = adhesion

        self._options = StaticNormalInterferenceOptions(periodic=periodic, interpolation_mode=interpolation_mode,
                                                        max_it_find_contact_nodes=max_it_find_contact_nodes,
                                                        material_options=material_options)

        if relative_off_set is None:
            if absolute_off_set is None:
                self._off_set = (0, 0)
                self._abs_off_set = False
            else:
                self._off_set = absolute_off_set
                self._abs_off_set = True
        else:
            if absolute_off_set is not None:
                raise ValueError("Only one mode of off set can be specified, both the absolute and relative off set "
                                 "were given")
            self._off_set = relative_off_set
            self._abs_off_set = False

        super().__init__(step_name)

    def data_check(self, current_state):
        pass

    def solve(self, current_state, output_file):
        height = current_state['interference'] * self.relative_interference + self.interference
        gap, surf_1_pts, surf_2_pts = get_gap_from_model(self.model, interferance=0, off_set=self._off_set,
                                                         mode=self._options.interpolation_mode,
                                                         periodic=self._options.periodic)
        adhesion_model = self.model._adhesion if self._adhesion else None
        initial_contact_nodes = current_state['contact_nodes'] if 'contact_nodes' in current_state else None

        loads, *disp_tup, contact_nodes = solve_normal_interference(height, gap=gap, model=self.model,
                                                                    adhesive_force=adhesion_model,
                                                                    contact_nodes=initial_contact_nodes,
                                                                    max_iter=self._options.max_it_find_contact_nodes,
                                                                    material_options=self._options.material_options)

        current_state['loads'] = loads
        current_state['total_disp'] = disp_tup[0]
        current_state['surf_1_disp'] = disp_tup[1]
        current_state['surf_2_disp'] = disp_tup[2]
        current_state['contact_nodes'] = contact_nodes
        current_state['nd_gap'] = gap
        # check the solution is reasnoble (check not 100% contact) check that the achived load is similar to the
        # actual load, check that the loop converged
        # TODO
        # find the loads on surface 1, 2
        current_state['interference'] = height
        ################################################################################################################
        # if not hasattr(np, 'ITNUM'):
        #    np.ITNUM = int(get_next_file_num(data_path))  #
        #
        #        total_load = np.sum(loads.z.flatten()) * self.model.surface_1.grid_spacing ** 2
        #        pickle.dump({b'nd_gap': nd_gap, b'height': height, b'total_load': total_load, b'interference': height,
        #                     b's1_disp': disp_tup[1].z, b's2_disp': disp_tup[2].z,  b'all_loads': loads.z,
        #                     b'props': {b'E1': self.model.surface_1.material.E, b'v1': self.model.surface_1.material.v,
        #                                b'E2': self.model.surface_2.material.E, b'v2': self.model.surface_2.material.v},
        #                     b'grid': self.model.surface_1.grid_spacing},
        #                    open(f"{data_path}\\{np.ITNUM}.pkl", "wb"))
        #
        #        np.ITNUM += 1
        ################################################################################################################
        # check out put requests, check optional extra stuff that can be truned on?????
        self.solve_sub_models(current_state)
        self.save_outputs(current_state, output_file)
        return current_state

    def __repr__(self):
        return 'Not yet'

    @classmethod
    def new_step(cls, model):
        pass


class ClosurePlot(_ModelStep):
    """
    Generate a closure plot for static contact between two surfaces
    """

    def __init__(self, step_name: str):
        super().__init__(step_name)

    def data_check(self, current_state):
        pass

    def solve(self, current_state, output_file):
        pass

    def __repr__(self):
        pass

    @classmethod
    def new_step(cls, model):
        pass


class PullOff(_ModelStep):
    def __init__(self, step_name: str):
        super().__init__(step_name)

    def data_check(self, current_state):
        pass

    def solve(self, current_state, output_file):
        pass

    def __repr__(self):
        pass

    @classmethod
    def new_step(cls, model):
        pass


class SurfaceLoading(_ModelStep):
    """
    Loading of surface points of a single surface
    """
    _surfaces_required = 1

    def __init__(self, step_name: str):
        super().__init__(step_name)

    def data_check(self, current_state):
        pass

    def solve(self, current_state, output_file):
        pass

    def __repr__(self):
        pass

    @classmethod
    def new_step(cls, model):
        pass


class SurfaceDisplacement(_ModelStep):
    """
    Displacement of surface points of a single surface
    """
    _surfaces_required = 1

    def __init__(self, step_name: str):
        super().__init__(step_name)

    def data_check(self, current_state):
        pass

    def solve(self, current_state, output_file):
        pass

    def __repr__(self):
        pass

    @classmethod
    def new_step(cls, model):
        pass


"""
class _StaticStep(_ModelStep):
     A static contact step

    Parameters
    ----------
    model : ContactModel
        The model that the step will be added to
    step_name : str
        The name of the step
    components : str
        The displacement/ load components to be calculated
    load : {float, dict} optional (None)
        either a float specifying the load between two surfaces or a dict of loads, see notes for desciption of valid
        dict. Either the load or the displacement should be set, not both.
    displacement : {float, dict} optional (None)
        either a float specifying the interference between two surfaces or a dict of loads, see notes for desciption of
        valid dict. Either the load or the displacement should be set, not both.
    simple : bool optional (True)
        A flag which can be set to False if a 'full' analysis is required, otherwise loads only cause displacements in
        the direction of the load. This is normally accurate enough, setting to false may cause convergence problems
    influence_matrix_span : int optinal (None)
        The span in number of grid points of the influence matrix, this defaults to the same size as the first surface
        in the model.

    
    _load = None
    _displacement = None
    options: StaticStepOptions = None

    def __init__(self, model: ContactModel, step_name: str, components: str = 'z', load: {float, dict} = None,
                 displacement: {float, dict} = None, simple: bool = True, influence_matrix_span: int = None,
                 tollerance_load_loop: float = 1e-4, max_iterations_load_loop: int = 100):

        super().__init__(step_name=step_name, model=model)
        self.load = float(load)  # # # ##FIX THIS
        self.displacement = float(displacement)  #### FIX THIS
        self.options = StaticStepOptions(components, influence_matrix_span, simple, max_iterations_load_loop,
                                         tollerance_load_loop)

    @property
    def load(self):
        return self._load

    @load.setter
    def load(self, value):
        if isinstance(value, Number):
            del self.displacement
            self._load = value
            self._surfaces_required = 2
        elif isinstance(value, dict):
            if not set(value.keys()).issubset({'x', 'y', 'z'}):
                raise ValueError("Invalid keys in loads dict, should be 'x', 'y', 'z' only")
            del self.displacement
            self._load = value
            self._surfaces_required = 1
        else:
            raise ValueError(f"Unsupported type for load: {type(value)}, supported types are dict or numeric")

    @load.deleter
    def load(self):
        self._load = None
        self._surfaces_required = None

    @property
    def displacement(self):
        return self._displacement

    @displacement.setter
    def displacement(self, value):
        if isinstance(value, Number):
            del self.load
            self._displacement = value
            self._surfaces_required = 2
        elif isinstance(value, dict):
            if not set(value.keys()).issubset({'x', 'y', 'z'}):
                raise ValueError("Invalid keys in displacements dict, should be 'x', 'y', 'z' only")
            del self.load
            self._displacement = value
            self._surfaces_required = 1
        else:
            raise ValueError(f"Unsupported type for displacement: {type(value)}, supported types are dict or numeric")

    @displacement.deleter
    def displacement(self):
        self._displacement = None
        self._surfaces_required = None

    def _data_check(self):
        # check either load or displacement is set
        if not (self.load is None ^ self.displacement is None):
            raise ValueError(f"Either Loads or displacement must be set for static step {self.name}")
        # check components
        if not set(self.options.components).issubset('xyz'):
            raise ValueError(f"Requested components {self.options.components} in step {self.name} contains invalid "
                             f"components, valid components are 'x', 'y', 'z'.")
        if self._surfaces_required == 1 and not (self._model.surface_1 is None ^ self._model.surface_2 is None):
            raise ValueError(f"Only one surface is required for step {self.name}, but two are set.")
        if self._surfaces_required == 2 and (self._model.surface_1 is None or self._model.surface_2 is None):
            raise ValueError(f"Two surfaces are required for step {self.name}, but at least one of the surfaces is not"
                             " set")
        #

    def _solve(self, previous_state, log_file, output_file):

        if self._surfaces_required == 1:

            if self.displacement is None:  # Must be one body set loads
                displacements = elastic_loading(self.load, self._model.surface_1.grid_spacing,
                                                self._model.surface_1.material.v,
                                                self._model.surface_1.material.G,
                                                self.options.components,
                                                self.options.influence_matrix_span,
                                                self.options.simple)
            else:  # Must be one body set displacement
                surface_loads = elastic_displacement(self.displacement,
                                                     self._model.surface_1.grid_spacing,
                                                     self._model.surface_1.material.G,
                                                     self._model.surface_1.material.v,
                                                     self.options.influence_matrix_span,
                                                     self.options.tolerance_load_loop,
                                                     self.options.simple,
                                                     self.options.max_iterations_load_loop,
                                                     self.options.components)

        elif self._model._is_rigid:
            if self.displacement is None:  # must be load between 2 bodies
                pass  # if one is rigid it must be the second one
            else:  # must be set interfearance between two bodies
                pass

        else:
            if self.displacement is None:  # must be load between 2 bodies
                pass  # if one is rigid it must be the second one
            else:  # must be set interfearance between two bodies
                pass

    def __repr__(self):
        return "StaticStep(model = " + repr(self._model) + ", components = " + self.options.components + \
               f", load = {self.load}, displacement = {self.displacement}, simple = {self.options.simple})"
"""
