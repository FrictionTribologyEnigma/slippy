from models.steps import _ModelStep
from numbers import Number
from models.models import ContactModel
from collections import namedtuple

__all__ = ['StaticStep']

StaticStepOptions = namedtuple('StaticStepOptions', ['components', 'influence_matrix_span', 'simple',
                                                     'max_iterations_load_loop', 'tolerance_load_loop'])


class StaticStep(_ModelStep):
    """ A static contact step

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
        either a float specifying the interferance between two surfaces or a dict of loads, see notes for desciption of
        valid dict. Either the load or the displacement should be set, not both.
    simple : bool optional (True)
        A flag which can be set to False if a 'full' analysis is required, otherwise loads only cause displacements in
        the direction of the load. This is normally accurate enough, setting to false may cause convergence problems
    influence_matrix_span : int optinal (None)
        The span in number of grid points of the influence matrix, this defaults to the same size as the first surface
        in the model.

    """
    _load = None
    _displacement = None
    options: StaticStepOptions = None

    def __init__(self, model: ContactModel, step_name: str, components: str = 'z', load: {float, dict}=None,
                 displacement: {float, dict}=None, simple: bool = True, influence_matrix_span: int = None,
                 tollerance_load_loop: float = 1e-4, max_iterations_load_loop: int = 100):

        super().__init__(step_name=step_name, model=model)
        self.load = float(load) # # # ##FIX THIS
        self.displacement = float(displacement) #### FIX THIS
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

    def _solve(self, current_state, log_file, output_file):

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