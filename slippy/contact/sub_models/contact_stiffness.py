import numpy as np

from slippy.abcs import _SubModelABC
from slippy.contact.influence_matrix_utils import plan_convolve, bccg
from slippy.contact.materials import _IMMaterial


class ContactStiffness(_SubModelABC):
    """A sub model for finding contact stiffness of influence matrix based materials

    Parameters
    ----------
    name: str
        The name of the sub model, used for outputs and debugging
    loading: bool, optional (True)
    unloading: bool, optional (True)
    direction: str, optional ('z')
    tol: float, optional (1e-4)
    max_it: int, optional (100)
    """
    requires = {'contact_nodes'}
    provides = set()

    def __init__(self, name: str, loading: bool = True, unloading: bool = True, direction: str = 'z', tol: float = 1e-4,
                 max_it: int = 100):
        super().__init__(name)
        if direction not in ['x', 'y', 'z']:
            raise ValueError('direction should be one of x, y or z')
        self.component = direction * 2

        self.loading = loading
        if loading:
            self.provides.add(f's_contact_stiffness_loading_{direction}')
        self.unloading = unloading
        if unloading:
            self.provides.add(f's_contact_stiffness_unloading_{direction}')
        if not (loading or unloading):
            raise ValueError("No output requested")

        self.tol = tol
        self.max_it = max_it

    def _solve(self, current_state, loading):
        surf_1 = self.model.surface_1
        surf_2 = self.model.surface_2

        if not (isinstance(surf_1.material, _IMMaterial) and isinstance(surf_2.material, _IMMaterial)):
            raise ValueError('Contact stiffness sub model will only work with influence matrix based materials')

        if loading:
            max_pressure = min(surf_1.material.max_load, surf_2.material.max_load)
            contact_nodes = np.logical_and(current_state['contact_nodes'],
                                           current_state['loads'].z < max_pressure * 0.999)
        else:
            contact_nodes = current_state['contact_nodes']

        span = tuple(s * 2 for s in contact_nodes.shape)

        comp = self.component

        im1 = surf_1.material.influence_matrix(span=span, grid_spacing=[surf_1.grid_spacing] * 2,
                                               components=[comp])[comp]
        im2 = surf_2.material.influence_matrix(span=span, grid_spacing=[surf_1.grid_spacing] * 2,
                                               components=[comp])[comp]
        total_im = im1 + im2

        displacement = np.ones(contact_nodes.shape)

        convolution_func = plan_convolve(displacement, total_im, contact_nodes)

        loads_in_domain, failed = bccg(convolution_func, displacement[contact_nodes], self.tol,
                                       self.max_it, x0=displacement[contact_nodes],
                                       min_pressure=0, max_pressure=np.inf)

        return np.sum(loads_in_domain)/contact_nodes.size

    def solve(self, current_state):
        results = dict()
        direction = self.component[0]
        if self.loading:
            results[f's_contact_stiffness_loading_{direction}'] = self._solve(current_state, True)
        if self.unloading:
            results[f's_contact_stiffness_unloading_{direction}'] = self._solve(current_state, False)
        return results
