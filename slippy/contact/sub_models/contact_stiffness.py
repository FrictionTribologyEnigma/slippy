import numpy as np

import slippy
if slippy.CUDA:
    import cupy as cp
from slippy.abcs import _SubModelABC  # noqa: E402
from slippy.contact.influence_matrix_utils import plan_convolve, bccg  # noqa: E402
from slippy.contact.materials import _IMMaterial  # noqa: E402


class ContactStiffness(_SubModelABC):
    """A sub model for finding contact stiffness of influence matrix based materials

    Parameters
    ----------
    name: str
        The name of the sub model, used for outputs and debugging
    loading: bool, optional (True)
        If True the contact stiffness will be found in the loading direction
    unloading: bool, optional (True)
        If True the contact stiffness will be found in the unloading direction
    direction: str, optional ('z')
        The component of contact stiffness to find, note that 'mean lines' definition is only available for the
        z (normal) component
    tol: float, optional (1e-6)
        Tolerance to us for the BCCG iterations
    max_it: int, optional (300)
        The maximum number of BCCG iterations
    definition: {'mean lines', 'far points', 'far points minus uniform'}, optional ('mean lines')
        The definition of contact stiffness to use:
        - 'mean lines' the change in average gap height per unit force.
        - 'far points' the approach of points infinitely deep in each half space per unit of force
        - 'far points minus uniform' as above minus the approach for the same force applied uniformly over the region

    Returns
    -------
    Results are added to current state dict as:
    's_contact_stiffness_unloading_{direction}' or
    's_contact_stiffness_loading_{direction}'
    For example the normal contact stiffness in the loading direction will be
    """
    requires = {'contact_nodes'}
    provides = set()

    def __init__(self, name: str, loading: bool = True, unloading: bool = True, direction: str = 'z', tol: float = 1e-6,
                 max_it: int = 300, definition: str = 'mean lines'):
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

        valid_defs = {'mean lines', 'far points', 'far points minus uniform'}
        if definition not in valid_defs:
            raise ValueError(f'Definition not recognised must be one of {valid_defs}, received: {definition}')

        self.definition = definition
        self.tol = tol
        self.max_it = max_it
        self.k_smooth = None
        self.last_converged_result = {True: None, False: None}

    def _solve(self, current_state, loading):
        surf_1 = self.model.surface_1
        surf_2 = self.model.surface_2

        if not (isinstance(surf_1.material, _IMMaterial) and isinstance(surf_2.material, _IMMaterial)):
            raise ValueError('Contact stiffness sub model will only work with influence matrix based materials')

        if loading:
            max_pressure = min(surf_1.material.max_load, surf_2.material.max_load)
            contact_nodes = np.logical_and(current_state['contact_nodes'],
                                           current_state['loads'].z < max_pressure * 0.99999)
            p_contact = np.mean(current_state['contact_nodes'])
            p_plastic = np.mean(current_state['loads'].z > max_pressure * 0.99999)
            print("Percentage contact nodes:", p_contact)
            print("Percentage plastic nodes:", p_plastic)
            print("Percentage of contact plastic:", p_plastic/p_contact)
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

        try:
            initial_guess = self.last_converged_result[loading]
        except IndexError:
            initial_guess = None

        if initial_guess is None:
            initial_guess = displacement

        loads_in_domain, failed = bccg(convolution_func, displacement[contact_nodes], self.tol,
                                       self.max_it, x0=initial_guess[contact_nodes],
                                       min_pressure=0, max_pressure=np.inf)

        if slippy.CUDA:
            loads_in_domain = cp.asnumpy(loads_in_domain)

        if not failed:
            full_result = np.zeros_like(displacement)
            full_result[contact_nodes] = loads_in_domain
            self.last_converged_result[loading] = full_result

        k_rough = float(np.sum(loads_in_domain))

        if self.definition == 'far points':
            return k_rough/contact_nodes.size, failed

        if self.definition == 'mean lines':
            k_rough / (k_rough / contact_nodes.size - 1) / contact_nodes.size, failed

        if self.k_smooth is None:
            convolution_func = plan_convolve(displacement, total_im, np.ones_like(contact_nodes))
            loads_in_domain, failed = bccg(convolution_func, displacement.flatten(), self.tol,
                                           self.max_it, x0=displacement.flatten(),
                                           min_pressure=0, max_pressure=np.inf)
            self.k_smooth = float(np.sum(loads_in_domain))

        # (below) The deflection a smooth surface would have under the same total load as the rough surface
        delta_s = k_rough/self.k_smooth  # works because we used 1 as the deflection grid_spacing cancels

        contact_stiffness = k_rough/(1-delta_s)  # 1 is the deflection of the rough surface we solved for above

        cs_normalised = contact_stiffness/contact_nodes.size  # again grid spacings cancel here

        return cs_normalised, failed

    def solve(self, current_state):
        print(f"SUB MODEL: {self.name}")
        results = dict()
        direction = self.component[0]
        if self.loading:
            sl, failed = self._solve(current_state, True)
            print(f"Contact stiffness in loading direction, success: {not failed}, stiffness: {sl:.4}")
            results[f's_contact_stiffness_loading_{direction}'] = sl
        if self.unloading:
            su, failed = self._solve(current_state, False)
            print(f"Contact stiffness in unloading direction, success: {not failed}, stiffness: {su:.4}")
            results[f's_contact_stiffness_unloading_{direction}'] = su
        return results
