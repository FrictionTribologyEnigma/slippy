import numpy as np

import slippy
if slippy.CUDA:
    import cupy as cp
from slippy.abcs import _SubModelABC  # noqa: E402
from slippy.contact.influence_matrix_utils import plan_convolve, bccg  # noqa: E402
from slippy.contact.materials import _IMMaterial  # noqa: E402


class ResultContactStiffness(_SubModelABC):
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
    max_it: int, optional (None)
        The maximum number of BCCG iterations, None defaults to the size of the problem (contact_nodes.size)
    definition: {'mean lines', 'far points', 'both'}, optional ('mean lines')
        The definition of contact stiffness to use:
        - 'mean lines' the change in average gap height per unit force.
        - 'far points' the approach of points infinitely deep in each half space per unit of force
        - 'both' will find both of the above
    periodic_axes: tuple, optional ((False, False))
        For each True value the corresponding axis will be solved by circular convolution, meaning the result is
        periodic in that direction

    Notes
    -----
    Results are added to current state dict as:
    's_contact_stiffness_unloading_{ml or fp}_{direction}' or
    's_contact_stiffness_loading_{ml or fp}_{direction}'
    For example the normal contact stiffness in the loading direction will be:

    """

    def __init__(self, name: str, loading: bool = True, unloading: bool = True, direction: str = 'z', tol: float = 1e-6,
                 max_it: int = None, definition: str = 'mean lines', periodic_axes=(False, False)):

        if type(definition) is not str:
            raise ValueError(f"Definition of stiffness must be a string received {type(definition)}")

        definition = definition.lower()
        valid_defs = {'mean lines', 'far points', 'both'}
        if definition not in valid_defs:
            raise ValueError(f'Definition not recognised must be one of {valid_defs}, received: {definition}')

        if definition == 'both':
            definition = ['ml', 'fp']
        elif definition == 'mean lines':
            definition = ['ml']
        else:
            definition = ['fp']

        self.definition = definition

        if direction not in ['x', 'y', 'z']:
            raise ValueError('direction should be one of x, y or z')
        self.component = direction * 2

        self.loading = loading
        self.unloading = unloading
        self._periodic_axes = periodic_axes

        provides = set()

        for defin in definition:
            if loading:
                provides.add(f's_contact_stiffness_loading_{defin}_{direction}')
            if unloading:
                provides.add(f's_contact_stiffness_unloading_{defin}_{direction}')

        if not (loading or unloading):
            raise ValueError("No output requested")

        self.tol = tol
        self.max_it = max_it
        self.k_smooth = None
        self.last_converged_result = {True: None, False: None}
        super().__init__(name, {'contact_nodes', 'loads'}, provides)

    def _solve(self, current_state, loading):
        rtn_dict = dict()

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

        span = contact_nodes.shape

        comp = self.component

        im1 = surf_1.material.influence_matrix(span=span, grid_spacing=[surf_1.grid_spacing] * 2,
                                               components=[comp])[comp]
        im2 = surf_2.material.influence_matrix(span=span, grid_spacing=[surf_1.grid_spacing] * 2,
                                               components=[comp])[comp]
        total_im = im1 + im2

        displacement = np.ones(contact_nodes.shape)

        convolution_func = plan_convolve(displacement, total_im, contact_nodes, circular=self._periodic_axes)

        try:
            initial_guess = self.last_converged_result[loading]
        except IndexError:
            initial_guess = None

        if initial_guess is None:
            initial_guess = displacement * (1 / np.sum(total_im.flatten()))

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

        load_str = 'loading' if loading else 'unloading'

        if 'fp' in self.definition:
            rtn_dict[f's_contact_stiffness_{load_str}_fp_'] = k_rough/contact_nodes.size

        if 'ml' in self.definition:
            all_disp = convolution_func(loads_in_domain, ignore_domain=True)
            all_disp[all_disp > 1] = 1
            all_disp[contact_nodes] = 1
            rtn_dict[f's_contact_stiffness_{load_str}_ml_'] = (k_rough / (np.mean(all_disp) - 1)) / contact_nodes.size

        ''' This was another definition based on the difference between the actual contact and a evenly distributed load
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
                '''

        return rtn_dict, failed

    def solve(self, current_state):
        print(f"SUB MODEL: {self.name}")
        results = dict()
        direction = self.component[0]
        if self.loading:
            sl, failed = self._solve(current_state, True)
            print(f"Contact stiffness in loading direction, success: {not failed}, stiffness: {sl}")
            for key, value in sl.items():
                results[key + direction] = value
        if self.unloading:
            su, failed = self._solve(current_state, False)
            print(f"Contact stiffness in unloading direction, success: {not failed}, stiffness: {su}")
            for key, value in su.items():
                results[key + direction] = value
        return results
