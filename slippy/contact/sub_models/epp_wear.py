import warnings
import numpy as np

import slippy
from slippy.core import _SubModelABC


__all__ = ['WearElasticPerfectlyPlastic']


class WearElasticPerfectlyPlastic(_SubModelABC):
    r"""
    Remove overlap between surfaces left after contact with a maximum load

    Parameters
    ----------
    name: str
        The name of the sub model and wear source, used for outputs and error logging
    proportion_surface_1: float
        The proportion of the overlap to come be worn from the main surface in the simulation.
    proportion_surface_2: float, optional (None)
        The proportion of the overlap to be removed from surface 2, this defaults to the remaining overlap, after
        subtracting the proportion from the first surface
    no_time: bool, optional (False)
        Must be set to True if the step which this sub-model is added to is solved with no time dependence, otherwise
        the full wear will be applied for each time step leading to unphysical results

    Notes
    -----
    This sub model assumes that the grid spacings for the surfaces are the same, if this is not correct wear will be
    assigned incorrectly

    For this model to work there must be some overlap between the surfaces at the end of the model step, otherwise no
    wear will be applied.

    Provides:

    * 'total_plastic_deformation': The total material removed for this time step
    * 'wear_plastic_surface_1': The wear applied to each point of surface 1, applied at the points 'surface_1_points',
      only provided if proportion_surface_1 is greater than 0
    * 'wear_plastic_surface_2': The wear applied to each point of surface 2, applied at the points 'surface_2_points',
      only provided if proportion_surface_1 is less than 1


    For simulations with movement between the surfaces wear_plastic_surface_1 and 2 can be confusing as they are aligned
    with the points specified is surface_1_points, not the base coordinates of surface 1 or 2. For examining the output
    from this wear model over time it may be more simple to request the following output:
    'surface_1.wear_volumes['this_model_name']'
    This will always contain the cumulative wear from this model in the base coordinates of the surface.
    """
    n_calls = 0

    def __init__(self, name: str, proportion_surface_1: float, proportion_surface_2: float = None,
                 no_time: bool = False):
        requires = {'interference', 'total_displacement_z', 'just_touching_gap'}
        provides = {'total_plastic_deformation'}
        super().__init__(name, requires, provides)

        if proportion_surface_1 > 1 or proportion_surface_1 < 0:
            raise ValueError("Proportion of wear applied to surface 1 should be between 0 and 1")
        if proportion_surface_2 is not None:
            if proportion_surface_2 > 1 or proportion_surface_2 < 0:
                raise ValueError("Proportion of wear applied to surface 2 should be between 0 and 1")
            if (proportion_surface_1+proportion_surface_2) > 1.00000001:
                warnings.warn("Proportion surface 1 + proportion surface 2 is greater than 1, more wear will be applied"
                              " than the overlap between the surfaces, this will result in unphysical results")

        self.p_surf_1 = proportion_surface_1
        self.p_surf_2 = proportion_surface_2
        if proportion_surface_1 > 0:
            self.requires.add('surface_1_points')
            self.provides.add('wear_plastic_surface_1')
        if proportion_surface_2 > 0:
            self.requires.add('surface_2_points')
            self.provides.add('wear_plastic_surface_2')
        self.plastic_def_this_step = None
        self.no_time = no_time

    def solve(self, current_state: dict) -> dict:
        if 'converged' in current_state and not current_state['converged']:
            print(f"SUB MODEL: {self.name}, Solution did not converge, no wear")
            return current_state

        if self.no_time:

            just_touching_gap = current_state['just_touching_gap']

            if self.plastic_def_this_step is None or ('new_step' in current_state and current_state['new_step']):
                self.plastic_def_this_step = np.zeros_like(just_touching_gap)
            # need to sort out the discrepancy between the current just touching gap and the one used for the model
            gap = (just_touching_gap - current_state['interference'] + current_state['total_displacement_z'] +
                   self.plastic_def_this_step)

        else:
            # just use the current just touching gap and interference
            gap = slippy.asnumpy(current_state['gap'])
            # just_touching_gap = current_state['just_touching_gap']
            # gap = (just_touching_gap - current_state['interference'] + current_state['total_displacement_z'])

        max_load = min(self.model.surface_1.material.max_load,
                       self.model.surface_2.material.max_load)
        idx = np.logical_and(current_state['loads_z'] >= max_load,
                             np.logical_and(gap < 0, current_state['contact_nodes']))
        total_wear = -gap[idx]

        if self.no_time:
            self.plastic_def_this_step[idx] += total_wear
        if total_wear.size:
            tpd = np.sum(total_wear) * self.model.surface_1.grid_spacing ** 2 * (self.p_surf_1+self.p_surf_2)
        else:
            tpd = np.array(0.0)
        results = {'total_plastic_deformation': tpd}
        if self.p_surf_1 > 0:
            y_pts = current_state['surface_1_points'][0][idx]
            x_pts = current_state['surface_1_points'][1][idx]
            surface_1_wear = total_wear * self.p_surf_1
            results['wear_plastic_surface_1'] = surface_1_wear
            self.model.surface_1.wear(self.name, x_pts, y_pts, surface_1_wear)
        if self.p_surf_2 > 0:
            y_pts = current_state['surface_2_points'][0][idx]
            x_pts = current_state['surface_2_points'][1][idx]
            surface_2_wear = total_wear * self.p_surf_2
            results['wear_plastic_surface_2'] = surface_2_wear
            self.model.surface_2.wear(self.name, x_pts, y_pts, surface_2_wear)

        print(f"SUB MODEL: {self.name}, total deformation: {tpd}")

        return results
