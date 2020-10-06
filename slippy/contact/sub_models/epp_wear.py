import numpy as np

from slippy.abcs import _SubModelABC


__all__ = ['EPPWear']


class EPPWear(_SubModelABC):
    """
    Parameters
    ----------
    name: str
        The name of the sub model and wear source, used for outputs and error logging
    proportion_surface_1: float
        The proportion of the wear to come from the main surface in the simulation, the total wear will be enough to
        accommodate the remaining interference after elastic deformation has been taken into account.

    Notes
    -----
    This sub model assumes that the grid spacings for the surfaces are the same, if this is not correct wear will be
    assigned incorrectly
    """
    n_calls = 0

    def __init__(self, name: str, proportion_surface_1: float, no_time: bool = True):

        super().__init__(name)
        self.p_surf_1 = proportion_surface_1
        self.requires = {'interference', 'total_displacement', 'just_touching_gap'}
        self.provides = set()
        if proportion_surface_1 > 0:
            self.requires.add('surface_1_points')
        if proportion_surface_1 < 1:
            self.requires.add('surface_2_points')
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
            gap = (just_touching_gap - current_state['interference'] + current_state['total_displacement'].z +
                   self.plastic_def_this_step)

        else:
            # just use the current just touching gap and interference
            just_touching_gap = current_state['just_touching_gap']
            gap = (just_touching_gap - current_state['interference'] + current_state['total_displacement'].z)

        idx = np.logical_and(gap < 0, current_state['contact_nodes'])
        total_wear = -gap[idx]

        if self.no_time:
            self.plastic_def_this_step[idx] += total_wear

        if self.p_surf_1 > 0:
            y_pts = current_state['surface_1_points'][0][idx]
            x_pts = current_state['surface_1_points'][1][idx]
            self.model.surface_1.wear(self.name, x_pts, y_pts, total_wear * self.p_surf_1)
        if self.p_surf_1 < 1:
            y_pts = current_state['surface_2_points'][0][idx]
            x_pts = current_state['surface_2_points'][1][idx]
            self.model.surface_2.wear(self.name, x_pts, y_pts, total_wear * (1 - self.p_surf_1))
        tpd = np.sum(total_wear)*self.model.surface_1.grid_spacing**2
        print(f"SUB MODEL: {self.name}, total deformation: {tpd}")
        current_state['total_plastic_deformation'] = tpd
        return current_state
