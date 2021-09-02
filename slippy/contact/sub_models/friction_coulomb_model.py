from slippy.core import _SubModelABC

__all__ = ['FrictionCoulombSimple']


class FrictionCoulombSimple(_SubModelABC):

    def __init__(self, name: str, coefficient):
        """ Simple coulomb friction, limiting friction is normal force multiplied by a coefficient for each point

        Parameters
        ----------
        name: str
            The name of the model, used for debugging
        coefficient: float
            The value of the friction coefficient, must be grater than 0

        Notes
        -----
        This sub model finds the limiting friction force at each point on the surface. To apply the load a tangential
        model describing how much of the contact is sliding should also be added.

        Provides:
        * 'maximum_tangential_force': The maximum allowable tangential force at each point on the surface, aligned with
            The points which are in contact, described by surface_1_points and surface_2_points
        """
        requires = {'loads_z'}
        provides = {'maximum_tangential_force', 'coulomb_coefficient'}
        super().__init__(name, requires, provides)
        self.coefficient = coefficient

    def solve(self, current_state):
        return {'maximum_tangential_force': current_state['loads_z']*self.coefficient,
                'coulomb_coefficient': self.coefficient}
