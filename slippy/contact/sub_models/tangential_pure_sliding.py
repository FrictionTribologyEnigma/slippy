import typing
import numpy as np
from slippy.core import _SubModelABC

__all__ = ['TangentialPureSliding']


class TangentialPureSliding(_SubModelABC):
    """ Fill forces and displacements due to pure sliding

    Parameters
    ----------
    name: str
        The name of the sub model, used for debugging
    direction: {str, tuple}, optional ('x')
        The direction of the sliding motion, either 'x' or 'y' or a vector defining a direction, 'x' is equivalent
        to: (0, 1)

    Notes
    -----
    This sub model fills the 'loads' result with loads due to pure sliding in the specified direction

    This requires the normal component of the loads to be found

    """

    def __init__(self, name: str, direction: typing.Union[str, typing.Tuple[float]] = 'x'):
        requires = {'maximum_tangential_force'}
        provides = {'loads_x', 'loads_y'}
        if isinstance(direction, str):
            direction = direction.lower()
            if direction == 'x':
                direction = (0, 1)
            elif direction == 'y':
                direction = (1, 0)
            else:
                raise ValueError("Direction can only be 'x' or 'y' or vector defining a direction")
        try:
            length = len(direction)
        except TypeError:
            raise TypeError(f"direction not recognised, must be two element sequence or 'x' or 'y', "
                            f"received {type(direction)}")
        if length != 2:
            raise ValueError("Direction vector must be 2 element sequence")

        direction = np.array(direction)
        self.direction = direction / np.linalg.norm(direction)
        super().__init__(name, requires, provides)

    def solve(self, current_state: dict) -> dict:
        lf = current_state['maximum_tangential_force']
        return {'loads_x': self.direction[1] * lf,
                'loads_y': self.direction[0] * lf}
