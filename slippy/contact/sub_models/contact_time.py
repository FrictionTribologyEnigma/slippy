from slippy.core import _SubModelABC
import numpy as np
from numba import njit
import pickle

__all__ = ["ResultContactTime"]


class ResultContactTime(_SubModelABC):
    def __init__(self, name: str, interpolate_new: bool = True, movement_axis: int = 1):
        """
        Find the contact time for each point in the surface.

        Parameters
        ----------
        name: str
            The name of the sub model, used for debugging
        interpolate_new: bool, optional (True)
            If True points which are brought into contact on the current time step will have their initial contact time
            interpolated. If this is set, the movement_axis must be correct.
        movement_axis: int, optional (0)
            The axis along which the contact moves (1 for x direction)

        Notes
        -----
        Interpolation of new contact points follows these rules:
        First the longest line of new contact nodes in the sliding direction will be found and the direction of sliding
        (+ve/-ve along the axis) will be determined.
        then for each line of new contact nodes:
        If there is an adjacent node which was in contact on the last time step:
            The line is filled starting with the full time step value next to the adjacent node
        Otherwise:
            the line is filled starting with the minimum value at the side opposite to the adjacent contact nodes from
            other lines
        The step in values is decided by the maximum line length.

        This sub model requires:
            'surface_1_points', 'surface_2_points', 'time_step', 'contact_nodes'
        and provides:
            "contact_time_1", "contact_time_2"

        Which are the contact times of the points on the first and second surface respectively
        """
        super().__init__(name, requires={'surface_1_points', 'surface_2_points', 'time_step', 'contact_nodes'},
                         provides={"contact_time_1", "contact_time_2"})
        self._arrays_written = False
        self._interpolate_new = interpolate_new
        self._current_times = dict()
        self._axis = movement_axis

    def solve(self, current_state: dict) -> dict:
        rtn_dict = {}

        for surface_num in [1, 2]:
            surface = self.model.__getattribute__(f"surface_{surface_num}")
            y_real, x_real = surface.convert_coordinates(*current_state[f'surface_{surface_num}_points'])
            x_ind = np.mod(np.array(x_real / surface.grid_spacing + surface.grid_spacing / 2, dtype=np.uint16),
                           surface.max_shape()[1])
            y_ind = np.mod(np.array(y_real / surface.grid_spacing + surface.grid_spacing / 2, dtype=np.uint16),
                           surface.max_shape()[0])
            if not self._arrays_written:
                self._current_times[surface_num] = np.zeros(surface.max_shape())
            sub_view = self._current_times[surface_num][y_ind, x_ind]
            sub_view += current_state['time_step']
            sub_view *= current_state['contact_nodes']
            if self._interpolate_new and self._arrays_written:  # don't attempt on first go
                if not self._axis:
                    worked = _interpolate_and_fill(sub_view.T, current_state['contact_nodes'].T,
                                                   current_state['time_step'])
                else:
                    worked = _interpolate_and_fill(sub_view, current_state['contact_nodes'], current_state['time_step'])
                if not worked:
                    pickle.dump({'time': current_state['time'],
                                 'axis': self._axis,
                                 'sub_view': sub_view,
                                 'cn': current_state['contact_nodes'],
                                 'time_step': current_state['time_step']}, open("save.p", "wb"))
                    raise ValueError(f"Sub model {self.name} could not be solved, this is often cause by the sliding "
                                     f"axis being wrong in the sub model definition")
            rtn_dict[f"contact_time_{surface_num}"] = sub_view.copy()
            self._current_times[surface_num][y_ind, x_ind] = sub_view
        self._arrays_written = True
        return rtn_dict


@njit
def _interpolate_and_fill(current_times: np.ndarray, contact_nodes: np.ndarray, time_step: float):
    first_index = current_times == time_step

    undecided = True
    start = None
    longest_line = 0

    for line, ct, i in zip(first_index, current_times, range(len(first_index))):
        if np.any(line):
            # get the ends of the line
            idx = np.where(line[:-1] != line[1:])[0]
            line_len = idx[-1] - idx[0]
            if undecided:
                if ct[idx[0]]:
                    start = True
                    undecided = False
                elif ct[idx[-1] + 1]:
                    start = False
                    undecided = False
            if line_len > longest_line:
                longest_line = line_len

    if longest_line == 0:
        return True
    if undecided:
        return False

    fill_values = time_step * (np.arange(longest_line) + 1) / longest_line
    if start:
        fill_values = np.flip(fill_values)

    for line, ct, i in zip(first_index, current_times, range(len(first_index))):
        if np.any(line):
            # get the ends of the line
            idx = np.where(line[:-1] != line[1:])[0]
            line_len = idx[-1] - idx[0]
            if start:
                if ct[idx[0]]:
                    ct[idx[0] + 1:idx[-1] + 1] = fill_values[:line_len]
                else:
                    ct[idx[0] + 1:idx[-1] + 1] = fill_values[-line_len:]
            else:
                if ct[idx[-1] + 1]:
                    ct[idx[0] + 1:idx[-1] + 1] = fill_values[-line_len:]
                else:
                    ct[idx[0] + 1:idx[-1] + 1] = fill_values[:line_len]
    current_times *= contact_nodes
    return True
