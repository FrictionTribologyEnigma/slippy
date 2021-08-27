from abc import ABC, abstractmethod

import slippy.core as core
from numbers import Number
from slippy.contact._step_utils import make_interpolation_func


class _TransientSubModelABC(core._SubModelABC, ABC):
    def __init__(self, name, requires, provides, transient_values, transient_names, interpolation_mode):
        self.updated_dict = dict()
        self.update_funcs = dict()
        for key, value in zip(transient_names, transient_values):
            if isinstance(value, Number):
                self.updated_dict[key] = value
            else:
                self.updated_dict[key] = None
                self.update_funcs[key] = make_interpolation_func(value, interpolation_mode, key)

        super().__init__(name, requires, set(list(provides) + list(transient_names)))

    def update_transience(self, time):
        relative_time = (time - self.model.current_step_start_time) / self.model.current_step.max_time
        for key, value in self.update_funcs.items():
            self.updated_dict[key] = float(self.update_funcs[key](relative_time))

    def solve(self, current_state: dict) -> dict:
        self.update_transience(current_state['time'])
        rtn_dict = self._solve(current_state, **self.updated_dict)
        rtn_dict.update(self.updated_dict)
        return rtn_dict

    @abstractmethod
    def _solve(self, current_state: dict, **kwargs) -> dict:
        pass
