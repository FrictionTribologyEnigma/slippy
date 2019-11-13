from .steps import _ModelStep
from .lubricant import LubricantABC


class UnifiedLubricationStep(_ModelStep):
    """A step solved using the unified lubrication solver for mixed lubrication without cavitation



    """
    def __init__(self, step_name: str, lubricant: LubricantABC = None, maxit: int = 100, relax: float = 0.2):
        super().__init__(step_name)

    def _data_check(self, current_state: set):
        pass

    def _solve(self, current_state, output_file):
        pass

    def __repr__(self):
        pass

    @classmethod
    def new_step(cls, model):
        pass


class PayvarSalantStep(_ModelStep):
    """A step solved by the Payvar Salant solver for mixed lubrication with cavitation

    """

    def __init__(self, step_name: str):
        super().__init__(step_name)

    def _data_check(self, current_state: set):
        pass

    def _solve(self, current_state, output_file):
        pass

    def __repr__(self):
        pass

    @classmethod
    def new_step(cls, model):
        pass