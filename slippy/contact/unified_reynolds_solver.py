from slippy.abcs import _ReynoldsSolverABC


class UnifiedReynoldsSolver(_ReynoldsSolverABC):

    def __init__(self):
        # get everything ready to rumble
        pass

    def solve(self, previous_state):
        # rumble
        current_state = dict()
        current_state['interference'] = previous_state['interference']

        return current_state
