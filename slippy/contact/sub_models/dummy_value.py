from slippy.core import _SubModelABC


class DummyValue(_SubModelABC):
    def __init__(self, dummy_dict: dict):
        self.dummy_dict = dummy_dict
        super().__init__('Dummy', set(), set(dummy_dict.keys()))

    def solve(self, current_state: dict) -> dict:
        return self.dummy_dict
