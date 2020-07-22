import typing

__all__ = ['OutputRequest']


class OutputRequest:
    """An output request for a multi step contact model

    Parameters
    ----------
    name: str
        The name of the output request
    parameter: str
        The name of the parameter to be saved
    slices: tuple[slice], optional (None)
        A tuple of slice objects used to slice the parameter before saving, for example, to save the first row of an
        array use slices = (slice(0,1), slice(None)). If not set the object will not be sliced, should not be used for
        scalar values
    sub_steps: tuple[int], optional (None)
        A tuple of integers, only used if a multi step is used, the sub_steps for which this output should be active.
        If not set it will be active for all sub steps.

    """
    name: str
    parameter: str
    slices: typing.Optional[tuple]
    sub_steps: typing.Optional[tuple]

    def __init__(self, name: str, parameter: str, slices: tuple = None, sub_steps: tuple = None):
        assert isinstance(name, str), f"Output request name must be a string received {type(name)}"
        self.name = name
        assert isinstance(parameter, str), f"Output request parameter must be a string received {type(parameter)}"
        assert parameter.isidentifier(), "Output request parameter must be a valid variable name"
        self.parameter = parameter
        if slices is not None:
            for this_slice in slices:
                assert isinstance(this_slice, slice)
            self.slices = tuple(slices)
        else:
            self.slices = None
        self.sub_steps = sub_steps
