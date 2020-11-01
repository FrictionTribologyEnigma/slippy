import warnings
import typing
import zipfile
import tinydb
import slippy
import os
import numpy as np
if slippy.CUDA:
    import cupy as cp

__all__ = ['OutputRequest', 'OutputSaver', 'OutputReader', 'read_output']


class OutputSaver:
    """
    A context manager for saving model output

    Parameters
    ----------
    model_name: str
        The name of the model, to be used for the file names of the outputs

    Notes
    -----
    This writes the outputs to a database file (*.sdb). The arrays cannot be stored in the database and are instead
    stored in binary in a zip repository (*.sar). The entries in the data base are replaced with the file names in the
    zip repository, the shape of the output and the data type.

    See Also
    --------
    OutputReader - A class for reading output files back in
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.in_context = False
        self.database_file = None
        self.array_file = None
        self.array_number = 0

    def write(self, output: dict):
        """Write outputs to file,

        Parameters
        ----------
        output: dict
            The outputs to be added to the database

        """
        if not self.in_context:
            raise ValueError("Output saver is not in context cannot save")
        clean_dict = dict()
        for key, value in output.items():
            if slippy.CUDA and isinstance(value, cp.ndarray):
                value = cp.asnumpy(value)
            if isinstance(value, np.ndarray) and value.size == 1:
                try:
                    value = float(value)
                except TypeError:
                    pass
            if isinstance(value, np.ndarray):
                name = 'a' + str(self.array_number)
                if value.flags.f_contiguous:
                    value = np.ascontiguousarray(value)
                self.array_file.writestr(name, value.tobytes())
                value = '**array**#' + str(self.array_number) + '#' + str(value.shape) + '#' + str(value.dtype)
                self.array_number += 1
            clean_dict[key] = value
        self.data_file.insert(clean_dict)

    def __enter__(self):
        db_filename = os.path.join(slippy.OUTPUT_DIR, self.model_name + '.sdb')
        if os.path.exists(db_filename):
            os.remove(db_filename)
        self.data_file = tinydb.TinyDB(db_filename, 'w')
        self.array_file = zipfile.ZipFile(os.path.join(slippy.OUTPUT_DIR, self.model_name + '.sar'), 'w')
        self.in_context = True
        return self

    def __exit__(self, err_type, value, traceback):
        self.data_file.close()
        self.array_file.close()
        self.in_context = False


class OutputRequest:
    parameters: typing.Sequence[str]

    def __init__(self, name: str, parameters: typing.Sequence[str], timing_mode: str = 'interval',
                 value: typing.Union[typing.Sequence, float] = 1):
        """An output request for a multi step contact model

        Parameters
        ----------
        name: str
            The name of the output request, used for debugging
        parameters: list[str]
            A list of parameters to be saved as part of the output, check step and sub-model descriptions for valid
            names, alternatively ['all'] will save the entire model state
        timing_mode: {'interval', 'time_interval', 'n_outputs_per_step', 'step_times', 'global_times'},
        optional ('interval')
            The timing mode used to control when this output is written to file, the value parameter controls how each
            method is executed:
            - 'interval' the output is written every n time points during execution, using this with value set to 1
               (as is the default behaviour) writes the output on every sub step / time point.
            - 'time_interval' the output is written every n units of time, for example if value is set to 0.25 a step
              which is 1 unit of time long will output 4 times [t = 0.25, 0.5, 0.75, 1]. This is reset at the
              beginning of each model step.
            - 'n_outputs_per_step' the output is written n times in each step it is active in regardless of the length
              of each step, the time points are evenly spaced. value is the number of outputs to be recorded
            - 'step times' the output is written at specific times measured from the beginning of the steps which the
              output is active in, this resets for every model step which uses this output. value should be a sorted
              list of relative times
            - 'global_times' the output is written at specific times measured from the start of model execution, no
              output will be written from steps where this output is not active. value should be a sorted list of
              absolute times
        value: Union[list, float], optional (1)
            Either a float or list of floats, as defined above

        Notes
        -----
        Outputs cannot be requested from the start of a step, outputs will only be written from steps which have the
        output added, this can be done through the individual steps or the contact model

        Example
        -------
        """
        self.name = name
        valid_modes = ('interval', 'time_interval', 'n_outputs_per_step', 'step_times', 'global_times')
        if timing_mode not in valid_modes:
            raise ValueError(f"Unrecognised timing mode: {timing_mode}, should be one of {', '.join(valid_modes)}")

        self.timing = timing_mode
        if timing_mode == 'interval':
            self.interval = value
            self._interval_count = value
        if timing_mode == 'time_interval':
            self.interval = value
            self.next_time_point = 0
        if timing_mode == 'n_outputs_per_step':
            self.interval = value
        if timing_mode == 'step_times':
            self.interval = None
            self.step_times = value
        if timing_mode == 'global_times':
            self.interval = None
            self.time_points = iter(value)
            try:
                self.next_time_point = next(self.time_points)
            except StopIteration:
                raise ValueError("Global times must have length of at least 1")

        self.parameters = [str(p) for p in parameters]
        self.start_time_this_step = 0
        self.end_time_this_step = None

    def new_step(self, current_time):
        """
        Called by the model when a new step is started
        """
        self.start_time_this_step = current_time
        self.end_time_this_step = None
        if self.timing == 'time_interval':
            self.next_time_point = current_time

    def is_active(self, current_time: float, step_max_time: float):
        """True if the output should be written on this time step
        """
        if self.timing == 'interval':
            if self._interval_count < self.interval:
                self._interval_count += 1
                return False
            else:
                self._interval_count = 1
                return True

        if self.timing == 'time_interval':
            if current_time >= self.next_time_point:
                self.next_time_point += self.interval
                return True
            return False

        if self.end_time_this_step is None:  # then new_step was just called
            self.end_time_this_step = self.start_time_this_step + step_max_time
            if self.timing == 'n_outputs_per_step':
                self.time_points = iter(np.linspace(self.start_time_this_step, self.end_time_this_step, self.interval))
            if self.timing == 'step_times':
                self.time_points = iter([self.start_time_this_step + st for st in self.step_times])
            try:
                self.next_time_point = next(self.time_points)
            except StopIteration:
                self.next_time_point = None

        if self.timing in ['global_times', 'n_outputs_per_step', 'step_times']:
            if self.next_time_point is not None and current_time >= self.next_time_point:
                try:
                    self.next_time_point = next(self.time_points)
                except StopIteration:
                    self.next_time_point = None
                return True
            return False

        warnings.warn(f"Output {self.name}, could not be written, output timing not recognised")
        return False


class OutputReader:
    """
    A class for reading and querying output files (.sdb) and array files (.sar)

    Parameters
    ----------
    file_name: str
        The path to the .sdb file with or without the extension, there should be a corresponding .sar file with the
        same name in the same directory

    Attributes
    ----------
    fields: set
        A set of all the fields which appear in the output database
    time_points: list
        A list of the all the time points in the output file


    See Also
    --------

    Notes
    -----
    All arrays in the output will be lazily read from the array file. These will not be read until the data from the
    array is requested. However, the lazy array objects are written so that numpy functions can use them as if they are
    arrays and they can be indexed as if they are arrays. Indexing the array will read the entire array into memory.

    Examples
    --------
    >>> with OutputSaver('test') as output:
    >>>     output.write({'time':0,'a':5, 'b':np.array([1,2,3,4]), 'c':np.array([1,2,3,4])})
    >>>     output.write({'time':1,'a':10, 'c':np.array([1,2,3,4])})
    >>>     output.write({'time':2,'a':15, 'b':np.array([1,2,3,4]), })
    >>>     output.write({'time':3,'a':20, 'b':np.array([1,2,3,4]), 'c':np.array([1,2,3,4])})
    >>> output = OutputReader('test')
    >>> output.time_points
    [0,1,2,3]
    >>> output[1]
    {'time': 1, 'a': 10, 'c': array, shape:(4,), dtype:int32}
    >>> output.fields
    {'a', 'b', 'c', 'time'}
    >>> output['a']
    {0: 5, 1: 10, 2: 15, 3: 20}
    >>> output['a'][1] == output[1]['a']
    True
    >>> from tinydb import Query
    >>> result = Query()
    >>> output.search(result.a == 10)
    [{'time': 1, 'a': 10, 'c': array, shape:(4,), dtype:int32}]
    >>> for result_from_time_point in output:
    >>>     print(result_from_time_point['time'])
    0
    1
    2
    3
    """
    def __init__(self, file_name):
        if not file_name.endswith('.sdb'):
            file_name += '.sdb'
        self.file_name = file_name
        with tinydb.TinyDB(file_name) as db:
            self.all_entries = db.search(tinydb.Query().time >= 0)
            self.fields = set()
            self.time_points = []
            for entry in self.all_entries:
                self.fields.update(entry)
                self.time_points.append(entry['time'])

    def __getitem__(self, time_point_or_field):
        with tinydb.TinyDB(self.file_name) as db:
            if time_point_or_field in self.time_points:
                query = tinydb.Query()
                raw_dict = db.search(query.time == time_point_or_field)[0]
            elif time_point_or_field in self.fields:
                f = time_point_or_field
                raw_dict = {entry['time']: (entry[f] if f in entry else None) for entry in self.all_entries}
            else:
                raise IndexError("Time point or field not recognised")
        return _array_gen(raw_dict, self.file_name[:-4] + '.sar')

    def search(self, query):
        """Wraps TinyDB search method replacing array codes in database with lazy arrays

        Parameters
        ----------
        query: tinydb.Query
            A valid tinydb Query

        Returns
        -------
        list of results that match the query

        Notes
        -----
        Queries cannot match against arrays as these are not stored directly in the database

        Examples
        --------
        >>> with OutputSaver('test') as output:
        >>>     output.write({'time':0,'a':5, 'b':np.array([1,2,3,4]), 'c':np.array([1,2,3,4])})
        >>>     output.write({'time':1,'a':10, 'c':np.array([1,2,3,4])})
        >>>     output.write({'time':2,'a':15, 'b':np.array([1,2,3,4]), })
        >>>     output.write({'time':3,'a':20, 'b':np.array([1,2,3,4]), 'c':np.array([1,2,3,4])})
        >>> output = OutputReader('test')
        >>> from tinydb import Query
        >>> result = Query()
        >>> output.search(result.a == 10)
        [{'time': 1, 'a': 10, 'c': array, shape:(4,), dtype:int32}]
        """
        with tinydb.TinyDB(self.file_name) as db:
            results_list = db.search(query)
        filled_results = []
        for result_dict in results_list:
            filled_results.append(_array_gen(result_dict, self.file_name[:-4] + '.sar'))
        return filled_results

    def __iter__(self):
        current = 0
        while current < len(self.time_points):
            yield self[self.time_points[current]]
            current += 1

    def __repr__(self):
        return (f"OutputReader \n"
                f"fields: {', '.join(self.fields)} \n"
                f"time points {', '.join(self.time_points)}")


def _array_gen(output_dict, array_file_name):
    """
    Helper function to replace string codes for _ArrayReader objects in the output of a query

    Parameters
    ----------
    output_dict: dict
        A dict, normally the result of a query on the database
    array_file_name: str
        The file name of the array file which goes with the output database

    Returns
    -------
    output_dict: dict
        The input dict modified by changing any array code values to _ArrayReader objects (lazily read arrays)
    """
    for key, value in output_dict.items():
        if isinstance(value, str) and value.startswith('**array**'):
            output_dict[key] = _ArrayReader(array_file_name, value)
    return output_dict


class _ArrayReader:
    """
    Helper class for reading arrays from zip file, should act like an array but
    lazily loads from the zipfile.

    Parameters
    ----------
    file_name: str
        The full path to the array file (.sar)
    entry_string: str
        The string from the corresponding entry in the output database

    Examples
    --------
    >>> with OutputSaver('test') as output_files:
    >>>     output_files.write({'my_array':np.array([1,2,3,4])})
    >>> lazy_array = _ArrayReader('test.sar', '**array**#0#(4,)#int32')
    >>> #                                                ^ file name in zip repo
    >>> #                                                   ^ shape of array
    >>> #                                                       ^ dtype of array
    >>> np.max(lazy_array)
    4
    >>> lazy_array[0]
    1
    """
    def __init__(self, file_name: str, entry_string: str):
        self.file_name = file_name
        hashes = [n for n, v in enumerate(entry_string) if v == '#']
        self.file_num = entry_string[hashes[0] + 1:hashes[1]]
        shape = entry_string[hashes[1] + 1:hashes[2]]
        if any([s not in '1234567890 ,()' for s in shape]):
            raise ValueError("Data in this file has been modified, unable to read")
        self.shape = eval(shape)
        self.dtype_name = entry_string[hashes[2] + 1:]
        # noinspection PyUnresolvedReferences
        self.dtype = np.__getattribute__(self.dtype_name)
        self._array = None

    @property
    def array(self):
        if self._array is None:
            self._fill_array()
        return self._array

    def _fill_array(self):
        with zipfile.ZipFile(self.file_name, 'r') as file:
            self._array = np.frombuffer(file.read('a' + self.file_num), dtype=self.dtype).reshape(self.shape)

    def __array__(self):
        return self.array

    def __getitem__(self, *items):
        return self.array.__getitem__(*items)

    def __repr__(self):
        return f"array, shape:{self.shape}, dtype:{self.dtype_name}"


def read_output(file_name: str) -> OutputReader:
    """Read a pair of output files

    Parameters
    ----------
    file_name: str
        The file name or full path to the database file (.sdb)

    Returns
    -------
    OutputReader object

    Notes
    -----
    The returned OutputReader object has attributes time_points and fields, these are all the time points for which
    an output was recorded (for single step static analysis there will be one time point at t=1)

    Indexing the OutputReader object with a time point will return all of the results from that time point as a
    dictionary

    Indexing the OutputReader with a field will give a dict of the results from that field for every time point.
    With the keys being the time values for each measurement, missing values will be replaced by None

    The result is that indexing with a time point then a field or a field then a time point gives the result for that
    field at the specific time.

    Any arrays in the results will be lazily read when they are first accessed, these can be converted to numpy arrays
    by array = np.array(lazy_array). However they should work with numpy functions and indexing without conversion.

    Examples
    --------
    >>> with OutputSaver('test') as output:
    >>>     output.write({'time':0,'a':5, 'b':np.array([1,2,3,4]), 'c':np.array([1,2,3,4])})
    >>>     output.write({'time':1,'a':10, 'c':np.array([1,2,3,4])})
    >>>     output.write({'time':2,'a':15, 'b':np.array([1,2,3,4]), })
    >>>     output.write({'time':3,'a':20, 'b':np.array([1,2,3,4]), 'c':np.array([1,2,3,4])})
    >>> output = read_output('test')
    >>> output.time_points
    [0,1,2,3]
    >>> output[1]
    {'time': 1, 'a': 10, 'c': array, shape:(4,), dtype:int32} # dict of results at a specific time
    >>> output.fields
    {'a', 'b', 'c', 'time'}
    >>> output['a']
    {0: 5, 1: 10, 2: 15, 3: 20} # dict with keys of time points and values of 'a' at each time point
    >>> output['a'][1] == output[1]['a']
    True
    """
    return OutputReader(file_name)
