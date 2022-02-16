import slippy.core as core
import numpy as np
from tinydb import Query


def test_read_write_output():
    with core.OutputSaver('test') as output_w:
        output_w.write({'time': 0, 'a': 5, 'b': np.array([1, 2, 3, 4]), 'c': np.array([1, 2, 3, 4], dtype=float)})
        output_w.write({'time': 1, 'a': 10, 'c': np.array([1, 2, 3, 4], dtype=float)})
        output_w.write({'time': 2, 'a': 15, 'b': np.array([1, 2, 3, 4]), })
        output_w.write({'time': 3, 'a': 20, 'b': np.array([1, 2, 3, 4]), 'c': np.array([1, 2, 3, 4], dtype=float)})
    output = core.OutputReader('test')
    assert output.time_points == [0, 1, 2, 3], str(output.time_points)
    assert output.fields == {'a', 'b', 'c', 'time'}, str(output.fields)
    assert output[1]['time'] == 1
    assert output['a'] == {0: 5, 1: 10, 2: 15, 3: 20}
    assert output['a'][1] == output[1]['a']
    result = Query()
    assert output.search(result.a == 10)
    times = []
    for out in output:
        times.append(out['time'])
    assert times == [0, 1, 2, 3]


def test_array_reading():
    dtypes = [int, np.int64, float, np.complex64, np.complex128]
    shapes = [(1,), (5,), (5, 6), (5, 6, 7)]
    sizes = [1, 5, 30, 210]
    i = 0
    strings = []
    with core.OutputSaver('test') as output_w:
        for dtype in dtypes:
            for size, shape in zip(sizes, shapes):
                for order in ['C', 'F']:
                    array = np.reshape(np.arange(size, dtype=dtype), newshape=shape)
                    if order == 'F':
                        array = np.asfortranarray(array)
                    output_w.write({'time': i, 'array': array})
                    i += 1
                    strings.append(f'dtype:{str(dtype)}, shape:{str(shape)}, order={order}')

    output = core.OutputReader('test')
    i = 0
    for output_line in output:
        array = np.array(output_line['array']).flatten()
        assert np.all(np.diff(array) == 1), "array not sorted " + strings[i]
        assert np.all((np.round(array)-array) == 0), "array incorrect data " + strings[i]
        i += 1


def test_lazy_array():
    with core.OutputSaver('test') as output_files:
        output_files.write({'my_array': np.array([1, 2, 3, 4], dtype=np.int32)})
    lazy_array = core.outputs._ArrayReader('test.sar', '**array**#0#(4,)#int32')
    assert np.max(lazy_array) == 4
    lazy_array = core.outputs._ArrayReader('test.sar', '**array**#0#(4,)#int32')
    assert lazy_array[0] == 1
