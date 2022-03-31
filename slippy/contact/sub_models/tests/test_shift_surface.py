import numpy as np
import numpy.testing as npt
import slippy.surface as s
import slippy.contact as c


def test_shift_surface():
    n = 10
    flat1 = s.FlatSurface(shape=(n, n), grid_spacing=1, generate=True)
    flat2 = s.FlatSurface(shape=(n, n), grid_spacing=1, generate=True)
    roughness = s.FlatSurface(shape=(n, 2 * n), grid_spacing=1, generate=True)
    rolling = s.RollingSurface(roughness, flat2)

    model = c.ContactModel('test_mod', rolling, flat1)

    state = {"contact_nodes": np.ones(shape=(n, n)),
             'time_step': 1.0,
             'surface_1_points': rolling.get_points_from_extent(),
             'surface_2_points': flat1.get_points_from_extent()}
    step = c.RepeatingStateStep('', time_steps=n, state=state)
    model.add_step(step)

    ct = c.sub_models.ResultContactTime('', False)
    shift = c.sub_models.UpdateShiftRollingSurface('', 1, 1)
    step.add_sub_model(ct)
    step.add_sub_model(shift)

    result = model.solve(skip_data_check=True)
    npt.assert_array_equal(result['contact_time_1'][0], np.flip(np.arange(1, 11)))
    npt.assert_array_equal(result['contact_time_2'], n * np.ones_like(result['contact_time_2']))


def test_shift_surface_wrap():
    n = 10
    flat1 = s.FlatSurface(shape=(n, n), grid_spacing=1, generate=True)
    flat2 = s.FlatSurface(shape=(n, n), grid_spacing=1, generate=True)
    roughness = s.FlatSurface(shape=(n, 2 * n), grid_spacing=1, generate=True)
    rolling = s.RollingSurface(roughness, flat2)

    model = c.ContactModel('test_mod', rolling, flat1)

    state = {"contact_nodes": np.ones(shape=(n, n)),
             'time_step': 1.0,
             'surface_1_points': rolling.get_points_from_extent(),
             'surface_2_points': flat1.get_points_from_extent()}
    step = c.RepeatingStateStep('', time_steps=n, state=state)
    model.add_step(step)

    ct = c.sub_models.ResultContactTime('', False)
    shift = c.sub_models.UpdateShiftRollingSurface('', 1, 0.0, 1)
    step.add_sub_model(ct)
    step.add_sub_model(shift)

    result = model.solve(skip_data_check=True)
    npt.assert_array_equal(result['contact_time_1'], n * np.ones((n, n)))
    npt.assert_array_equal(result['contact_time_2'], n * np.ones((n, n)))
