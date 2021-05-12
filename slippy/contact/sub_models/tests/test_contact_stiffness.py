import numpy as np
import numpy.testing as npt
import slippy.surface as s
import slippy.contact as c


def test_contact_stiffness():
    """
    Test the contact stiffness sub model against a known analytical result
    Also tests the example
    """
    diameter = 1
    resolution = 512
    x, y = np.meshgrid(np.linspace(-diameter, diameter, resolution),
                       np.linspace(-diameter, diameter, resolution))
    indenter_profile = np.array((x ** 2 + y ** 2) < (diameter / 2) ** 2, dtype=np.float32)
    grid_spacing = x[1, 1] - x[0, 0]
    indenter = s.Surface(profile=indenter_profile, grid_spacing=grid_spacing)
    half_space = s.FlatSurface(shape=(resolution, resolution), grid_spacing=grid_spacing,
                               generate=True)

    indenter.material = c.rigid
    e, v = 200e9, 0.3
    half_space.material = c.Elastic('steel', {'E': e, 'v': v})
    reduced_modulus = 1 / ((1 - v ** 2) / e)

    my_model = c.ContactModel('Contact_stiffness_example', half_space, indenter)

    step = c.StaticStep('loading', interference=1e-4, periodic_geometry=True)
    sub_model = c.sub_models.ResultContactStiffness('stiffness', definition='far points',
                                                    loading=False)
    step.add_sub_model(sub_model)
    my_model.add_step(step)

    # we don't need to solve the contact model we already know what the contact nodes will be

    contact_nodes = (x ** 2 + y ** 2) < (diameter / 2) ** 2

    current_state = {'contact_nodes': contact_nodes,
                     'loads_z': np.zeros_like(contact_nodes)}

    results = sub_model.solve(current_state)

    numerical_stiffness = results['s_contact_stiffness_unloading_fp_z_0'] * (2 * diameter) ** 2
    analytical_stiffness = reduced_modulus * diameter

    npt.assert_approx_equal(numerical_stiffness, analytical_stiffness, 2)
