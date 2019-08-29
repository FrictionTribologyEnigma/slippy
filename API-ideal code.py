# read in surfaces

import slippy.surface as S

surface_1 = S.read_surface_from_file()
surface_2 = S.FlatSurface(0)

# apply materials

import slippy.contact as C
import slippy.lubrication as L

steel = C.material('elastic', 200, 0.3)  # < materials done
surface_1.material = steel  # this should work now
surface_2.set_material('rigid')  # equavilent to above 2 lines #Check this

viscosity = 1
parameters = 1e-10
# init gives viscosity behaviour
# add density changes under load or temp after or in kwargs for init mehtod
oil = L.lubricant('newtonian', viscosity)
oil.add_behaviour('property', 'rule', parameters)
oil.add_behaviour('property', 'rule', parameters)

# make model
my_model = C.model(surface_1, surface_2)
my_model.set_lubricant(oil)
my_model.set_adhesion(adhesion_model)
my_model.set_friction(friction_model)
my_model.set_wear(wear_model)

field_output_requests = C.field_output(['flash_temperature', 'subsurface_stress'], time='end', steps='all')
history_output = C.history_output(['tangential_load', 'total_wear', 'maximum strees', 'contact_points'],
                                  time=np.arange(10), steps=1)
my_model.new_step('main', 'relative_motion_cmbem', speed=[1, 0], load=100,
                  f_output=1, time_step=0.001, position='last')

my_model.solve('output_filename')

my_results = C.Results('output_filename')

my_results.show('flash_temperature')