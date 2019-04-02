#read in surfaces

import slippy.surface as S
surface_1=S.read_surface_from_file()
surface_2=S.FlatSurface(0)

#apply materials

import slippy.contact as C

steel=C.material('elastic', 200, 0.3)
surface_1.material=steel
surface_2.set_material('rigid') # equavilent to above 2 lines

viscosity=1
parameters=1e-10
oil=C.lubricant('newtonian', viscosity)
C.add_behaviour('property', 'rule', parameters)
C.add_behaviour('property', 'rule', parameters)

# make model
my_model=C.model(surface_1, surface_2)
my_model.set_lubricant(oil)
my_model.set_adhesion(adhesion_model)
my_model.set_friction(friction_model)
my_model.set_wear(wear_model)

field_output_requests=['flash_temperature', 'subsurface_stress']
history_output=['tangential_load', 'total_wear', 'maximum strees', 'contact_points']
my_model.new_step('main', 'relative_motion_cmbem', speed=[1,0], load=100, 
                  output=output_requests, time_step=0.001, position='last')

possible_modes=['separation', 'static', 'roling_sliding_cmbem']

my_model.solve('output_filename')

my_results=C.Results('output_filename')

my_results.show('flash_temperature')
