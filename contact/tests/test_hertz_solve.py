import numpy.testing as npt
from slippy.contact.hertz import solve_hertz_point, solve_hertz_line


# noinspection PyProtectedMember
def test_solve_hertz_point():
    set_params = ['r_rel', 'e1', 'e2', 'v1', 'v2', 'load']

    point_solution = solve_hertz_point(r_rel=0.01, e1=200e9, v1=0.3, e2=250e9, v2=0.33, load=1000)

    derived_params = [key for key in point_solution._asdict() if key not in set_params]
    derived_params.remove('e_star')

    for set_p in set_params:
        psd = point_solution._asdict()
        del psd[set_p]
        del psd['e_star']

        for der_p in derived_params:
            del psd[der_p]

        for der_p in derived_params:
            msg = f"Hertz point solution error: {set_p} not found accurately given {der_p}."
            try:
                print(set_p, der_p)
                psd[der_p] = getattr(point_solution, der_p)
                result = solve_hertz_point(**psd)
                npt.assert_approx_equal(getattr(result, set_p), getattr(point_solution, set_p), err_msg=msg)
            except NotImplementedError:
                raise AssertionError(msg + " This has not been implemented yet.")
            except StopIteration:
                raise AssertionError(msg + " Solver failed to converge.")
            finally:
                del psd[der_p]


# noinspection PyProtectedMember
def test_solve_hertz_line():

    set_params = ['r_rel', 'e1', 'e2', 'v1', 'v2', 'load']

    line_solution = solve_hertz_line(r_rel=0.01, e1=200e9, v1=0.3, e2=250e9, v2=0.33, load=1000)

    derived_params = [key for key in line_solution._asdict() if key not in set_params]
    derived_params.remove('e_star')

    for set_p in set_params:
        psd = line_solution._asdict()
        del psd[set_p]
        del psd['e_star']

        for der_p in derived_params:
            del psd[der_p]

        for der_p in derived_params:
            msg = f"Hertz line solution error: {set_p} not found accurately given {der_p}."
            try:
                print(set_p, der_p)
                psd[der_p] = getattr(line_solution, der_p)
                result = solve_hertz_line(**psd)
                npt.assert_approx_equal(getattr(result, set_p), getattr(line_solution, set_p), err_msg=msg)
            except NotImplementedError:
                raise AssertionError(msg + " This has not been implemented yet.")
            except StopIteration:
                raise AssertionError(msg + " Solver failed to converge.")
            finally:
                del psd[der_p]
