import typing
from numba import njit
import numpy as np
from scipy.linalg.lapack import dgtsv

from slippy.abcs import _NonDimensionalReynoldSolverABC

__all__ = ['UnifiedReynoldsSolver']


class UnifiedReynoldsSolver(_NonDimensionalReynoldSolverABC):
    # noinspection SpellCheckingInspection
    """
        The unified reynolds solver for use in lubrication steps

        Parameters
        ----------
        time_step: float
            The dimentional time step for the calculation in seconds, minutes ... etc.
        grid_spacing: float
            The dimentional grid spacing of the master surface
        hertzian_pressure: float
            The static hertzian pressure of the contact, or any other representative pressure, this is used only to
            non-dimentionalise the problem
        radius_in_rolling_direction: float
            The radius of the ball or dic in the rolling direction, again this is just used to non-dimentionalise the
            problem, if it is not known a different representative length can be used
        hertzian_half_width: float
            The hertzian half width of the contact, this is used to non-dimentionalise the problem, if this is not known
            another representative length can be used
        dimentional_viscosity:float
            The viscosity used to non-dimentionalise the problem, usually the viscosity when the pressure is 0.
        dimentional_density: float
            The density used to non-dimentionalise the problem, usually the density when the pressure is 0.
        rolling_speed: float, optional (None)
            The dimentional mean speed of the surfaces (u1+u2)/2, this is normally set by the step
        sweep_direction: str {'forward', 'backward'}, optional ('forward')
            The direction which the reynolds solver moves through the pressure array.

        Attributes
        ----------

        time_step
            The dimentionalised time step
        nd_time_step: read only
            The non dimentionalised time step
        grid_spacing: read only
            The dimentionalised grid spacing
        nd_grid_spacing: read only
            The non dimentionalised grid spacing
        lambda_bar: read only
            The lambda parameter for the problem
        rolling_speed
            The mean speed of the surfaces (u1+u2)/2
        hertzian_pressure
            The non dimentionalising pressure
        hertzian_half_width
            The non dimentionalising length

        See Also
        --------
        IterSemiSystemLoad - semi system iteration lubrication step

        Notes
        -----
        The units used for the input values do not matter but they must be consistent. eg if the time step is in seconds
        and the hertzian half width is in meters the rolling speed must be in meters/second

        Values for rolling speed and non dimentionalising values can be updated by the user or the step

        Examples
        --------
        #TODO

        References
        ----------
        Azam, A., Dorgham, A., Morina, A., Neville, A., & Wilson, M. C. T. (2019). A simple deterministic
        plastoelastohydrodynamic lubrication (PEHL) model in mixed lubrication. Tribology International,
        131(November 2018), 520–529. https://doi.org/10.1016/j.triboint.2018.11.011
        """
    requires = {'nd_gap', 'nd_pressure', 'nd_viscosity', 'nd_density'}
    provides = {'nd_pressure', 'previous_nd_gap', 'previous_nd_density'}
    _row_order = None  # order the rows will be solved in, controlled by the sweep direction

    _hertzian_pressure: float = None
    _hertzian_half_width: float = None
    _dimentional_viscosity: float = None
    _dimentional_density: float = None
    _rolling_speed: typing.Optional[float] = None
    _lambda_bar: typing.Optional[float] = None
    _radius: float = None

    def __init__(self, time_step: float,
                 grid_spacing: float,
                 hertzian_pressure: float,
                 radius_in_rolling_direction: float,
                 hertzian_half_width: float,
                 dimentional_viscosity: float,
                 dimentional_density: float,
                 sweep_direction: str = 'backward'):
        # these automatically calculate the non dimentional versions
        self.grid_spacing = grid_spacing
        self.time_step = time_step

        # find lambda bar (all of these are properties apart from dimentional_density)
        self.radius = radius_in_rolling_direction
        self.hertzian_pressure = hertzian_pressure
        self.hertzian_half_width = hertzian_half_width
        self.dimentional_viscosity = dimentional_viscosity
        self.dimentional_density = dimentional_density
        self.rolling_speed = None

        # get first 3 components of influence matrix
        def stencil(x, y):
            return x + np.sqrt(x ** 2 + y ** 2)

        ak = np.zeros((3,))

        for i in range(3):
            xp = i + 0.5
            xm = i - 0.5
            ym = -0.5
            yp = 0.5
            a1 = stencil(yp, xp) / stencil(ym, xp)
            a2 = stencil(xm, ym) / stencil(xp, ym)
            a3 = stencil(ym, xm) / stencil(yp, xm)
            a4 = stencil(xp, yp) / stencil(xm, yp)
            ak[i] = xp * np.log(a1) + ym * np.log(a2) + xm * np.log(a3) + yp * np.log(a4)

        self.ak00 = 2 / np.pi ** 2 * ak[0]
        self.ak10 = 2 / np.pi ** 2 * ak[1]
        self.ak20 = 2 / np.pi ** 2 * ak[2]

        if sweep_direction == 'forward':
            self._step = 1
        elif sweep_direction == 'backward':
            self._step = -1
        else:
            raise ValueError(f"Unrecognised sweep direction: {sweep_direction}")

    @property
    def nd_grid_spacing(self):
        return self.grid_spacing / self.hertzian_half_width

    @property
    def nd_time_step(self):
        return self.time_step * self.rolling_speed / self.hertzian_half_width

    @property  # Lambda bar cannot be set directly
    def lambda_bar(self):
        if self._lambda_bar is not None:
            return self._lambda_bar
        else:
            self._lambda_bar = (12 * self.rolling_speed * self.dimentional_viscosity * self.radius ** 2 /
                                (self.hertzian_half_width ** 3 * self.hertzian_pressure))
            return self._lambda_bar

    # properties for everything that would change lambda bar if it changed
    @property
    def hertzian_pressure(self):
        return self._hertzian_pressure

    @hertzian_pressure.setter
    def hertzian_pressure(self, value):
        self._hertzian_pressure = value
        self._lambda_bar = None

    @property
    def hertzian_half_width(self):
        return self._hertzian_half_width

    @hertzian_half_width.setter
    def hertzian_half_width(self, value):
        self._hertzian_half_width = value
        self._lambda_bar = None

    @property
    def dimentional_viscosity(self):
        return self._dimentional_viscosity

    @dimentional_viscosity.setter
    def dimentional_viscosity(self, value):
        self._dimentional_viscosity = value
        self._lambda_bar = None

    @property
    def rolling_speed(self):
        return self._rolling_speed

    @rolling_speed.setter
    def rolling_speed(self, value):
        self._rolling_speed = value
        self._lambda_bar = None

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value
        self._lambda_bar = None

    def data_check(self, previous_state: set) -> set:
        for requirement in self.requires:
            if requirement not in previous_state:
                raise ValueError(f"Unified reynolds solver requires {requirement}, but this is not provided by the "
                                 "step")
        previous_state = set(self.provides)
        return previous_state

    def solve(self, previous_state: dict, max_pressure: float) -> dict:
        # rumble
        nd_gap = previous_state['nd_gap']
        width, length = nd_gap.shape
        pressure = previous_state['nd_pressure'].copy()
        current_state = dict()

        if 'previous_nd_density' not in previous_state:  # first time step
            previous_state['previous_nd_density'] = previous_state['nd_density']
            previous_state['previous_nd_gap'] = previous_state['nd_gap']

        # These values are from the last time step not the last iteration
        current_state['previous_nd_density'] = previous_state['previous_nd_density']
        current_state['previous_nd_gap'] = previous_state['previous_nd_gap']

        # pre calculate some values to save time
        recip_dx_squared_rho = 1 / (self.nd_grid_spacing ** 2 * previous_state['nd_density'])
        recip_dx = 1 / self.nd_grid_spacing
        recip_dt = 1 / self.nd_time_step if self.nd_time_step else 0.0

        epsilon = self._get_epsilon(previous_state)

        # sort out the row order
        if not self._row_order:
            if self._step == 1:
                self._row_order = [1, length - 1]
            elif self._step == -1:
                self._row_order = [length - 2, 0]
            else:
                raise ValueError("Row step must be -1 or 1")

        ak00 = self.ak00
        ak10 = self.ak10
        ak20 = self.ak20

        nd_density = previous_state['nd_density']
        previous_nd_density = previous_state['previous_nd_density']
        previous_nd_gap = previous_state['previous_nd_gap']

        a_all, c_all = np.zeros_like(epsilon[:-1, 0]), np.zeros_like(epsilon[:-1, 0])
        b_all, f_all = np.ones_like(epsilon[:, 0]), np.zeros_like(epsilon[:, 0])
        # solve line by line
        for row in range(self._row_order[0], self._row_order[1], self._step):
            a_all, b_all, c_all, f_all = _solve_row(epsilon, row, pressure, recip_dx_squared_rho, recip_dx, recip_dt,
                                                    a_all, b_all, c_all, f_all, ak00, ak10, ak20, nd_gap, nd_density,
                                                    previous_nd_density, previous_nd_gap)

            p1d = np.clip(thomas_tdma(a_all, b_all, c_all, f_all), 0, max_pressure)

            pressure[1:-1, row] = p1d[1:-1]

        current_state['nd_pressure'] = pressure

        return current_state

    def _get_epsilon(self, previous_state: dict) -> np.ndarray:
        nd_gap = previous_state['nd_gap']
        epsilon = previous_state['nd_density'] * nd_gap ** 3 / previous_state['nd_viscosity'] * (1 / self.lambda_bar)
        epsilon[nd_gap < self.dimensionalise_gap(0.47e-9, True)] = 0
        return epsilon

    def dimensionalise_pressure(self, nd_pressure, un_dimensionalise: bool = False):
        if un_dimensionalise:
            return nd_pressure / self.hertzian_pressure
        return nd_pressure * self.hertzian_pressure

    def dimensionalise_viscosity(self, nd_viscosity, un_dimensionalise: bool = False):
        if un_dimensionalise:
            return nd_viscosity / self.dimentional_viscosity
        return nd_viscosity * self.dimentional_viscosity

    def dimensionalise_density(self, nd_density, un_dimensionalise: bool = False):
        if un_dimensionalise:
            return nd_density / self.dimentional_density
        return nd_density * self.dimentional_density

    def dimensionalise_gap(self, nd_gap, un_dimensionalise: bool = False):
        if un_dimensionalise:
            return self.radius / self.hertzian_half_width ** 2 * nd_gap
        return self.hertzian_half_width ** 2 / self.radius * nd_gap

    def dimensionalise_length(self, nd_length, un_dimensionalise: bool = False):
        if un_dimensionalise:
            return nd_length / self.hertzian_half_width
        return nd_length * self.hertzian_half_width


@njit
def _solve_row(epsilon, row, pressure, recip_dx_squared_rho, recip_dx, recip_dt, a_all, b_all, c_all, f_all,
               ak00, ak10, ak20, nd_gap, nd_density, previous_nd_density, previous_nd_gap):
    d1 = 0.5 * (epsilon[1:-1, row] + epsilon[0:-2, row])
    d2 = 0.5 * (epsilon[1:-1, row] + epsilon[2:, row])
    d4 = 0.5 * (epsilon[1:-1, row] + epsilon[1:-1, row - 1])
    d5 = 0.5 * (epsilon[1:-1, row] + epsilon[1:-1, row + 1])
    d3 = d1 + d2 + d4 + d5

    q1 = ak10 * pressure[0:-2, row] + ak00 * pressure[1:-1, row] + ak10 * pressure[2:, row]
    q2 = ak00 * pressure[0:-2, row] + ak10 * pressure[1:-1, row] + ak20 * pressure[2:, row]

    # Pressure flow terms
    a_p = d1 * recip_dx_squared_rho[1:-1, row]
    b_p = -d3 * recip_dx_squared_rho[1:-1, row]
    c_p = d2 * recip_dx_squared_rho[1:-1, row]
    f_p = -(d5 * pressure[1:-1, row + 1] + d4 * pressure[1:-1, row - 1]) * recip_dx_squared_rho[1:-1, row]

    # Wedge flow terms
    a_w = (ak00 - ak10) * recip_dx
    b_w = (ak10 - ak00) * recip_dx
    c_w = (ak20 - ak10) * recip_dx
    f_w = (((nd_gap[1:-1, row] - q1) - (nd_gap[0:-2, row] - q2)) * recip_dx +
           nd_gap[1:-1, row] * (1 - (nd_density[0:-2, row] / nd_density[1:-1, row])) * recip_dx)  # +
    # recip_dx * (nd_gap[1:-1, row] - nd_gap[0:-2, row]))  # these two gaps were roughness in fortran code

    # squeeze flow terms
    a_s = -1 * ak10 * recip_dt
    b_s = -1 * ak00 * recip_dt
    c_s = -1 * ak10 * recip_dt
    f_s = ((nd_gap[1:-1, row] - q1) - (previous_nd_density[1: -1, row] / nd_density[1:-1, row]) *
           previous_nd_gap[1: -1, row]) * recip_dt

    # add and apply boundary conditions here (a[-1] = c[0] = f[0 and -1] = 0, b[0 and -1] = 1)
    a_all[:-1] = a_p + a_s + a_w
    b_all[1:-1] = b_p + b_s + b_w
    c_all[1:] = c_p + c_s + c_w
    f_all[1:-1] = f_p + f_s + f_w

    return a_all, b_all, c_all, f_all


def thomas_tdma(lower_diagonal, main_diagonal, upper_diagonal, right_hand_side):
    """The thomas algorithm (TDMA) solution for tri-diagonal matrix inversion

    Parameters
    ----------
    lower_diagonal: np.ndarray
        The lower diagonal of the matrix length n-1
    main_diagonal: np.ndarray
        The main diagonal of the matrix length n
    upper_diagonal: np.ndarray
        The upper diagonal of the matrix length n-1
    right_hand_side: np.ndarray
        The array bof size max(1, ldb*n_rhs) for column major layout and max(1, ldb*n) for row major layout contains the
        matrix B whose columns are the right-hand sides for the systems of equations.

    Returns
    -------
    x: np.ndarray
        The solution array length n

    Notes
    -----
    Nothing is mutated by this function
    """
    _, _, _, x, _ = dgtsv(lower_diagonal, main_diagonal, upper_diagonal, right_hand_side)
    return x
