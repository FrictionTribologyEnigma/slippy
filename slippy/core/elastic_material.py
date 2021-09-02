import numpy as np
import typing
from collections import namedtuple
from .materials import _IMMaterial
from ._elastic_sub_surface_stresses import normal_conv_kernels, tangential_conv_kernels

__all__ = ['Elastic', 'elastic_influence_matrix']

ElasticProps = namedtuple('ElasticProperties', 'K E v Lam M G', defaults=(None,) * 6)


# noinspection PyPep8Naming
class Elastic(_IMMaterial):
    """ A Class for defining elastic materials

    Parameters
    ----------
    name: str
        The name of the material
    properties: dict
        dict of properties, dicts must have exactly 2 items.
        Allowed keys are : 'E', 'v', 'G', 'K', 'M', 'Lam'
        See notes for definitions
    max_load: float
        The maximum load on the surface, loads above this will be cropped during analysis, if this is specified a
        plastic deformation sub model should be added to the end of each model step to make the deformation permanent

    Methods
    -------
    speed_of_sound

    See Also
    --------

    Notes
    -----

    Keys refer to:
        - E   - Young's modulus
        - v   - Poission's ratio
        - K   - Bulk Modulus
        - Lam - Lame's first parameter
        - G   - Shear modulus
        - M   - P wave modulus

    Examples
    --------
    >>> # Make a material model for elastic steel
    >>> steel = Elastic('steel', {'E': 200e9, 'v': 0.3})
    >>> # Find it's p-wave modulus:
    >>> pwm = steel.M
    >>> # Find the speeds of sound:
    >>> sos = steel.speed_of_sound(7890)
    """

    material_type = 'Elastic'

    _properties = {'E': None,
                   'v': None,
                   'G': None,
                   'K': None,
                   'Lam': None,
                   'M': None, }

    _last_set = []
    density = None

    def __init__(self, name: str, properties: dict, max_load: float = np.inf):
        super().__init__(name, max_load)

        if len(properties) > 2:
            raise ValueError("Too many properties supplied, must be 1 or 2")

        for item in properties.items():
            self._set_props(*item)

    def _influence_matrix(self, components: typing.Union[typing.Sequence[str], str],
                          grid_spacing: {typing.Sequence[float], float}, span: typing.Sequence[int]):
        """
        Influence matrix for an elastic material

        Parameters
        ----------
        grid_spacing: tuple
            The spacing between grid points in the x and y directions
        span: tuple
            The span required in the x and y directions in number of grid points
        components: str or Sequence {'xx','xy','xz','yx','yy','yz','zx','zy','zz','all'}
            The required components eg the 'xy' component represents the x
            deflection caused by loads in the y direction

        Returns
        -------
        dict
            dict of the requested influence matrix or matrices

        See Also
        --------
        elastic_loading
        elastic_deflection

        Notes
        -----

        K^{ij i'j'}_zz=(1-v)/(2*pi*G)*Czz

        Czz=(hx*(k*log((m+sqrt(k**2+m**2))/(n+sqrt(k**2+n**2)))+
                 l*log((n+sqrt(l**2+n**2))/(m+sqrt(l**2+m**2))))+
             hy*(m*log((k+sqrt(k**2+m**2))/(l+sqrt(l**2+m**2)))+
                 n*log((l+sqrt(l**2+n**2))/(k+sqrt(k**2+n**2)))))

        In which:

        k=i'-i+0.5
        l=i'-i-0.5
        m=j'-j+0.5
        n=j'-j-0.5
        hx=grid_spacing[0]
        hy=grid_spacing[1]

        If both shear_modulus_2 and v_2 are supplied and are not None the combined IM is returned for the surface
        pair

        Examples
        --------


        References
        ----------
        Complete boundary element method formulation for normal and tangential
        contact problems

        """

        # if other is not None:
        #     if isinstance(other, Elastic) or isinstance(other, Rigid):
        #         shear_modulus_2 = other.G
        #         v_2 = other.v
        #     else:
        #         raise NotImplementedError("Combined influence matrix cannot be found for this material pair")
        # else:
        shear_modulus_2 = None
        v_2 = None

        shear_modulus = self.G
        v = self.v

        if len(span) == 1:
            span *= 2
        if len(grid_spacing) == 1:
            grid_spacing *= 2

        if components == 'all':
            components = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']

        components = {comp: elastic_influence_matrix(comp, span, grid_spacing, shear_modulus, v,
                                                     shear_mod_2=shear_modulus_2, v_2=v_2) for comp in components}

        return components

    def _del_props(self, prop):
        # delete any of the material properties
        keys = list(self._properties.keys())
        if self._last_set == prop:
            self._properties = {key: None for key in keys}
            self._last_set = None
        else:
            self._properties = {key: None for key in keys
                                if not key == self._last_set}

    def _set_props(self, prop, value):
        allowed_props = ['E', 'v', 'G', 'K', 'Lam', 'M']
        if prop not in allowed_props:
            msg = (f'property {prop} not recognised allowed propertied are: ' +
                   ' '.join(allowed_props))
            raise ValueError(msg)

        self._properties[prop] = np.float64(value)

        if len(self._last_set) == 0:
            self._last_set.append(prop)  # if none ever set just set it
        elif self._last_set[-1] != prop:
            self._last_set.append(prop)  # if the last set is different replace it

        if len(self._last_set) > 1:  # if 2 props have been set update all
            set_props = {prop: np.float64(value),
                         self._last_set[-2]: self._properties[self._last_set[-2]]}
            self._properties = _get_properties(set_props)
        return

    @property
    def E(self):
        """The Young's modulus of the material"""
        return self._properties['E']

    @E.deleter
    def E(self):
        self._del_props('E')

    @E.setter
    def E(self, value):
        self._set_props('E', value)

    @property
    def v(self):
        """The Poissions's ratio of the material"""
        return self._properties['v']

    @v.deleter
    def v(self):
        self._del_props('v')

    @v.setter
    def v(self, value):
        self._set_props('v', value)

    @property
    def G(self):
        """The shear modulus of the material"""
        return self._properties['G']

    @G.deleter
    def G(self):
        self._del_props('G')

    @G.setter
    def G(self, value):
        self._set_props('G', value)

    @property
    def K(self):
        """The bulk modulus of the material"""
        return self._properties['K']

    @K.deleter
    def K(self):
        self._del_props('K')

    @K.setter
    def K(self, value):
        self._set_props('K', value)

    @property
    def Lam(self):
        """Lame's first parameter for the material"""
        return self._properties['Lam']

    @Lam.deleter
    def Lam(self):
        self._del_props('Lam')

    @Lam.setter
    def Lam(self, value):
        self._set_props('Lam', value)

    @property
    def M(self):
        """The p wave modulus of the material"""
        return self._properties['M']

    @M.deleter
    def M(self):
        self._del_props('M')

    @M.setter
    def M(self, value):
        self._set_props('M', value)

    def speed_of_sound(self, density: float = None):
        """find the speed of sound in the material

        Parameters
        ----------
        density : float optional (None)
            The density of the material

        Returns
        -------

        speeds : dict
            With keys 's' and 'p' giving the s and p wave speeds

        Notes
        -----

        Finds speeds according to the following equations:

        Vs=sqrt(G/rho)
        Vp=sqrt(M/rho)

        Where rho is the density, G is the shear modulus and M is the p wave
        modulus

        Examples
        --------
        >>> # Find the speed of sound in steel
        >>> my_material = Elastic({'E': 200e9, 'v': 0.3})
        >>> my_material.speed_of_sound(7850)

        """
        if density is not None:
            self.density = density
        elif self.density is None:
            raise ValueError("Density not given or set")

        speeds = {'s': np.sqrt(self.G / self.density),
                  'p': np.sqrt(self.M / self.density)}

        return speeds

    def sss_influence_matrices_normal(self, components: typing.Sequence[str], grid_spacing: typing.Sequence[float],
                                      span: typing.Sequence[int], z: typing.Sequence[float] = None, cuda: bool = False):
        if z is None:
            z_len = min(span)
            print(span)
            gs = grid_spacing[span.index(z_len)]
            print(gs)
            z = gs*np.arange(z_len//2)
            z[0] = z[1]*1e-4
        all_matrices = normal_conv_kernels(span, z, grid_spacing, self.E, self.v, cuda=cuda)
        return {comp: all_matrices[comp] for comp in components}

    def sss_influence_matrices_tangential_x(self, components: typing.Sequence[str],
                                            grid_spacing: typing.Sequence[float], span: typing.Sequence[int],
                                            z: typing.Sequence[float] = None, cuda: bool = False):
        if z is None:
            z_len = min(span)
            gs = grid_spacing[span.index(z_len)]
            z = gs*np.arange(z_len//2)
            z[0] = z[1]*1e-4
        all_matrices = tangential_conv_kernels(span, z, grid_spacing, self.v, cuda=cuda)
        return {comp: all_matrices[comp] for comp in components}

    def __repr__(self):
        return "Elastic(name = '" + self.name + f"', properties = {{ 'E':{self.E}, 'v':{self.v} }}"


def _get_properties(set_props: dict):
    """Get all elastic properties from any pair

    Parameters
    ----------
    set_props : dict
        dict of properties must have exactly 2 members valid keys are: 'K',
        'E', 'v', 'Lam', 'M', 'G'

    Returns
    -------
    out : dict
        dict of all material properties keys are: 'K', 'E', 'v', 'Lam', 'M', 'G'

    Notes
    -----

    Keys refer to:
        - E - Young's modulus
        - v - Poission's ratio
        - K - Bulk Modulus
        - Lam - Lame's first parameter
        - G - Shear modulus
        - M - P wave modulus

    """
    if len(set_props) != 2:
        raise ValueError("Exactly 2 properties must be set,"
                         " {} found".format(len(set_props)))

    valid_keys = ['K', 'E', 'v', 'G', 'Lam', 'M']

    set_params = [key for key in list(set_props.keys()) if key in valid_keys]

    if len(set_params) != 2:
        msg = ("Invalid keys in set_props keys found are: " +
               "{}".format(set_props.keys()) +
               ". Valid keys are: " + " ".join(valid_keys))
        raise ValueError(msg)

    out = set_props.copy()

    set_params = list(set_props.keys())
    set_params.sort()
    # p is properties this saves a lot of space
    p = ElasticProps(**set_props)

    if set_params[0] == 'E':
        if set_params[1] == 'G':
            out['K'] = p.E * p.G / (3 * (3 * p.G - p.E))
            out['Lam'] = p.G * (p.E - 2 * p.G) / (3 * p.G - p.E)
            out['M'] = p.G * (4 * p.G - p.E) / (3 * p.G - p.E)
            out['v'] = p.E / (2 * p.G) - 1
        elif set_params[1] == 'K':
            out['G'] = 3 * p.K * p.E / (9 * p.K - p.E)
            out['Lam'] = 3 * p.K * (3 * p.K - p.E) / (9 * p.K - p.E)
            out['M'] = 3 * p.K * (3 * p.K + p.E) / (9 * p.K - p.E)
            out['v'] = (3 * p.K - p.E) / (6 * p.K)
        elif set_params[1] == 'Lam':
            R = np.sqrt(p.E ** 2 + 9 * p.Lam ** 2 + 2 * p.E * p.Lam)
            out['G'] = (p.E - 3 * p.Lam + R) / 4
            out['K'] = (p.E + 3 * p.Lam + R) / 6
            out['M'] = (p.E - p.Lam + R) / 2
            out['v'] = 2 * p.Lam / (p.E + p.Lam + R)
        elif set_params[1] == 'M':
            S = np.sqrt(p.E ** 2 + 9 * p.M ** 2 - 10 * p.E * p.M)
            out['G'] = (3 * p.M + p.E - S) / 8
            out['K'] = (3 * p.M - p.E + S) / 6
            out['Lam'] = (p.M - p.E + S) / 4
            out['v'] = (p.E - p.M + S) / (4 * p.M)
        else:  # set_params[1]=='v'
            out['G'] = p.E / (2 * (1 + p.v))
            out['K'] = p.E / (3 * (1 - 2 * p.v))
            out['Lam'] = p.E * p.v / ((1 + p.v) * (1 - 2 * p.v))
            out['M'] = p.E * (1 - p.v) / ((1 + p.v) * (1 - 2 * p.v))
    elif set_params[0] == 'G':
        if set_params[1] == 'K':
            out['E'] = 9 * p.K * p.G / (3 * p.K + p.G)
            out['Lam'] = p.K - 2 * p.G / 3
            out['M'] = p.K + 4 * p.G / 3
            out['v'] = (3 * p.K - 2 * p.G) / (2 * (3 * p.K + p.G))
        elif set_params[1] == 'Lam':
            out['E'] = p.G * (3 * p.Lam + 2 * p.G) / (p.Lam + p.G)
            out['K'] = p.Lam + 2 * p.G / 3
            out['M'] = p.Lam + 2 * p.G
            out['v'] = p.Lam / (2 * (p.Lam + p.G))
        elif set_params[1] == 'M':
            out['E'] = p.G * (3 * p.M - 4 * p.G) / (p.M - p.G)
            out['K'] = p.M - 4 * p.G / 3
            out['Lam'] = p.M - 2 * p.G
            out['v'] = (p.M - 2 * p.G) / (2 * p.M - 2 * p.G)
        else:  # set_params[1]=='v'
            out['E'] = 2 * p.G * (1 + p.v)
            out['K'] = 2 * p.G * (1 + p.v) / (3 * (1 - 2 * p.v))
            out['Lam'] = 2 * p.G * p.v / (1 - 2 * p.v)
            out['M'] = 2 * p.G * (1 - p.v) / (1 - 2 * p.v)
    elif set_params[0] == 'K':
        if set_params[1] == 'Lam':
            out['E'] = 9 * p.K * (p.K - p.Lam) / (3 * p.K - p.Lam)
            out['G'] = 3 * (p.K - p.Lam) / 2
            out['M'] = 3 * p.K - 2 * p.Lam
            out['v'] = p.Lam / (3 * p.K - p.Lam)
        elif set_params[1] == 'M':
            out['E'] = 9 * p.K * (p.M - p.K) / (3 * p.K + p.M)
            out['G'] = 3 * (p.M - p.K) / 4
            out['Lam'] = (3 * p.K - p.M) / 2
            out['v'] = (3 * p.K - p.M) / (3 * p.K + p.M)
        else:  # set_params[1]=='v'
            out['E'] = 3 * p.K * (1 - 2 * p.v)
            out['G'] = (3 * p.K * (1 - 2 * p.v)) / (2 * (1 + p.v))
            out['Lam'] = 3 * p.K * p.v / (1 + p.v)
            out['M'] = 3 * p.K * (1 - p.v) / (1 + p.v)
    elif set_params[0] == 'Lam':
        if set_params[1] == 'M':
            out['E'] = (p.M - p.Lam) * (p.M + 2 * p.Lam) / (p.M + p.Lam)
            out['G'] = (p.M - p.Lam) / 2
            out['K'] = (p.M + 2 * p.Lam) / 3
            out['v'] = p.Lam / (p.M + p.Lam)
        else:
            out['E'] = p.Lam * (1 + p.v) * (1 - 2 * p.v) / p.v
            out['G'] = p.Lam(1 - 2 * p.v) / (2 * p.v)
            out['K'] = p.Lam * (1 + p.v) / (3 * p.v)
            out['M'] = p.Lam * (1 - p.v) / p.v
    else:
        out['E'] = p.M * (1 + p.v) * (1 - 2 * p.v) / (1 - p.v)
        out['G'] = p.M * (1 - 2 * p.v) / (2 * (1 - p.v))
        out['K'] = p.M * (1 + p.v) / (3 * (1 - p.v))
        out['Lam'] = p.M * p.v / (1 - p.v)

    return out


# noinspection PyTypeChecker
def elastic_influence_matrix(comp: str, span: typing.Sequence[int], grid_spacing: typing.Sequence[float],
                             shear_mod: float, v: float,
                             shear_mod_2: typing.Optional[float] = None,
                             v_2: typing.Optional[float] = None) -> np.array:
    """Find influence matrix components for an elastic contact problem

    Parameters
    ----------
    comp : str {'xx','xy','xz','yx','yy','yz','zx','zy','zz'}
        The component to be returned
    span: Sequence[int]
        The span of the influence matrix in the x and y directions
    grid_spacing: Sequence[float]
        The grid spacings in the x and y directions
    shear_mod : float
        The shear modulus of the surface material
    v : float
        The Poission's ratio of the surface material
    shear_mod_2: float (optional) None
        The shear modulus of the second surface for a combined stiffness matrix
    v_2: float (optional) None
        The Poisson's ratio of the second surface for a combined stiffness matrix

    Returns
    -------
    C : array
        The influence matrix component requested

    See Also
    --------
    elastic_im

    Notes
    -----

    Don't use this function, used by: elastic_im

    References
    ----------
    Complete boundary element method formulation for normal and tangential
    contact problems

    """
    span = tuple(span)
    try:
        # lets just see how this changes
        # i'-i and j'-j
        idmi = (np.arange(span[1]) - span[1] // 2 + (1 - span[1] % 2))
        jdmj = (np.arange(span[0]) - span[0] // 2 + (1 - span[0] % 2))
        mesh_idmi = np.tile(idmi, (span[0], 1))
        mesh_jdmj = np.tile(np.expand_dims(jdmj, -1), (1, span[1]))

    except TypeError:
        raise TypeError("Span should be a tuple of integers")

    k = mesh_idmi + 0.5
    el = mesh_idmi - 0.5
    m = mesh_jdmj + 0.5
    n = mesh_jdmj - 0.5

    hy = grid_spacing[1]
    hx = grid_spacing[0]

    second_surface = (shear_mod_2 is not None) and (v_2 is not None)
    if not second_surface:
        v_2 = 1
        shear_mod_2 = 1

    if (shear_mod_2 is not None) != (v_2 is not None):
        raise ValueError('Either both or neither of the second surface parameters must be set')

    if comp == 'zz':
        c_zz = (hx * (k * np.log((m + np.sqrt(k ** 2 + m ** 2)) / (n + np.sqrt(k ** 2 + n ** 2))) +
                      el * np.log((n + np.sqrt(el ** 2 + n ** 2)) / (m + np.sqrt(el ** 2 + m ** 2)))) +
                hy * (m * np.log((k + np.sqrt(k ** 2 + m ** 2)) / (el + np.sqrt(el ** 2 + m ** 2))) +
                      n * np.log((el + np.sqrt(el ** 2 + n ** 2)) / (k + np.sqrt(k ** 2 + n ** 2)))))

        const = (1 - v) / (2 * np.pi * shear_mod) + second_surface * ((1 - v_2) / (2 * np.pi * shear_mod_2))
        return const * c_zz
    elif comp == 'xx':
        c_xx = (hx * (1 - v) * (k * np.log((m + np.sqrt(k ** 2 + m ** 2)) / (n + np.sqrt(k ** 2 + n ** 2))) +
                                el * np.log(
                (n + np.sqrt(el ** 2 + n ** 2)) / (m + np.sqrt(el ** 2 + m ** 2)))) +
                hy * (m * np.log((k + np.sqrt(k ** 2 + m ** 2)) / (el + np.sqrt(el ** 2 + m ** 2))) +
                      n * np.log((el + np.sqrt(el ** 2 + n ** 2)) / (k + np.sqrt(k ** 2 + n ** 2)))))
        const = 1 / (2 * np.pi * shear_mod) + second_surface * (1 / (2 * np.pi * shear_mod_2))
        return const * c_xx
    elif comp == 'yy':
        c_yy = (hx * (k * np.log((m + np.sqrt(k ** 2 + m ** 2)) / (n + np.sqrt(k ** 2 + n ** 2))) +
                      el * np.log((n + np.sqrt(el ** 2 + n ** 2)) / (m + np.sqrt(el ** 2 + m ** 2)))) +
                hy * (1 - v) * (m * np.log((k + np.sqrt(k ** 2 + m ** 2)) / (el + np.sqrt(el ** 2 + m ** 2))) +
                                n * np.log((el + np.sqrt(el ** 2 + n ** 2)) / (k + np.sqrt(k ** 2 + n ** 2)))))
        const = 1 / (2 * np.pi * shear_mod) + second_surface * (1 / (2 * np.pi * shear_mod_2))
        return const * c_yy
    elif comp in ['xz', 'zx']:
        c_xz = (hy / 2 * (m * np.log((k ** 2 + m ** 2) / (el ** 2 + m ** 2)) +
                          n * np.log((el ** 2 + n ** 2) / (k ** 2 + n ** 2))) +
                hx * (k * (np.arctan(m / k) - np.arctan(n / k)) +
                      el * (np.arctan(n / el) - np.arctan(m / el))))
        const = (2 * v - 1) / (4 * np.pi * shear_mod) + second_surface * (
            (2 * v_2 - 1) / (4 * np.pi * shear_mod_2))
        return const * c_xz
    elif comp in ['yx', 'xy']:
        c_yx = (np.sqrt(hy ** 2 * n ** 2 + hx ** 2 * k ** 2) -
                np.sqrt(hy ** 2 * m ** 2 + hx ** 2 * k ** 2) +
                np.sqrt(hy ** 2 * m ** 2 + hx ** 2 * el ** 2) -
                np.sqrt(hy ** 2 * n ** 2 + hx ** 2 * el ** 2))
        const = v / (2 * np.pi * shear_mod) + second_surface * (v_2 / (2 * np.pi * shear_mod_2))
        return const * c_yx
    elif comp in ['zy', 'yz']:
        c_zy = (hx / 2 * (k * np.log((k ** 2 + m ** 2) / (n ** 2 + k ** 2)) +
                          el * np.log((el ** 2 + n ** 2) / (m ** 2 + el ** 2))) +
                hy * (m * (np.arctan(k / m) - np.arctan(el / m)) +
                      n * (np.arctan(el / n) - np.arctan(k / n))))
        const = (1 - 2 * v) / (4 * np.pi * shear_mod) + second_surface * (1 - 2 * v_2) / (
            4 * np.pi * shear_mod_2)
        return const * c_zy
    else:
        ValueError('component name not recognised: ' + comp + ', components must be lower case')
