import abc
from collections.abc import Sequence
import typing
from itertools import product
import numpy as np

from slippy.abcs import _MaterialABC
from .influence_matrix_utils import _solve_im_loading, _solve_im_displacement, elastic_influence_matrix, \
    guess_loads_from_displacement
from ._material_utils import _get_properties, Loads, Displacements, memoize_components

__all__ = ["Elastic", "_IMMaterial", "Rigid", 'rigid', 'elastic_influence_matrix']


# The base class for materials contains all the iteration functionality for contacts
class _IMMaterial(_MaterialABC):
    """ A class for describing material behaviour, the materials do the heavy lifting for the contact mechanics analysis
    """
    material_type: str
    name: str
    _subclass_registry = []

    def __init__(self, name: str, max_load: float = np.inf):
        self.name = name
        self.material_type = self.__class__.__name__
        self.max_load = max_load

    # keeps a registry of the materials
    @classmethod
    def __init_subclass__(cls, is_abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not is_abstract:
            _IMMaterial._subclass_registry.append(cls)

    # Each material must define an influence matrix method that given the grid spacing and the span returns the
    # influence matrix

    # should memoize the results so that the deflection from loads method can be called directly
    @abc.abstractmethod
    def influence_matrix(self, span: typing.Sequence[int], grid_spacing: typing.Sequence[float],
                         components: typing.Sequence[str]):
        """
        Find the influence matrix components for the material

        Parameters
        ----------
        span: Sequence[int]
            The span of the influence matrix (pts_in_x_direction, pts_in_y_direction)
        grid_spacing
            The distance between grid points of the parent surface
        components
            The required components of the influence matrix such as: ['xx', 'xy', 'xz'] which would be the components
            which relate loads in the x direction with displacements in each direction

        Returns
        -------
        dict of components

        Notes
        -----
        If overloaded the results should be memoised as variables in the class, as this function is called frequently
        during an analysis
        """
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass

    def displacement_from_surface_loads(self, loads: {dict, Loads, typing.Sequence[np.ndarray]},
                                        grid_spacing: {typing.Sequence[float], float},
                                        deflections: str = 'xyz',
                                        current_state: dict = None,
                                        span: typing.Optional[typing.Sequence[int]] = None,
                                        simple: bool = False) -> Displacements:
        """Find surface displacements from a set of loads on an elastic half-space

        Parameters
        ----------
        loads: {dict, Loads, Sequence[np.ndarray]}
            Loads namedtuple of loads with fields 'x' 'y' or 'z', dict with keys 'x', 'y', 'z' or sequence of np.arrays
            in order x,y,z, must be length 3. Each field should be a np.array of surface loads at each grid point or
            None to specify no load on in that direction
        deflections: str {'xyz', 'x', 'y', 'z' or any combination}
            The components of the surface deflections to be calculated
        current_state : dict, optional (None)
            The current state of the contact model
        span: tuple
            The span of the influence matrix in grid points defaults to same as the loads span
        grid_spacing : tuple or float
            The grid spacing in each direction, if float it is assumed to be the same in each direction
        simple: bool, optional (False)
            If true only deflections in the directions of the loads are calculated,
            only the Cxx, Cyy and Czz components of the influence matrix are used

        Returns
        -------
        displacements : Displacements
            named tuple of surface displacements

        See Also
        --------
        elastic_im
        elastic_displacement

        Notes
        -----
        If the other material is supplied and is elastic the problem will be solved using the combined influence matrix,
        the displacements that are returned are the total displacements for both surfaces.

        For custom materials that do not use influence matrices this method should be over ridden

        Examples
        --------

        """
        if not isinstance(loads, Loads):
            if isinstance(loads, dict):
                try:
                    loads = {key: value for key, value in loads if value is not None}
                    loads = Loads(**{key: np.array(value, dtype=float) for key, value in loads.items()})
                except TypeError:
                    raise ValueError("Unrecognised keys in loads dict, allowed keys are x,y,z found keys are: "
                                     f"{', '.join(loads.keys())}")
            elif isinstance(loads, Sequence):
                try:
                    loads = Loads(*[np.array(ld, dtype=float) if ld is not None else None for ld in loads])
                except TypeError:
                    raise ValueError(f"length of the loads sequence must be 3, length is {len(loads)}")

        valid_directions = 'xyz'
        load_directions = [vd for vd, el in zip(valid_directions, loads) if el is not None]

        shapes = [ld.shape for ld in loads if ld is not None]

        if len(set(shapes)) != 1:
            raise ValueError("Load vectors are not all the same shape")

        if span is None:
            span = shapes[0]

        if not isinstance(grid_spacing, Sequence):
            grid_spacing = (grid_spacing,) * 2
        elif len(grid_spacing) == 1:
            grid_spacing *= 2
        elif len(grid_spacing) > 2:
            raise ValueError("Too many elements in grid_spacing sequence, should be 1 or 2")

        # u_a=ifft(fft(K_ab) * fft(sigma_b))

        if simple:
            comp_names = [ld * 2 for ld in load_directions]  # ['xx', 'yy'] etc.
        else:
            comp_names = list(product(deflections, load_directions))
            comp_names = [a + b for a, b in comp_names]  # 'xy'...

        # get the influence matrix components from the class methods

        components = self.influence_matrix(span=span, grid_spacing=grid_spacing, components=comp_names)
        displacements = _solve_im_loading(loads, components)
        return displacements

    def loads_from_surface_displacement(self,
                                        displacements: typing.Union[dict, Displacements,
                                                                    typing.Sequence[typing.Optional[np.ndarray]]],
                                        grid_spacing: float,
                                        other: typing.Optional[_MaterialABC] = None,
                                        current_state: dict = None,
                                        span: typing.Sequence[int] = None,
                                        tol: float = 1e-8,
                                        simple: bool = True,
                                        max_it: int = 100):
        """
        Find surface loading from surface displacements for an influence matrix based material by the conjugate gradient
        decent technique

        Parameters
        ----------
        displacements : dict, Displacements, Sequence[np.ndarray]
            dict of arrays of deflections with keys 'x' 'y' or 'z' allowed, or Displacements named tuple, or sequence of
            length 3 of numpy arrays or None. The surface displacements at each grid point to be solved for.
        grid_spacing : float
            The grid spacing only needed if surface is an array
        other : _Material, optional (None)
            If supplied the problem will be solved on the combined material, the results will then be split by material
        current_state : dict, optional (None)
            The current state of the contact model
        span : int
            The span of the influence matrix in grid points defaults to same as the surface size

        simple : bool, optional (True)
            If true only deflections in the directions of the loads are calculated, only the Cxx, Cyy and Czz components
            of the influence matrix are used
        max_it : int, optional (100)
            The maximum number of iterations before aborting the loop
        tol : float
            The tolerance on the iterations, the loop is ended when the norm of the residual is below tol

        Returns
        -------
        Loads : Loads (namedtuple)
            Loads object of arrays of surface loads with fields' 'y' and 'z', if simple is True loads will only be
            defined in the same direction as the displacements were specified, in other directions the filed will be
            None
        displacements : tuple
            Tuple of Displacement named tuples the first element is the total deflection, the remaining 2 are for this
            surface and the other surface (if the other surface is given)

        See Also
        --------


        Notes
        -----

        Examples
        --------

        References
        ---------

        Complete boundary element formulation for normal and tangential contact

        """
        if not isinstance(displacements, Displacements):
            if type(displacements) is dict:
                try:
                    displacements = Displacements(**{key: np.array(value, dtype=float) for key, value in
                                                     displacements.items() if value is not None})
                except TypeError:
                    # noinspection PyTypeChecker
                    raise ValueError(
                        f'Unexpected key in displacements dict, valid keys are "x", "y", "z", found keys are: '
                        f'{", ".join(displacements.keys())}')
            elif isinstance(displacements, Sequence):
                if len(displacements) != 3:
                    raise ValueError(f"Deflections sequence must be length 3, length is: {len(displacements)}")
                displacements = Displacements(*[np.array(de, dtype=float) for de in displacements if de is not None])

        shapes = [el.shape for el in displacements if el is not None]
        valid_directions = 'xyz'
        def_directions = [vd for vd, el in zip(valid_directions, displacements) if el is not None]

        if len(set(shapes)) != 1:
            raise ValueError("Deflection vectors are not all the same shape, or no displacements are specified")

        if span is None:
            span = shapes[0]

        if not isinstance(grid_spacing, Sequence):
            grid_spacing = (grid_spacing,) * 2

        if len(grid_spacing) == 1:
            grid_spacing *= 2
        elif len(grid_spacing) > 2:
            raise ValueError("Too many elements in grid_spacing, should be 1 or 2")

        if simple or len(def_directions) == 1:
            comp_names = [dd * 2 for dd in def_directions]  # ['xx','yy' etc.] if the are set directions
        else:
            comp_names = list(product(displacements, valid_directions))
            comp_names = [a + b for a, b in comp_names]

        # get components of the influence matrix
        if other is None:
            components = self.influence_matrix(grid_spacing=grid_spacing, span=span, components=comp_names)
            initial_guess = guess_loads_from_displacement(displacements, components)
            loads, full_deflections = _solve_im_displacement(displacements=displacements, components=components,
                                                             max_it=max_it, tol=tol, initial_guess=initial_guess)
            return loads, (full_deflections,)

        # other is not None
        elif isinstance(other, _IMMaterial):
            components_other = other.influence_matrix(grid_spacing=grid_spacing, span=span, components=comp_names)
            components_self = self.influence_matrix(grid_spacing=grid_spacing, span=span, components=comp_names)

            combined_components = dict()

            for comp in comp_names:
                combined_components[comp] = components_self[comp] + components_other[comp]
            # solve the problem
            initial_guess = guess_loads_from_displacement(displacements, combined_components)
            loads, full_deflections = _solve_im_displacement(displacements=displacements,
                                                             components=combined_components,
                                                             max_it=max_it, tol=tol, initial_guess=initial_guess)
            # split up the deflections by surface
            deflections_1 = self.displacement_from_surface_loads(loads=loads, grid_spacing=grid_spacing,
                                                                 deflections=''.join(def_directions), simple=simple,
                                                                 span=span)
            deflections_2 = other.displacement_from_surface_loads(loads=loads, grid_spacing=grid_spacing,
                                                                  deflections=''.join(def_directions), simple=simple,
                                                                  span=span)

            return loads, (full_deflections, deflections_1, deflections_2)

        else:
            raise NotImplementedError("Not currently possible to solve the current material pair")

        # if you got here then this, the other surface or both have no influence matrix method, so we have to iterate
        # on the sum of the displacements using the displacement_from_surface_loads method directly
        # TODO

    def _solve_general_displacement(self, other, displacements, components, max_it, tol, simple, span, grid_spacing):
        """
        Solves the general case, where only displacement_from_surface_loads is implemented for one of the surfaces

        Parameters
        ----------
        other
        displacements
        components
        max_it
        tol
        simple
        span
        grid_spacing

        Returns
        -------

        """
        pass

    @memoize_components(False)
    def _jac_im_getter(self, component: str, surface_shape: tuple, periodic: bool, *im_args, **im_kwargs):
        """Get a single component of the Jacobian matrix

        Parameters
        ----------
        component: str
            The desired component (should only be a single component eg 'xx')
        surface_shape: tuple
            The shape of the surface array, a 2 element tuple of ints
        periodic: bool
            If true the Jacobian for a periodic surface is returned
        im_args, im_kwargs
            args and kwargs to be passed to the influence matrix method

        Returns
        -------
        jac_comp: np.array
            The requested component of the Jacobian matrix

        """

        inf_comp = self.influence_matrix(component, *im_args, **im_kwargs)[component]
        influence_martix_span = inf_comp.shape
        if periodic:
            # check that the surface shape is odd in both dimensions
            if not all([el % 2 for el in surface_shape]):
                raise ValueError("Surface shape must be odd in both dimensions for periodic surfaces")
            # trim the influence matrix if necessary
            dif = [int((ims - ss) / 2) for ims, ss in zip(influence_martix_span, surface_shape)]
            if dif[0] > 0:
                inf_comp = inf_comp[dif[0]:-1 * dif[0], :]
            if dif[1] > 0:
                inf_comp = inf_comp[:, dif[1]:-1 * dif[1]]
            trimmed_ims = inf_comp.shape
            # pad to the same shape as the surface (this is why it has to be odd size)
            inf_mat = np.pad(inf_comp, ((0, surface_shape[0] - trimmed_ims[0]),
                                        (0, surface_shape[1] - trimmed_ims[1])), mode='constant')
            inf_mat = np.roll(inf_mat, (-1 * int(trimmed_ims[0] / 2), -1 * int(trimmed_ims[1] / 2)),
                              axis=[0, 1]).flatten()
            jac_comp = []
            roll_num = 0
            # roll the influence matrix to fill in rows of the jacobian
            for n in range(surface_shape[0]):
                for m in range(surface_shape[1]):
                    jac_comp.append(np.roll(inf_mat, roll_num))
                    roll_num += 1
            jac_comp = np.asarray(jac_comp)

        else:  # not periodic
            pad_0 = int(surface_shape[0] - np.floor(influence_martix_span[0] / 2))
            pad_1 = int(surface_shape[1] - np.floor(influence_martix_span[1] / 2))
            if pad_0 < 0:
                inf_comp = inf_comp[-1 * pad_0:pad_0, :]
                pad_0 = 0
            if pad_1 < 0:
                inf_comp = inf_comp[:, -1 * pad_1:pad_1]
                pad_1 = 0
            inf_mat = np.pad(inf_comp, ((pad_0, pad_0), (pad_1, pad_1)), mode='constant')
            jac_comp = []
            idx_0 = 0
            for n in range(surface_shape[0]):
                idx_1 = 0
                for m in range(surface_shape[1]):
                    jac_comp.append(inf_mat[surface_shape[0] - idx_0:2 * surface_shape[0] - idx_0,
                                    surface_shape[1] - idx_1:2 * surface_shape[1] - idx_1].copy().flatten())
                    idx_1 += 1
                idx_0 += 1
            jac_comp = np.asarray(jac_comp)

        return jac_comp


class Rigid(_IMMaterial):
    """ A rigid material

    Parameters
    ----------
    name: str
        The name of the material
    """
    material_type = 'Rigid'

    E = None
    v = None
    G = None
    lam = None
    K = None
    M = None

    def __init__(self, name: str):
        super().__init__(name)

    def influence_matrix(self, span: typing.Sequence[int], grid_spacing: typing.Sequence[float],
                         components: typing.Sequence[str]):
        return {comp: np.zeros(span) for comp in components}

    def displacement_from_surface_loads(self, loads, *args, **kwargs):
        return Displacements(*[np.zeros_like(l) for l in loads])  # noqa: E741

    def __repr__(self):
        return "Rigid(" + self.name + ")"


rigid = Rigid('rigid')


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

    def influence_matrix(self, span: typing.Sequence[int], grid_spacing: {typing.Sequence[float], float},
                         components: typing.Union[typing.Sequence[str], str], other: _IMMaterial = None):
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
        other: _IMMaterial
            If supplied the combined modulus for the material pair will be returned, only works for pairs of elastic
            materials

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
        span is automatically rounded up to the next odd number to ensure
        symmetrical results

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

        If both shear_modulus_2 and v_2 are supplied and are not None the combined modulus is returns for the surface
        pair

        Examples
        --------


        References
        ----------
        Complete boundary element method formulation for normal and tangential
        contact problems

        """

        if other is not None:
            if isinstance(other, Elastic) or isinstance(other, Rigid):
                shear_modulus_2 = other.G
                v_2 = other.v
            else:
                raise NotImplementedError("Combined influence matrix cannot be found for this material pair")
        else:
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

    def __repr__(self):
        return "Elastic(name = '" + self.name + f"', properties = {{ 'E':{self.E}, 'v':{self.v} }}"
