import abc
from collections.abc import Sequence
import typing
import warnings
from itertools import product

import numpy as np
from scipy.signal import fftconvolve

from slippy.abcs import _MaterialABC
from ._material_utils import _get_properties, Loads, Displacements, memoize_components

__all__ = ["Elastic", "_Material", "Rigid", 'rigid']


# The base class for materials contains all the iteration functionality for contacts
class _Material(_MaterialABC):
    """ A class for describing material behaviour, the materials do the heavy lifting for the contact mechanics analysis
    """
    material_type: str
    name: str
    _subclass_registry = []

    def __init__(self, name: str):
        self.name = name
        self.material_type = self.__class__.__name__

    # keeps a registry of the materials
    @classmethod
    def __init_subclass__(cls, is_abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not is_abstract:
            _Material._subclass_registry.append(cls)

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
            The distacne between grid points of the parent surface
        components
            The required components of the influence matrix such as: ['xx', 'xy', 'xz'] which would be the components
            which relate loads in the x direction with displacements in each direction

        Returns
        -------
        dict of fourier transformed components

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
                                        deflections: str = 'xyz', span: typing.Optional[typing.Sequence[int]] = None,
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
        span: tuple
            The span of the influence matrix in grid points defaults to same as the loads span
        grid_spacing : tuple or float
            The grid spacing in each direction, if float it is assumed to be the same in each direction
        simple: bool optional (False)
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
            if type(loads) is dict:
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
        displacements = self._solve_im_loading(loads, components)
        return displacements

    @staticmethod
    def _solve_im_loading(loads, components) -> Displacements:
        """The meat of the elastic loading algorithm

        Parameters
        ----------
        loads : Loads
            Loads named tuple of N by M arrays of surface loads with fields 'x', 'y', 'z' or any combination
        components : dict
            Components of the influence matrix keys are 'xx', 'xy' ... 'zz' for
            each name the first character represents the displacement and the
            second represents the load, eg. 'xy' is the deflection in the x
            direction caused by a load in the y direction

        Returns
        -------
        displacements : Displacements
            namedtuple of N by M arrays of surface displacements with labels 'x', 'y',
            'z'

        See Also
        --------
        elastic_im
        elastic_loading

        Notes
        -----
        This function has very little error checking and may give unexpected
        results with bad inputs please use elastic_loading to check inputs
        before using this directly

        Components of the influence matrix can be found by elastic_im

        References
        ----------

        Complete boundry element formulation for normal and tangential contact
        problems

        """
        shape = [ld.shape for ld in loads if ld is not None][0]

        displacements = Displacements(np.zeros(shape), np.zeros(shape), np.zeros(shape))

        for c_name, component in components.items():
            load_name = c_name[1]
            dis_name = c_name[0]
            dis_comp = displacements.__getattribute__(dis_name)
            dis_comp += fftconvolve(loads.__getattribute__(load_name), component, mode='same')
        return displacements

    def loads_from_surface_displacement(self,
                                        displacements: typing.Union[dict, Displacements,
                                                                    typing.Sequence[typing.Optional[np.ndarray]]],
                                        grid_spacing: float,
                                        other: typing.Optional['_Material'] = None,
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
        span : int
            The span of the influence matrix in grid points defaults to same as the surface size
        grid_spacing : float
            The grid spacing only needed if surface is an array
        simple : bool, optional (True)
            If true only deflections in the directions of the loads are calculated, only the Cxx, Cyy and Czz components
            of the influence matrix are used
        max_it : int, optional (100)
            The maximum number of iterations before aborting the loop
        tol : float
            The tolerance on the iterations, the loop is ended when the norm of the residual is below tol
        other : _Material, optional (None)
            If supplied the problem will be solved on the combined material, the results will then be split by material

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

        Complete boundry element formulation for normal and tangential contact

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
            initial_guess = self.guess_loads_from_displacement(displacements, grid_spacing, components)
            loads, full_deflections = self._solve_im_displacement(displacements=displacements, components=components,
                                                                  max_it=max_it, tol=tol, initial_guess=initial_guess)
            return loads, (full_deflections,)

        # other is not None
        try:  # see if the other provides an influence matrix
            components_other = other.influence_matrix(grid_spacing=grid_spacing, span=span, components=comp_names)
            components_self = self.influence_matrix(grid_spacing=grid_spacing, span=span, components=comp_names)

            combined_components = dict()

            for comp in comp_names:
                combined_components[comp] = components_self[comp] + components_other[comp]
            # solve the problem
            initial_guess = self.guess_loads_from_displacement(displacements, grid_spacing, combined_components)
            loads, full_deflections = self._solve_im_displacement(displacements=displacements,
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

        except NotImplementedError:
            raise NotImplementedError("Not currently possible to solve the current material pair")

        # if you got here then this, the other surface or both have no influence matrix method, so we have to iterate
        # on the sum of the displacements using the displacement_from_surface_loads method directly
        # TODO

    @staticmethod
    def guess_loads_from_displacement(displacements: Displacements, grid_spacing: typing.Sequence,
                                      components: dict) -> Loads:
        """
        Defines the starting point for the default loads from displacement method

        This method should be overwritten for non IM based surfaces

        Parameters
        ----------
        displacements: Displacements
            The point wise displacement
        grid_spacing: float
            The spacing of the grid points
        components: dict
            Dict of influence ematrix components

        Returns
        -------
        guess_of_loads: Loads
            A named tuple of the loads
        """

        directions = 'xyz'

        loads = dict()
        for direction in directions:
            if displacements.__getattribute__(direction) is None:
                continue
            max_im = max(components[direction * 2].flatten())
            load = displacements.__getattribute__(direction) * max_im
            load = np.nan_to_num(load)
            loads[direction] = load

        return Loads(**loads)

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

    def _solve_im_displacement(self, displacements: Displacements, components: dict, max_it: int,
                               tol: float, initial_guess: Loads) -> typing.Tuple[Loads, Displacements]:
        """ Given displacements find the loads needed to cause then for an influence matrix based material

        Split away from the main function to allow for faster computation avoiding
        calculating the influence matrix for each iteration

        Parameters
        ----------
        displacements : Displacements
            The deflections at the surface with fields 'x', 'y', 'z', use float('nan') to signify free points (eg points
            out of the contact)
        components : dict
            Components of the influence matrix keys are 'xx', 'xy' ... 'zz' for
            each name the first character represents the displacement and the
            second represents the load, eg. 'xy' is the deflection in the x
            direction caused by a load in the y direction
        max_it : int
            The maximum number of iterations before breaking the loop
        tol : float
            The tolerance on the iterations, the loop is ended when the norm of
            the residual is below tol
        initial_guess : Loads
            The initial guess of the surface loads

        Returns
        -------
        loads : Loads
            The loads on the surface to cause the specified displacement with fields 'x' 'y' and 'z'
        full_displacement : Displacements
            The displacement results for the full surface/ problem

        See Also
        --------

        Notes
        -----

        References
        ----------

        Complete boundary element formulation for normal and tangential contact
        problems

        """
        valid_directions = 'xyz'
        def_directions = [vd for vd, el in zip(valid_directions, displacements) if el is not None]

        sub_domains = {dd: np.logical_not(np.isnan(displacements.__getattribute__(dd))) for dd in def_directions}
        sub_domain_sizes = {key: np.sum(value) for (key, value) in sub_domains.items()}

        # find first residual
        loads = initial_guess
        calc_displacements = self._solve_im_loading(loads, components)

        residual = np.array([])

        for dd in def_directions:
            residual = np.append(residual, displacements.__getattribute__(dd)[sub_domains[dd]] -
                                 calc_displacements.__getattribute__(dd)[sub_domains[dd]])

        # start loop
        search_direction = residual
        itnum = 0
        resid_norm = np.linalg.norm(residual)

        while resid_norm >= tol:
            # put calculated values back into right place

            start = 0
            search_direction_full = dict()
            for dd in def_directions:
                end = start + sub_domain_sizes[dd]
                search_direction_full[dd] = np.zeros_like(displacements.__getattribute__(dd))
                search_direction_full[dd][sub_domains[dd]] = search_direction[start:end]
                start = end

            # find z (equation 25 in ref):
            z_full = self._solve_im_loading(Loads(**search_direction_full), components)

            z = np.array([])
            for dd in def_directions:
                z = np.append(z, z_full.__getattribute__(dd)[sub_domains[dd]])

            # find alpha (equation 26)
            alpha = np.matmul(residual, residual) / np.matmul(search_direction, z)

            # update stresses (equation 27)
            update_vals = alpha * search_direction
            start = 0
            for dd in def_directions:
                end = start + sub_domain_sizes[dd]
                loads.__getattribute__(dd)[sub_domains[dd]] += update_vals[start:end]
                start = end

            r_new = residual - alpha * z  # equation 28

            # find new search direction (equation 29)
            beta = np.matmul(r_new, r_new) / np.matmul(residual, residual)
            residual = r_new
            resid_norm = np.linalg.norm(residual)

            search_direction = residual + beta * search_direction

            itnum += 1

            if itnum > max_it:
                msg = (f"Max itteration: ({max_it}) reached without convergence,"
                       f" residual was: {resid_norm}, convergence declared at: {tol}")
                warnings.warn(msg)
                break

        calc_displacements = self._solve_im_loading(loads, components)

        return loads, calc_displacements

    @memoize_components(False)
    def _jac_im_getter(self, component: str, surface_shape: tuple, periodic: bool, *im_args, **im_kwargs):
        """Get a single component of the jacobian matrix

        Parameters
        ----------
        component: str
            The desired component (should only be a single component eg 'xx')
        surface_shape: tuple
            The shape of the surface array, a 2 element tuple of ints
        periodic: bool
            If true the jacobian for a periodic surface is returned
        im_args, im_kwargs
            args and kwargs to be passed to the influence matrix method

        Returns
        -------
        jac_comp: np.array
            The requested component of the jacobian matrix

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


class Rigid(_Material):
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
        return Displacements(*[np.zeros_like(l) for l in loads])

    def __repr__(self):
        return "Rigid(" + self.name + ")"


rigid = Rigid('rigid')


# noinspection PyPep8Naming
class Elastic(_Material):
    """ A Class for defining elastic materials

    Parameters
    ----------
    name: str
        The name of the material
    properties: dict
        dict of properties, dicts must have exactly 2 items.
        Allowed keys are : 'E', 'v', 'G', 'K', 'M', 'Lam'
        See notes for definitions

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
    >>> # Make a material model for elastic steel on a rigid substarte with a
    >>> # thickness of 1mm
    >>> steel = Elastic({'E': 200e9, 'v': 0.3}, density=7850)
    >>> # Find it's pwave modulus:
    >>> pwm = steel.M
    >>> # Find the speeds of sound:
    >>> sos = steel.speed_of_sound()
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

    def __init__(self, name: str, properties: dict):
        super().__init__(name)

        if len(properties) > 2:
            raise ValueError("Too many properties suplied, must be 1 or 2")

        for item in properties.items():
            self._set_props(*item)

    def influence_matrix(self, grid_spacing: {typing.Sequence[float], float}, span: typing.Sequence[int],
                         components: typing.Union[typing.Sequence[str], str], other: _Material = None):
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
        other: _Material
            If supplied the combined modulus for the material pair will be returned, only works for pairs of elastic
            materials

        Returns
        -------
        dict
            dict of the requested influence matrix or matricies

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
        Complete boundry element method formulation for normal and tangential
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

        components = {comp: self._elastic_im_getter(comp, span, grid_spacing, shear_modulus, v,
                                                    shear_mod_2=shear_modulus_2, v_2=v_2) for comp in components}

        return components

    @staticmethod
    @memoize_components(True)
    def _elastic_im_getter(comp: str, span: typing.Sequence[int], grid_spacing: typing.Sequence[float],
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
            The shear modulus of the surface materuial
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
        Complete boundry element method formulation for normal and tangential
        contact problems

        """
        if any([not s % 2 for s in span]):
            warnings.warn('Even number of values requested in influence matrix, shape has been updated to one more than'
                          'requested along even dimensions')

        try:
            # lets just see how this changes
            # i'-i and j'-j
            idmi = (np.arange(span[1] + ~span[1] % 2) - int(span[1] / 2))
            jdmj = (np.arange(span[0] + ~span[0] % 2) - int(span[0] / 2))
            mesh_idmi, mesh_jdmj = np.meshgrid(idmi, jdmj)

        except TypeError:
            msg = "span should be a tuple of integers, {} found".format(
                type(span[0]))
            raise TypeError(msg)

        k = mesh_idmi + 0.5
        l = mesh_idmi - 0.5
        m = mesh_jdmj + 0.5
        n = mesh_jdmj - 0.5

        hx = grid_spacing[0]
        hy = grid_spacing[1]

        second_surface = (shear_mod_2 is not None) and (v_2 is not None)
        if not second_surface:
            v_2 = 1
            shear_mod_2 = 1

        if (shear_mod_2 is not None) != (v_2 is not None):
            raise ValueError('Either both or neither of the second surface parameters must be set')

        if comp == 'zz':
            c_zz = (hx * (k * np.log((m + np.sqrt(k ** 2 + m ** 2)) / (n + np.sqrt(k ** 2 + n ** 2))) +
                          l * np.log((n + np.sqrt(l ** 2 + n ** 2)) / (m + np.sqrt(l ** 2 + m ** 2)))) +
                    hy * (m * np.log((k + np.sqrt(k ** 2 + m ** 2)) / (l + np.sqrt(l ** 2 + m ** 2))) +
                          n * np.log((l + np.sqrt(l ** 2 + n ** 2)) / (k + np.sqrt(k ** 2 + n ** 2)))))

            const = (1 - v) / (2 * np.pi * shear_mod) + second_surface * ((1 - v_2) / (2 * np.pi * shear_mod_2))
            return const * c_zz
        elif comp == 'xx':
            c_xx = (hx * (1 - v) * (k * np.log((m + np.sqrt(k ** 2 + m ** 2)) / (n + np.sqrt(k ** 2 + n ** 2))) +
                                    l * np.log((n + np.sqrt(l ** 2 + n ** 2)) / (m + np.sqrt(l ** 2 + m ** 2)))) +
                    hy * (m * np.log((k + np.sqrt(k ** 2 + m ** 2)) / (l + np.sqrt(l ** 2 + m ** 2))) +
                          n * np.log((l + np.sqrt(l ** 2 + n ** 2)) / (k + np.sqrt(k ** 2 + n ** 2)))))
            const = 1 / (2 * np.pi * shear_mod) + second_surface * (1 / (2 * np.pi * shear_mod_2))
            return const * c_xx
        elif comp == 'yy':
            c_yy = (hx * (k * np.log((m + np.sqrt(k ** 2 + m ** 2)) / (n + np.sqrt(k ** 2 + n ** 2))) +
                          l * np.log((n + np.sqrt(l ** 2 + n ** 2)) / (m + np.sqrt(l ** 2 + m ** 2)))) +
                    hy * (1 - v) * (m * np.log((k + np.sqrt(k ** 2 + m ** 2)) / (l + np.sqrt(l ** 2 + m ** 2))) +
                                    n * np.log((l + np.sqrt(l ** 2 + n ** 2)) / (k + np.sqrt(k ** 2 + n ** 2)))))
            const = 1 / (2 * np.pi * shear_mod) + second_surface * (1 / (2 * np.pi * shear_mod_2))
            return const * c_yy
        elif comp in ['xz', 'zx']:
            c_xz = (hy / 2 * (m * np.log((k ** 2 + m ** 2) / (l ** 2 + m ** 2)) +
                              n * np.log((l ** 2 + n ** 2) / (k ** 2 + n ** 2))) +
                    hx * (k * (np.arctan(m / k) - np.arctan(n / k)) +
                          l * (np.arctan(n / l) - np.arctan(m / l))))
            const = (2 * v - 1) / (4 * np.pi * shear_mod) + second_surface * ((2 * v_2 - 1) / (4 * np.pi * shear_mod_2))
            return const * c_xz
        elif comp in ['yx', 'xy']:
            c_yx = (np.sqrt(hy ** 2 * n ** 2 + hx ** 2 * k ** 2) -
                    np.sqrt(hy ** 2 * m ** 2 + hx ** 2 * k ** 2) +
                    np.sqrt(hy ** 2 * m ** 2 + hx ** 2 * l ** 2) -
                    np.sqrt(hy ** 2 * n ** 2 + hx ** 2 * l ** 2))
            const = v / (2 * np.pi * shear_mod) + second_surface * (v_2 / (2 * np.pi * shear_mod_2))
            return const * c_yx
        elif comp in ['zy', 'yz']:
            c_zy = (hx / 2 * (k * np.log((k ** 2 + m ** 2) / (n ** 2 + k ** 2)) +
                              l * np.log((l ** 2 + n ** 2) / (m ** 2 + l ** 2))) +
                    hy * (m * (np.arctan(k / m) - np.arctan(l / m)) +
                          n * (np.arctan(l / n) - np.arctan(k / n))))
            const = (1 - 2 * v) / (4 * np.pi * shear_mod) + second_surface * (1 - 2 * v_2) / (4 * np.pi * shear_mod_2)
            return const * c_zy
        else:
            msg = ('component name not recognised: '
                   '{}, components are lower case'.format(comp))
            raise ValueError(msg)

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
