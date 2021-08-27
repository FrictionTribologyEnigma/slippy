import abc
import itertools
import typing
import slippy
import numpy as np
from collections.abc import Sequence
from .abcs import _MaterialABC
from .influence_matrix_utils import plan_convolve, plan_coupled_convolve, bccg
from ._material_utils import memoize_components

__all__ = ["_IMMaterial", "Rigid", "rigid"]


# The base class for materials contains all the iteration functionality for contacts
class _IMMaterial(_MaterialABC):
    """ A class for describing material behaviour, the materials do the heavy lifting for the contact mechanics analysis
    """
    material_type: str
    name: str
    _subclass_registry = []

    def __init__(self, name: str, max_load: float = np.inf):
        if name in slippy.material_names:
            raise ValueError(f"Materials must have unique names, currently in use names are: {slippy.material_names}")
        slippy.material_names.append(name)
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
    def _influence_matrix(self, components: typing.Sequence[str], grid_spacing: typing.Sequence[float],
                          span: typing.Sequence[int]):
        pass

    @memoize_components(False)
    def influence_matrix(self, components: typing.Sequence[str], grid_spacing: typing.Sequence[float],
                         span: typing.Sequence[int], periodic_strides=(1, 1)):
        """
        Find the influence matrix components for the material relating surface pressures to

        Parameters
        ----------
        span: Sequence[int]
            The span of the influence matrix (pts_in_y_direction, pts_in_x_direction)
        grid_spacing: float
            The distance between grid points of the parent surface
        components: Sequence[str]
            The required components of the influence matrix such as: ['xx', 'xy', 'xz'] which would be the components
            which relate loads in the x direction with displacements in each direction
        periodic_strides: Sequence[int]
            The influence matrix is wrapped this number of times to ensure results represent truly periodic behaviour.
            Both elements must be odd integers. For example if (1, 3) is given with (128, 128) span: a (128, 128*3) size
            influence matrix is calculated, 128 by 128 blocks are then summed to give the 128 by 128 result.


        Returns
        -------
        dict of components

        Notes
        -----
        """
        ps = []
        for s in periodic_strides:
            try:
                s = int(s)
            except ValueError:
                raise ValueError(f"Periodic strides must be odd integers, received: {s}")
            if not s % 2:
                raise ValueError(f"Periodic strides must be odd integers, received: {s}")
            ps.append(s)

        total_span = [sp*s for sp, s in zip(span, ps)]

        rtn_dict = self._influence_matrix(components, grid_spacing, total_span)
        for key in rtn_dict:
            original = rtn_dict[key]
            rtn_dict[key] = np.zeros(span, slippy.dtype)
            for i in range(periodic_strides[0]):
                for j in range(periodic_strides[1]):
                    rtn_dict[key] += original[i*span[0]:(i+1)*span[0], j*span[1]:(j+1)*span[1]]
        return rtn_dict

    @abc.abstractmethod
    def sss_influence_matrices_normal(self, components: typing.Sequence[str], grid_spacing: typing.Sequence[float],
                                      span: typing.Sequence[int], z: typing.Sequence[float] = None,
                                      cuda: bool = False) -> dict:
        """
        Optional, should give the sub surface stress influence matrix components

        Parameters
        ----------
        span: Sequence[int]
            The span of the influence matrix (pts_in_y_direction, pts_in_x_direction)
        grid_spacing
            The distance between grid points of the parent surface
        components
            The required components of the influence matrix such as: ['xx', 'xy', 'xz'] which would be components
            relating pressure in the z direction to the stress tensor terms xx, xy and xz. Shear stress terms should be
            in alphabetical order only; as in: s_xy, and not: s_yx.
        z: Sequence[float], optional (None)
            The depths of interest in the material, if none a grid of half the shortest dimension in the span is used.
        cuda: bool, optional (False)

        Returns
        -------
        dict of components with same keys as components arg

        Notes
        -----
        Do not memoize results from this method, there are more options in sub models etc, to allow users to control
        memory usage
        """
        pass

    @abc.abstractmethod
    def sss_influence_matrices_tangential_x(self, components: typing.Sequence[str],
                                            grid_spacing: typing.Sequence[float], span: typing.Sequence[int],
                                            z: typing.Sequence[float] = None, cuda: bool = False) -> dict:
        """
        Optional, should give the sub surface stress influence matrix components

        Parameters
        ----------
        span: Sequence[int]
            The span of the influence matrix (pts_in_y_direction, pts_in_x_direction)
        grid_spacing
            The distance between grid points of the parent surface
        components
            The required components of the influence matrix such as: ['xx', 'xy', 'xz'] which would be components
            relating traction in the x direction to the stress tensor terms xx, xy and xz. Shear stress terms should be
            in alphabetical order only; as in: xy, and not: yx.
        z: Sequence[float], optional (None)
            The depths of interest in the material, if none a grid of half the shortest dimension in the span is used.
        cuda: bool, optional (False)

        Returns
        -------
        dict of components with same keys as components arg

        Notes
        -----
        Do not memoize results from this method, there are more options in sub models etc, to allow users to control
        memory usage
        """
        pass

    def sss_influence_matrices_tangential_y(self, components: typing.Sequence[str],
                                            grid_spacing: typing.Sequence[float], span: typing.Sequence[int],
                                            z: typing.Sequence[float] = None, cuda: bool = False) -> dict:
        """
        Optional, overwrite only for non homogenous materials

        Parameters
        ----------
        span: Sequence[int]
            The span of the influence matrix (pts_in_y_direction, pts_in_x_direction)
        grid_spacing
            The distance between grid points of the parent surface
        components
            The required components of the influence matrix such as: ['xx', 'xy', 'xz'] which would be components
            relating traction in the x direction to the stress tensor terms xx, xy and xz. Shear stress terms should be
            in alphabetical order only; as in: xy, and not: yx.
        z: Sequence[float], optional (None)
            The depths of interest in the material, if none a grid of half the shortest dimension in the span is used.
        cuda: bool, optional (False)

        Returns
        -------
        dict of components with same keys as components arg

        Notes
        -----
        Do not memoize results from this method, there are more options in sub models etc, to allow users to control
        memory usage
        """
        span = tuple(reversed(span))
        grid_spacing = tuple(reversed(grid_spacing))
        replace = {'x': 'y', 'y': 'x', 'z': 'z'}
        new_comps = [''.join(sorted(replace[comp[0]] + replace[comp[1]])) for comp in components]
        comps = self.sss_influence_matrices_tangential_x(new_comps, grid_spacing, span, z, cuda)
        rtn_dict = dict()
        for key, comp in comps.items():
            new_key = components[new_comps.index(key)]
            rtn_dict[new_key] = comp.swapaxes(1, 2)
        return rtn_dict

    def displacement_from_surface_loads(self, loads: dict,
                                        grid_spacing: float,
                                        deflections: str = 'xyz',
                                        simple: bool = True,
                                        periodic_axes: typing.Sequence[bool] = (True, True)):
        load_dirs = [key for key in loads if key in 'xyz']
        load_dirs.sort()
        shape = loads[load_dirs[0]].shape
        if isinstance(grid_spacing, Sequence):
            if len(grid_spacing) < 2:
                grid_spacing = (grid_spacing, grid_spacing)
            elif len(grid_spacing) > 2:
                raise ValueError("Grid spacing should be a number or a two element sequence of numbers")
        else:
            grid_spacing = (grid_spacing, grid_spacing)

        if simple:
            component_names = [2 * dir for dir in load_dirs]
        else:
            component_names = list(a + b for a, b in itertools.product(load_dirs, deflections))
        components = self.influence_matrix(component_names, grid_spacing, shape)

        conv_func = plan_coupled_convolve(loads, components, None, periodic_axes)

        return conv_func(loads)

    def loads_from_surface_displacement(self,
                                        displacements: dict,
                                        grid_spacing: float,
                                        other: typing.Optional[_MaterialABC] = None,
                                        tol: float = 1e-8,
                                        simple: bool = True,
                                        max_it: int = None,
                                        periodic_axes: typing.Sequence[bool] = (True, True)):

        load_dirs = [key for key in displacements if key in 'xyz']
        load_dirs.sort()
        shape = displacements[load_dirs[0]].shape
        size = displacements[load_dirs[0]].size

        if isinstance(grid_spacing, Sequence):
            if len(grid_spacing) < 2:
                grid_spacing = (grid_spacing, grid_spacing)
            elif len(grid_spacing) > 2:
                raise ValueError("Grid spacing should be a number or a two element sequence of numbers")
        else:
            grid_spacing = (grid_spacing, grid_spacing)

        if simple:
            component_names = [2 * d for d in load_dirs]
        else:
            component_names = list(a + b for a, b in itertools.product(load_dirs, load_dirs))

        components = self.influence_matrix(component_names, grid_spacing, shape)

        domain = slippy.xp.ones(displacements[load_dirs[0]].shape, dtype=bool)

        if len(component_names) == 1:
            load_dir = load_dirs[0]
            conv_func = plan_convolve(displacements[load_dir], components[load_dir * 2], domain, periodic_axes)
            b = displacements[load_dir].flatten()
        else:
            load_dir = None
            conv_func = plan_coupled_convolve(displacements, components, domain, periodic_axes)
            b = np.concatenate(tuple(displacements[d].flatten() for d in load_dirs))

        loads, failed = bccg(conv_func, b, tol=tol, max_it=max_it, x0=np.zeros_like(b), min_pressure=-np.inf)
        full_loads = dict()
        for i in range(len(load_dirs)):
            full_loads[load_dirs[i]] = slippy.xp.zeros(shape)
            full_loads[load_dirs[i]][domain] = loads[i * size:(i + 1) * size]

        return full_loads


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

    def _influence_matrix(self, components: typing.Sequence[str], grid_spacing: typing.Sequence[float],
                          span: typing.Sequence[int] = (1, 1)):
        return {comp: np.zeros(span) for comp in components}

    def displacement_from_surface_loads(self, loads, *args, **kwargs):
        return [np.zeros_like(l) for l in loads]  # noqa: E741

    def sss_influence_matrices_normal(self, components: typing.Sequence[str], grid_spacing: typing.Sequence[float],
                                      span: typing.Sequence[int], z: typing.Sequence[float] = None,
                                      cuda: bool = False) -> dict:
        raise NotImplementedError("Subsurface stresses are not implemented for rigid materials")

    def sss_influence_matrices_tangential_x(self, components: typing.Sequence[str],
                                            grid_spacing: typing.Sequence[float], span: typing.Sequence[int],
                                            z: typing.Sequence[float] = None, cuda: bool = False) -> dict:
        raise NotImplementedError("Subsurface stresses are not implemented for rigid materials")

    def __repr__(self):
        return "Rigid(" + self.name + ")"


rigid = Rigid('rigid')
