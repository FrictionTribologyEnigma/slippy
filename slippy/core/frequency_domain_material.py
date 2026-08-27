import abc
import typing
import numpy as np

from .materials import _IMMaterial
from .elastic_material import get_angular_velocity

__all__ = ['_FrequencyDomainMaterial']


class _FrequencyDomainMaterial(_IMMaterial, is_abstract=True):
    """Base class for materials defined by a frequency response function (FRF)

    This is the recommended starting point for new material models. Any material whose surface
    response to a distributed load can be written as a static frequency response function
    C(q) relating the fourier transforms of pressure and surface displacement:

        u(q) = C(q) * p(q)

    only needs to implement the _frf method, which evaluates C on a grid of wavenumbers, for
    each requested component. Everything else (solvers, two body contacts, memoization of the
    influence matrix, the elastic-perfectly-plastic max_load cap and the GPU backend) is
    inherited.

    For reference the frequency response function of an isotropic elastic half space for the
    'zz' (normal load, normal displacement) component is 2/(E* |q|).

    Parameters
    ----------
    name: str
        The name of the material, must be unique
    max_load: float, optional (inf)
        The maximum load supported, loads above this will cause perfectly plastic deformation
        in the solvers which support it
    periodic_im_repeats: tuple, optional (1, 1)
        The influence matrix repeats used for periodic simulations, see _IMMaterial
    zero_frequency_value: float or dict, optional (None)
        The value of the zero frequency (DC) component of the FRF. Most half space kernels are
        singular at q = 0 and the fully periodic convention used here sets the DC term to 0:
        the mean pressure then produces no displacement and the mean surface level is carried
        by the rigid body interference. (Note the Elastic material instead defaults its DC
        term to the sum of its truncated spatial kernel, a small finite value.) Materials with
        a genuinely finite compliance at q = 0 (for example a bonded layer on a rigid base)
        should pass the analytical q -> 0 limit here, either as a float or as a per-component
        dict.

    Notes
    -----
    Influence matrices are memoized by material name and grid arguments: all parameters that
    the FRF depends on must be treated as immutable after construction. If a parameter must
    change, make a new material instance (with a new name) instead.

    In a two body contact the solvers add the influence matrices of both materials; a DC=0
    material paired with a default Elastic body mixes the two DC conventions. Contact results
    (pressures, contact area) are unaffected, but the reported interference contains the
    corresponding constant offset.

    Subclasses of this class cannot add a spatial influence matrix implementation: the class
    creation hook has already marked the spatial path unavailable. Subclass _IMMaterial
    directly if both domains are needed.
    """

    def __init__(self, name: str, max_load: float = np.inf, periodic_im_repeats: tuple = (1, 1),
                 zero_frequency_value=None):
        super().__init__(name, default_fft=True, max_load=max_load, periodic_im_repeats=periodic_im_repeats,
                         zero_frequency_value=zero_frequency_value)

    @abc.abstractmethod
    def _frf(self, components: typing.Sequence[str], q_y: np.ndarray, q_x: np.ndarray,
             q_norm: np.ndarray) -> dict:
        """Evaluate the frequency response function on a wavenumber grid

        Parameters
        ----------
        components: Sequence[str]
            The requested components, named as load direction then displacement direction, e.g.
            'zz' is the normal displacement caused by a normal load. Normal contact solvers only
            request 'zz'. Raise ValueError for unsupported components.
        q_y, q_x: np.ndarray
            The angular wavenumber grids in the y and x directions (rad / length)
        q_norm: np.ndarray
            The euclidean norm of the wavenumber vectors, zero at the [0, 0] element

        Returns
        -------
        dict of np.ndarray
            One array (same shape as q_norm) per requested component. The [0, 0] (DC) element
            may be left inf/nan, it is overwritten with zero_frequency_value by the base class.
        """
        raise NotImplementedError

    def _influence_matrix_frequency(self, components: typing.Sequence[str],
                                    grid_spacing: typing.Sequence[float], span: typing.Sequence[int]):
        q_x, q_y = get_angular_velocity(span, grid_spacing)
        q_norm = np.sqrt(q_x ** 2 + q_y ** 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            rtn = self._frf(components, q_y, q_x, q_norm)
        for comp in rtn:
            # the DC term is patched with zero_frequency_value by _IMMaterial.influence_matrix
            rtn[comp][0, 0] = 0.0
        return rtn

    def sss_influence_matrices_normal(self, components: typing.Sequence[str], grid_spacing: typing.Sequence[float],
                                      span: typing.Sequence[int], z: typing.Sequence[float] = None,
                                      cuda: bool = False) -> dict:
        raise NotImplementedError(f"Sub surface stresses are not implemented for material type: "
                                  f"{self.__class__.__name__}")

    def sss_influence_matrices_tangential_x(self, components: typing.Sequence[str],
                                            grid_spacing: typing.Sequence[float], span: typing.Sequence[int],
                                            z: typing.Sequence[float] = None, cuda: bool = False) -> dict:
        raise NotImplementedError(f"Sub surface stresses are not implemented for material type: "
                                  f"{self.__class__.__name__}")
