import typing
from .materials import _IMMaterial


class SurfaceTensedMaterial(_IMMaterial):

    def __init__(self, name, modulus, p_ratio, tau_0, max_load, im_intergration_error=1e-8):
        super().__init__(name, max_load)
        self.e_star = modulus / (1 - p_ratio ** 2)
        self.s = 2 * tau_0 / self.e_star
        self.int_tol = im_intergration_error

    def _influence_matrix(self, components: typing.Sequence[str], grid_spacing: typing.Sequence[float],
                          span: typing.Sequence[int]):
        if len(components) > 1 or 'zz' not in components:
            raise ValueError("Only normal loading is implemented for surface tensed materials")

    def sss_influence_matrices_normal(self, components: typing.Sequence[str], grid_spacing: typing.Sequence[float],
                                      span: typing.Sequence[int], z: typing.Sequence[float] = None,
                                      cuda: bool = False) -> dict:

        raise NotImplementedError("Sub surface stresses are not implemented for this material")

    def sss_influence_matrices_tangential_x(self, components: typing.Sequence[str],
                                            grid_spacing: typing.Sequence[float], span: typing.Sequence[int],
                                            z: typing.Sequence[float] = None, cuda: bool = False) -> dict:

        raise NotImplementedError("Sub surface stresses are not implemented for this material")
