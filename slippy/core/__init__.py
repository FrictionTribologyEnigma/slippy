"""
Minimal abstract base classes and core functionality to avoid circular imports while type checking
"""

from .abcs import (_SurfaceABC, _AdhesionModelABC, _MaterialABC, _StepABC, _FrictionModelABC, _WearModelABC,
                   _ACFABC, _LubricantModelABC, _ContactModelABC, _ReynoldsSolverABC,
                   _NonDimensionalReynoldSolverABC, _SubModelABC)
from .materials import _IMMaterial, Rigid, rigid
from .elastic_material import Elastic, elastic_influence_matrix_spatial, elastic_influence_matrix_frequency, \
    get_angular_velocity
from .influence_matrix_utils import bccg, plan_convolve, guess_loads_from_displacement, plan_multi_convolve, \
    plan_coupled_convolve, polonsky_and_keer, rey, ConvolutionFunction
from .outputs import OutputReader, OutputRequest, OutputSaver, read_output
from ._stress_utils import get_derived_stresses, solve_cubic
from .gmres import gmres


__all__ = ['_SurfaceABC', '_AdhesionModelABC', '_MaterialABC', '_StepABC', '_FrictionModelABC', '_WearModelABC',
           '_ACFABC', '_LubricantModelABC', '_ContactModelABC', '_ReynoldsSolverABC',
           '_NonDimensionalReynoldSolverABC', '_SubModelABC', '_IMMaterial', 'Rigid', 'rigid', 'Elastic',
           'bccg', 'plan_convolve', 'OutputReader', 'OutputRequest', 'OutputSaver', 'guess_loads_from_displacement',
           'elastic_influence_matrix_spatial', 'elastic_influence_matrix_frequency', 'get_angular_velocity',
           'read_output', 'plan_multi_convolve', 'plan_coupled_convolve', 'ConvolutionFunction', 'rey',
           'get_derived_stresses', 'polonsky_and_keer', 'gmres', 'solve_cubic']
