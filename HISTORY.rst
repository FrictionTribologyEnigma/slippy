History
=======

0.6.0 (2026-08-26)
------------------

* Support for python 3.12+, numpy 2.x and current scipy, numba and scikit-image; the numpy<1.21 pin is removed
* Packaging moved to pyproject.toml; pyfftw is now the optional [fftw] extra (needed for CPU contact solving)
* Continuous integration moved from Travis to GitHub Actions
* The ERROR_IF_MISSING_MODEL and ERROR_IF_MISSING_SUB_MODEL checks were inverted and never ran, they are now active
  by default: steps and sub models that mis-declare their outputs will raise (set the flags to False to suppress)
* Transient sub models no longer silently discard their results
* User supplied initial guesses for lubrication steps are now used on the first time step
* Cache derived brackets are now used by the quasi static load balancing optimiser
* Fixed the alicona file reader, the ProbFreqSurface generator, mask handling in roughness functions,
  divide-by-zero guards in gmres, and several mutation bugs
* Performance: vectorised random surface generation, cached interpolators, numba compilation caching

0.3.0 (2021-09-2)
-----------------

* Sub surface stress calculations
* Backend for coupled and multi convolutions
* Contact stiffness and rolling sliding submodels
* More efficient normal contact solver

0.1.1 (2020-11-9)
-----------------

* Mixed lubrication api brought into line with other steps
* bug fixes in mixed lubrication solver
* added just in time compilation to unified reynolds solver
* converted IterSemiSystem to quasi static

0.1.0 (2020-11-7)
-----------------

* First release on PyPI.
