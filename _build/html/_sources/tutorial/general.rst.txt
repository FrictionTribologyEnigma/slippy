Introduction
============

SlipPy is a collection of surface analysis, generation and modelling algorithims. The aim of this project is to bring contact mechanics and other models in tribology to a wider audience and provide a high level interface to these models. 

This project is based on python which is a large open source programming language. Within python there are thousands of packages which bring convienient classes and functions to the programmer. At first keeping track of these packages and their different versions can be daunting, those familiar with matlab or labview may see this as an unnecessary difficulty. For people new to python we recommend installing an anaconda distribution and learning about conda virtual environments.

This tutorial will acquaint the first time SlipPY user with some of the most important features. It assumes the user has already installed the SlipPY package. 

Throughout this tutorial we will often assume that numpy and pyplot have been imported as follows::

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

SlipPy Organisation
===================

SlipPy is split into sub packages covering different functionallity.

============== ====================================
Subpackage     Description
============== ====================================
'surface'      Surface analysis and generation
'contact'      Contact mechanics models
'lubrication'  Lubrication models
'friction'     Friction models
'tribosonics'  Models and functions for tribosonics
============== ====================================

These need to be imported separately, for example::

    >>> from slippy import surface

Finding documentation
=====================

While the online documentaion should cover most aspects of the code this is not garanteed especially for non core functionallity. The online documention is generated from the docstrings of each object. These can be displayed for any of the objects by using the numpy info function::

    >>> np.info(surface.roughness)

The source code is avalible online (and on your compute) but can also be accesed on the fly with the numpy source funciton::

    >>> np.source(surface.subtract_polynomial)

