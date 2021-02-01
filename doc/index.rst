SlipPy
======

.. |SlipPY| image:: logo.svg
        :target: https://github.com/FrictionTribologyEnigma/slippy
        :alt: Slippy

|SlipPY|

SlipPy is an open source python package for tribologists. We aim to bring both basic and cutting edge trilogy models to the wider community.

This page contains useful links for installation, documentation, examples and a development guide. For a basic overview of what slippy can do, have a look at the examples.

Installation
------------

The following page describes how to install slippy, it assumes that the user does not have python insalled.

.. toctree::
   :maxdepth: 1

   installation

Usage
-----
Slippy provides access to solvers and code for simple tribology tasks such as analysing a surface or analytically solving simple contacts.
It also provides a more general interface to numerical solvers for contact problems, predominantly boundary element methods based on the half space approximation.
These numerical solvers can be accessed by making a contact model object, the link below gives detailed information of contact model objects and how they work.

.. toctree::
   :maxdepth: 1

   contact_models

Slippy also provides explained examples of common tasks, including making contact models, and using other functionality.

.. toctree::
   :maxdepth: 1

   examples

The exact API of all functions and classes, as given by the docstrings. These pages detail exactly how to use each function and class in slippy.
If you are having trouble with a particular function this is where to look. If you don't know how to get started with a project try the pages above.

.. toctree::
   :maxdepth: 1

   surface
   contact

Extending slippy
----------------
Slippy is built to be easy to extend, if there is functionality you need which is not present it is often simple to implement it.
The following page gives information on the types of extensions which are possible as well as where to start for each one.

.. toctree::
   :maxdepth: 1

   extensions


Development
-----------

If you are interested in contributing to SlipPy start here:

.. toctree::
   :maxdepth: 1

   contributing


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   authors
   history

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
