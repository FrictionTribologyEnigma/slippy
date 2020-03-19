.. _Insatllation:

Installation
============

This guide details one way of installing slippy. It is assumed that the reader has no experiance in python but is familar with some programming.

Installing python
-----------------

It is recomended to install python by installing anaconda, this is a popular scientific python distribution. The latest version of anaconda can be found at: https://www.anaconda.com/distribution/. 

SlipPY requires that the >=3.6 version should be installed. 

Making a virtual environment
----------------------------

When using python it is common to install new packages (such as slippy) to make certain tasks easier. Unfortunately sometimes these packages have conflicting requirements. Installing a package with conflicting requirements can cause other packages to stop working. 

Because of this it is always recommended to work in a 'virtual environment' and to use a different virtual environment for each project. 

When anaconda is installed there will be a program called anaconda prompt avaliable. This is not a python interpreter, it is used only for manging your installation and launching programs. 

To make a new virtual environment called 'slippy', type the following into this prompt::

	conda create --name slippy

When this environment is activated any packages installed will be installed only for this environment. The environment can be activated by typing the following::

	conda activate slippy

You should notice that the text to the left of the cursor now says slippy instead of base. 

Installing slippy
-----------------

Now that the environment is activated slippy can be installed by typing the following::

	conda install slippy

This will install slippy and all of it's dependencies for the current environment. 

Using slippy (recomendations)
-----------------------------

It is recommended that the user install an IDE (integrated development environment) to use with python. These are text editors that allow the user to run code and view current variables in a user friendly manner. The three listed below are easy to use and install:

spyder: Probably the most popular open source IDE for python, can be installed by typing: 'conda install spyder' and run by typing: 'spyder' into the anaconda prompt.

pycharm: An advance IDE with many extremely useful features

jupyter: An alternative development set up. Sets up a local server and runs in a browser, allows the user to mix code, text and inline plots. Installed by typing: 'conda install jupyter', run by typing: 'jupyter notebook'