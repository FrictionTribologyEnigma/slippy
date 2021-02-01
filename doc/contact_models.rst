.. _ContactModels:


Contact Models
==============

Slippy can solve rough surface contact problems with the boundary element method, based on the half space approximation.
In order to access this functionality you should make and solve a contact model which describes the problem you want to model.
When solved this model will run through each of the steps it contains, first solving the main problem described by the step then solving all the sub-models and outputs required.
This process is shown in the image below.

|solution|

Making a contact model has several steps but a common work flow can be used for a wide variety of simulations.

Define surface geometry
-----------------------

A contact model requires two surfaces to be defined. The first surface should also be descretised on a grid which will
form the mesh for the solution. Some sub models will also require the second surface to be descretised on a grid so
that processes such as wear can be applied.

Define materials
----------------

The material properties of each of the surfaces must be defined by making material objects and assigning them to the surfaces.
These objects both describe how the materials deform (eg elastic) and also provide the computational back end for solving the contact.

At this stage if the contact is lubricated the properties of the lubricant should also be defined.

Make contact model
------------------

Now you are ready to make a contact model object, to do this you simply pass the surfaces and the lubricant to the constructor.
This will make a new contact model object, but at this stage it is an empty shell, that will not do anything when solved.

Make a model step
-----------------

A model step describes a contact problem to solve, this might be the normal load between the surfaces and / or the tangential displacement.
More formally the step should include processes which must be two way coupled to the contact mechanics solution.
So, for example, for an EHL problem an appropriate step must be selected which will solve the fluid pressure problem and the deformation problem at the same time.

Add sub models
--------------

Sub - models can solve any additional behaviour which only needs to be one way coupled to the behaviour in the step.
This typically includes things which change more slowly over time, such as wear processes.
They can also be used to calculate output parameters, thus reducing the size of output files.

Add step to model
-----------------

When the step has been made and the sub model have been added, the step can be added to the model.
By default steps are solved in the order they are added, however this can be over ridden if needed.

Add output requests
-------------------

A model in slippy may have several steps, each of these may may have many time steps. Writing the entire state of the
model to file for each time step would be time consuming and result in very large output files.
To solve this problem slippy has an output request system which can be used to save output parameters at set time points.
If simulations are longer than one time step this should be used to save intermediate states.

Checking the model
------------------

As an optional measure you can check the model before running.
This will ensure that the surface are properly defined, with materials.
It will also check that parameters which are required for sub-models to run will be present in the model state when the sub-model is executed.
This can save errors in execution. Lastly, it will also check that all requested outputs will be available when they are to be written.
If an output cannot be found it will silently error, meaning that the model will not stop execution, but the output will not be present in the output file.

Solve model
-----------

The final step is to solve the model this will run thorough the steps in order, solving the main contact mechanics problem,
followed by the sub models and finally writing the requested outputs.
The output database can be read in by using the OutputReader object in slippy.contact.

.. |solution| image:: solving.svg
        :alt: Solving schematic
        :target: https://github.com/FrictionTribologyEnigma/slippy
