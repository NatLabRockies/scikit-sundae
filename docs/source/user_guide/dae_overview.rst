Differential Algebraic Equations
================================

What are DAEs?
^^^^^^^^^^^^^^
Differential algebraic equations (DAEs) are a classification of problem defined by a system of equations that includes both differential and algebraic components. Unlike ordinary differential equations (ODEs), which define relationsihps between one or more dependent variables based on their derivatives, DAEs involve a mixture of differential terms and algebraic constraints. The algebraic constraints must be satisfied alongside the differential equations as the system evolves over time.

The general form of a DAE is written as:

.. math:: 

    F(t, y, y') = 0,

where :math:`y(t)` includes both differential variables (variables with derivatives) and algebraic variables (variables without derivatives).

In practice, DAEs are typically expressed as a system of equations that must equal zero. This form reflects that the solution seeks to minimize the difference, or residual, between the expressions in :math:`F` and zero. The residuals play a key role in evaluating the performance of the solver at each time step, as they represent the "error" in how well the system satisfies the algebraic and differential equations simultaneously.

For example, in a mechanical system involving a constraint like a rigid rod, the distance between two points should ideally remain fixed. However, due to the limits of numerical precision, the DAE solver works to minimize any small violations in this constraint.

Where do DAEs Arise?
^^^^^^^^^^^^^^^^^^^^
DAEs commonly arise in real-world systems that include physical constraints. Examples include:

* Electrical circuits (e.g., Kirchhoff's laws for currents and voltages)
* Mechanical systems with rigid body constraints (e.g., constrained motion of linkages)
* Chemical processes with equilibrium reactions

These systems are often governed by a combination of laws of motion or flow (differential equations) and constraints that restrict certain variables (algebraic equations).

Classification
--------------
The index of a DAE refers to the number of times an algebraic equation must be differentiated with respect to time to reduce the system to a set of ODEs. This concept is critical for understanding the complexity of solving DAEs.

* **Index-1 DAEs:** These are the simplest type of DAEs, where the algebraic constraints are directly compatible with the differential equations. In other words, the system can be solved without needing to differentiate the algebraic constraints.
* **Higher-index DAEs:** These require one or more differentiations of the algebraic constraints to be reduced to an ODE-like system. Higher-index DAEs are typically more challenging to solve numerically and may require specific reformulations.

The `IDA` solver in scikit-SUNDAE is designed to handle index-1 DAEs. Users working with higher-index DAEs often need to reformulate their systems to reduce them to index-1. One common method is to differentiate the algebraic equations until they become compatible with the differential system.

Intro to IDA
------------
The IDA solver, part of the SUNDIALS suite, is specifically designed to handle index-1 DAEs. It uses implicit methods to solve both the differential and algebraic parts of the system simultaneously. By combining backward differentiation formulas (BDF) for the differential components and Newton-based methods for solving the resulting nonlinear systems, IDA efficiently handles DAEs in which algebraic constraints must be satisfied at each time step.

Setting up DAEs for IDA
^^^^^^^^^^^^^^^^^^^^^^^
When using the `IDA` solver in scikit-SUNDAE, users must define which variables are differential and which are algebraic. The solver requires the user to provide:

* The system of governing equations, written as residuals, like :math:`F(t, y, y') = 0`
* Initial conditions for all variables (differential and algebraic) and their derivatives

In regards to the initial conditions, `IDA` offers an initial correction scheme to help refine their values before solving the system. For instance, it is common to know either :math:`y_0` or :math:`y'_0` but not necessarily both. In such cases, users can rely on the solver to determine whichever is missing, ensuring the system is correctly initialized for the numerical solution. For more information see the `calc_initcond` setting in the API reference.

Numerical Methods 
^^^^^^^^^^^^^^^^^
IDA applies several key numerical techniques for solving DAEs:

* **Implicit Time-Stepping:** IDA uses implicit methods like BDF, which allow for large time steps and ensure stability, particularly for stiff DAEs.
* **Newton Iteration:** To solve the nonlinear system at each time step, IDA uses Newton's method with the option to provide explicit Jacobians or rely on finite-difference approximations.
* **Constraint Handling:** IDA enforces algebraic constraints through its stepper, ensuring that both differential and algebraic variables evolve together in a physically consistent manner.

These methods allow IDA to handle DAEs that cannot be solved with traditional ODE solvers, making it an ideal tool for systems with physical constraints.

Summary
-------
DAEs are an extension of ODEs that include algebraic equations in addition to differential equations. While they are more complex to solve, the `IDA` solver in scikit-SUNDAE provides a powerful tool for solving index-1 DAEs using implicit methods and Newton-based iterations. Properly defining the differential and algebraic variables and understanding the system's index is essential for setting up and solving a DAE problem efficiently.
