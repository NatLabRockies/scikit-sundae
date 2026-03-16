Explicit Jacobians
==================
In the context of solving differential equations, the Jacobian matrix represents the partial derivatives of a system's functions with respect to its variables. It plays a critical role in solvers like CVODE and IDA, where it is used to linearize the system of equations around the current solution, aiding in both convergence and stability, especially for stiff problems.

Explicit Jacobians, where the user provides the exact derivative values, allow the solver to perform more efficiently by avoiding costly numerical approximations. Both CVODE (for ODEs) and IDA (for DAEs) can leverage explicit Jacobians, though the mathematical form of the Jacobian differs between the two solvers.

.. _Mathematical Definitions:

Mathematical Definitions
------------------------

CVODE - Jacobians for ODEs
^^^^^^^^^^^^^^^^^^^^^^^^^^
For a system of ordinary differential equations (ODEs): 

.. math:: 

    \frac{dy}{dt} = f(t, y), 

the Jacobian is defined as 

.. math:: 

    J_{ij} = \frac{\partial f_i}{\partial y_j}.

This matrix represents the rate of change of the system's functions :math:`f` with respect to the variables :math:`y`. In CVODE, this Jacobian is used to update the solution during each time step by approximating how the system evolves near the current state.

IDA - Jacobians for DAEs
^^^^^^^^^^^^^^^^^^^^^^^^
For a system of differential algebraic equations (DAEs), the situation is more complex. The system takes the form:

.. math:: 

    F(t, y, \dot{y}) = 0,

where :math:`y` is the set of variables and :math:`\dot{y}` are the time derivatives of :math:`y`. The Jacobian matrix in this context is defined as:

.. math:: 

    J_{ij} = \frac{\partial F_i}{\partial y_j} + \alpha \frac{\partial F_i}{\partial \dot{y}_j}.

Here, :math:`\alpha` is a scalar that the solver computes during the integration process. This expression reflects both the differential and algebraic components of the DAE, making the Jacobian more involved than in the ODE case.

Numerical Approximations
------------------------
When an explicit Jacobian is not provided, both CVODE and IDA use finite difference approximations to estimate the Jacobian matrix. This involves computing small changes in :math:`y` and/or :math:`\dot{y}` to approximate derivatives. This approach can be slow, especially for large systems, because the solver must repeatedly perturb variables and evaluate system expressions.

Setting up Jacobians in scikit-SUNDAE
-------------------------------------
In scikit-SUNDAE, users can supply explicit Jacobian functions for both CVODE and IDA. This is done during the solver initialization phase, where you can pass a user-defined Jacobian function, as demonstrated in the dummy examples below.

.. code-block:: python

    from sksundae.ida import IDA
    from sksundae.cvode import CVODE 

    # Dummy residual function for IDA
    def resfn(t, y, yp, res, userdata):
        """
        res[0] = ...
        res[...] = ...
        """
        pass 

    # Dummy Jacobian function for IDA
    def jacfn(t, y, yp, res, cj, JJ, userdata):
        """
        JJ[0,0] = ...
        JJ[...,...] = ...
        """
        pass

    solver = IDA(resfn, jacfn=jacfn, userdata={'data': None})

    # Dummy right-hand-side function for CVODE
    def rhsfn(t, y, yp):
        """
        yp[0] = ...
        yp[...] = ...
        """
        pass 

    # Dummy Jacobian function for CVODE
    def jacfn(t, y, yp, JJ):
        """
        JJ[0,0] = ...
        JJ[...,...] = ...
        """
        pass

    solver = CVODE(rhsfn, jacfn=jacfn)

As demonstrated in the examples, the Jacobian functions do not have return values. Instead, they each have `JJ` inputs, which are pre-allocated 2D matrices that you can fill within the functions. This is similar to how the residual and right-hand-side functions do not require return values and instead fill the pre-allocated `res` and `yp` arrays, respectively. The optional `userdata` argument must be present in ALL or NO user-defined functions. For example, we provide it in both expressions for the `IDA` example, but neither in the `CVODE` example. Even if `userdata` is only used in one user-defined function, it must be present in all signatures so the solver can handle them correctly. 

.. note:: 
    
    Take care to not overwrite any pre-allocated arrays. You should NEVER set the variable directly equal to a value, e.g., `JJ = ...`. Instead, ALWAYS use indices. Even if you can fill all values in `JJ` using vector math, the correct way to do this is to write out `JJ[:,:] = ...`. This makes sure the pre-allocated matrix is filled rather than being overwritten. If you want to make sure the matrix starts out as zeros, use something like `JJ[:,:] = np.zeros_like(JJ)`, and then fill in the non-zero indices.

In the case of the `IDA` example there is a `cj` defined in the input signature. This `cj` corresponds to the :math:`\alpha` value in :ref:`Mathematical Definitions` section. You do not need to define `cj` yourself, but you should include it in your expressions as needed. The SUNDIALS backend calculates `cj` for you based on the internal step size and order being used. It exists in the function signature so that it can be accessed in your Jacobian function as it is internally updated.

Additional Considerations
-------------------------
Explicit Jacobians offer significant advantages for improving solver performance and accuracy, particularly for stiff or complex systems. However, they also introduce additional complexity and potential challenges in implementation. Below, we outline the key benefits and tradeoffs of providing an explicit Jacobian.

Benefits
^^^^^^^^
1. **Improved Performance:** Explicit Jacobians eliminate the need for the solver to approximate derivatives using finite differences, which is slower and less accurate, particularly for large or stiff systems.

2. **Better Accuracy for Stiff Problems:** Stiff systems, where small changes in variables can cause large changes in the solution, benefit from explicit Jacobians because they allow for more precise linearization of the system.

3. **Explicit Jacobians for Banded Problems:** If your problem has a banded Jacobian, it is still worth setting the linear solver to `band` rather than using the default `dense`. The Python bindings have to copy arrays back and forth into forms that Python and C can understand. If you leave your solver as `dense` then the computation time to copy unnecessary zeros can add up, especially for large problems. Therefore, banded problems can see a stacked benefit from setting both options, i.e., the linear solver and Jacobian function.

Considerations and Tradeoffs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. **Increased Complexity:** Defining the Jacobian manually for complex systems can be challenging. The Jacobian must be consistent with the function :math:`f` or :math:`F`, which can lead to errors if there are inconsistencies in indexing or derivatives.

2. **Error-Prone:** For large, intricate systems, ensuring the correctness of the Jacobian can be difficult and error-prone. Inconsistent definitions can lead to solver crashes or incorrect results.

3. **Development Overhead:** Writing out the explicit Jacobian requires careful planning and a deeper understanding of the system's mathematics. This can increase the development time and complexity of the code.
