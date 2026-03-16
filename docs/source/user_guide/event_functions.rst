Event Functions
===============
Event functions allow you to detect and record when specific conditions or "events" occur while integrating differential equations. Events typically define when a variable crosses a threshold and they enable the solver to respond by either stopping integration or saving the solution where they are detected, which is useful for post processing.

Events are crucial in applications where system behavior changes based on certain conditions, such as hitting a physical boundary or exceeding a maximum value. The solver evaluates these event conditions at each internal step during integration.

When to Use Events
------------------
Event functions are useful in a variety of situations, including:

* **State Transitions:** Detect when a variable crosses a boundary (e.g., an object hitting the ground).
* **Early Termination:** Stop the solver when a condition is met (e.g., a voltage threshold is reached).
* **Variable Resets:** Adjust system states or reset variables upon some condition being satisfied.

In general, event functions are ideal when your problem's solution depends on detecting and reacting to specific conditions during the integration process.

Setting Events in scikit-SUNDAE
-------------------------------
In scikit-SUNDAE, events are set up as user-defined functions that return zero when the event condition occurs. Each event function can have two key attributes:

* `terminal`: This attribute controls whether the solver stops when the event occurs. If set to `True` (default), the solver halts at the first detection of the event. Alternatively, you can provide an integer to stop the solver after some number of occurrences. If set to `False`, the event is recorded but the integration continues.

* `direction`: This attribute specifies the slope of the zero crossing that should trigger the event. A value of `+1` triggers the event only when the event expression crosses zero with a positive slope, `-1` for a negative slope, and `0` for either crossing direction.

Users can define as many events as they want within an events function. However, they should be aware of the performance considersations we list below.

In the code blocks below, we provide dummy event functions for both the `IDA` and `CVODE` solvers. These examples are simply to demonstrate the form of the event functions and how to pass them to the solvers. 

.. code-block:: python 

    from sksundae.ida import IDA 

    # Dummy residuals function
    def resfn(t, y, yp, res, userdata): 
        pass 

    def eventsfn(t, y, yp, events, userdata):
        events[0] = y[0] - userdata['limit']
        events[1] = yp[0]

    userdata = {'limit': 0.1}

    eventsfn.terminal = [True, False]
    eventsfn.direction = [0, 1]

    solver = IDA(resfn, eventsfn=eventsfn, num_events=2,             
                 userdata=userdata)

In the example above, `eventsfn` does not include a return value. Instead, the input signature must include an array by some name in the fourth position, here we call it `events`. This array is pre-allocated for you and only needs to be filled with expressions that, if equal to zero, will trigger an event. Even if you only have one event, you must fill it as `events[0]`, rather than setting `events` itself.

After `eventsfn` is defined, we set the `terminal` and `direction` attributes for each event by adding them to `eventsfn`. These must be set using lists with lengths matching the number of events. If you don't define these attributes, the defaults are `True` and `0`, respectively, for all events. Since the `events` array is pre-allocated for you, you must also specify the number of events `num_events` when you pass an events function to the solver. Failing to do so will raise an error.

Notice that we defined `userdata` in both the residuals function and `eventsfn` signatures. If you need to pass data to ANY user defined functions, then ALL user defined functions must include the extra argument. The data can be any Python type, but only takes up one input argument slot, not multiple. It is passed to the solver using the `userdata` keyword argument.

The settings we used in this example would perform the following:

* Trigger a terminal event, on the first occurrence, if `y[0]` is equal to the limit `0.1`, regardless of the zero-crossing direction.
* Record the event if `yp[0]` is ever equal to zero, but do not stop integrating, even if this occurs multiple times. `yp[0]` is the time derivative of `y[0]` so it represents local minima and maxima. By setting the direction for this event to `+1`, we only track when zero-crossings have positive slope (went from a negative to a positive value). Therefore, we only record local maxima, and ignore local minima.

CVODE also supports all of the same event function settings, however, its function signatures are slightly different. The corresponding example for CVODE is:

.. code-block:: python 

    from sksundae.cvode import CVODE

    # Dummy right-hand-side function
    def rhsfn(t, y, yp, userdata): 
        pass 

    def eventsfn(t, y, events, userdata):
        events[0] = y[0] - userdata['limit']
        events[1] = "..."  # some expression equivalent to yp[0], as specified
                           # in the right-hand-side function  

    userdata = {'limit': 0.1}

    eventsfn.terminal = [True, False]
    eventsfn.direction = [0, 1]

    solver = CVODE(rhsfn, eventsfn=eventsfn, num_events=2,             
                   userdata=userdata)

Performance Impact
------------------
Using event functions introduces additional computational overhead because the solver must evaluate the event conditions at each integration step. This can slow down the integration process, particularly when multiple events are being tracked. To minimize the performance impact, it is best to add only critical events.

**Note:** While event functions slow down the solver, they can decrease overall computational costs/time by terminating integration early, if set to do so.
