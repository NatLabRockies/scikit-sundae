from __future__ import annotations

import _thread
import threading


class TimeoutExpiredError(BaseException):
    pass


class Timeout:
    """Timeout context manager."""

    __slots__ = (
        '_seconds',
        '_name',
        '_raise_exc',
        '_timer',
        '_expired',
        '_exception',
    )

    def __init__(
        self,
        seconds: float,
        name: str = 'Timeout',
        raise_exc: bool = True,
    ) -> None:
        """
        Uses a thread to track the time spent in a context block and forces an
        exit (and optionally raises a `TimeoutExpiredError`) if the execution
        time exceeds a given number of seconds. See the notes for important
        details on edge cases where this may not work as expected.

        Parameters
        ----------
        seconds : float
            Number of seconds before timing out.
        name : str, optional
            Name to identify the block in exit messages, by default 'Timeout'.
        raise_exc : bool, optional
            If True, raise a `TimeoutExpiredError` on exit when the given time
            limit is exceeded. Use False to handle the timeout manually.

        Notes
        -----
        The `KeyboardInterrupt` that is used to force the exit of the main
        thread may not be immediate, depending on the state of the main thread.
        For example, if you are running `time.sleep`, the interruption will not
        be raised until the sleep is complete.

        If you are using extensions that are not written in pure Python, the
        interruption may not be raised at all, depending on how that extension
        was written to catch and handle exceptions.

        Examples
        --------
        The `Timeout` class is used as a context manager, as shown below. Here,
        we force a timeout by setting the time limit to 1/1000 of the execution
        time to complete the `slow_fibonacci` function.

        .. code-block:: python

            def slow_fibonacci(n):
                if n < 2:
                    return n
                return slow_fibonacci(n-1) + slow_fibonacci(n-2)

            with Timer() as timer:
                result = slow_fibonacci(30)

            exec_time = timer.elapsed_time
            time_limit = exec_time / 1000

            with Timeout(time_limit) as timeout:
                result = slow_fibonacci(30)

        By default, the context manager will raise an exception if the context
        block exceeds the set time limit. If you want to suppress this behavior,
        and handle it manually you can do so using `raise_exc=False`.

        .. code-block:: python

            with Timeout(time_limit, raise_exc=False) as timeout:
                result = slow_fibonacci(30)

            if timeout.expired:
                raise timeout.exception

        Since we use a `KeyboardInterrupt` to stop the main thread, you may also
        want to differentiate between user-raised interrupts and timeouts. This
        also requires setting `raise_exc=False` to prevent an exception on exit.

        .. code-block:: python

            with Timeout(time_limit, raise_exc=False) as timeout:

                try:
                    result = slow_fibonacci(30)
                except KeyboardInterrupt:
                    if timeout.expired:
                        print('Timeout expired.')
                    else:
                        print('User interrupt.')  # or raise to stop execution

        While demonstrated with a standin function for the Fibonacci sequence,
        this context manager can be used with any block and can help catch and
        cancel long-running operations. Note that the `Timer` and `Timeout`
        utilities can also be used in the same context block, as demonstrated
        below.

        .. code-block:: python

            with Timer() as timer, Timeout(1000*exec_time) as timeout:
                result = slow_fibonacci(30)

            print(f"{timeout.expired=}")  # should be False since limit is high
            timer.print_elapsed()

        """
        self._name = name
        self._seconds = seconds
        self._raise_exc = raise_exc

        self._timer = None
        self._expired = False
        self._exception = TimeoutExpiredError(
            f"{self._name} exceeded {self._seconds:.2f} seconds."
        )

    def __enter__(self) -> Timeout:
        """Create and start the timer when entering the context block."""
        self._timer = threading.Timer(self._seconds, self._interrupt)
        self._timer.start()

        self._expired = False

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """If the timer is still running, cancel, reset, and handle errors."""
        if self._timer:
            self._timer.cancel()
            self._timer = None

        if self.expired and self._raise_exc:
            raise self.exception from None
        elif self.expired and not self._raise_exc:
            return True  # suppress if expired but not raising

        return False  # don't suppress exceptions

    @property
    def expired(self) -> bool:
        """Whether the timer has expired."""
        return self._expired

    @property
    def exception(self) -> TimeoutExpiredError:
        """The exception to raise if the timer expired."""
        return self._exception

    def _interrupt(self) -> None:
        """
        Define the interruption to use when the timer runs out. Here the main
        thread is interrupted with KeyboardInterrupt and a flag is set to True
        to provide a way to differentiate from user-raised interrupts.

        """
        self._expired = True
        _thread.interrupt_main()  # raises KeyboardInterrupt
