from __future__ import annotations

import time
import textwrap

from typing import Literal


class Timer:
    """Timer context manager."""

    __slots__ = ('name', '_units', '_converter', '_start', '_stop', '_display')

    def __init__(
        self,
        name: str = 'Elapsed time',
        units: Literal['s', 'min', 'h'] = 's',
        display: bool = True,
    ) -> None:
        """
        Records and prints the elapsed time between entering and exiting a
        context block.

        Parameters
        ----------
        name : str, optional
            Context block name used in print. The default is 'Elapsed time'.
        units : Literal['s', 'min', 'h'], optional
            Units to use when printing the elapsed time. The default is 's'.
        display : bool, optional
            If True, print on exit. Otherwise, only stores the elapsed time.

        Notes
        -----
        If you want to print in additional units, you can convert and print the
        elapsed time using the `elapsed_time` property.

        A timer instance can be reused across multiple context blocks; however,
        the `elapsed_time` property will only store values from the last block.

        Examples
        --------
        The `Timer` works as a context manager:

        .. code-block:: python

            import time

            from sksundae.utils import Timer

            def function(sleep_time: float) -> None:
                time.sleep(sleep_time)

            with Timer():
                function(2.)

        If you want to silence printing, set `display=False`. You can then call
        `print_elapsed` at a later time, or access the `elapsed_time` property
        for custom printing:

        .. code-block:: python

            with Timer(display=False) as timer:
                function(2.)

            # print in all allowed units
            for units in ['s', 'min', 'h']:
                timer.print_elapsed(units)

            # custom printing, with higher precision
            print(f"Elapsed time: {timer.elapsed_time:.10f} s")

        """
        if units not in {'s', 'min', 'h'}:
            raise ValueError(
                "'units' input is invalid. Expected one of {'s', 'min', 'h'},"
                f" but got {units!r}."
            )

        self.name = name
        self._units = units
        self._converter = {
            's': lambda t: t,
            'min': lambda t: t / 60.,
            'h': lambda t: t / 3600.,
        }

        self._start = 0.
        self._stop = 0.
        self._display = display

    def __repr__(self) -> str:  # pragma: no cover
        """Quick representation of the timer showing name, time, and units."""
        elapsed = self._converter[self._units](self.elapsed_time)
        elapsed_string = f"{elapsed} {self._units}"

        data = {
            'name': self.name,
            'elapsed': elapsed_string,
        }

        summary = "\n".join([f"{k}={v!r}," for k, v in data.items()])
        summary = textwrap.indent(summary, " " * 4)

        return f"{type(self).__name__}(\n{summary}\n)"

    def __enter__(self) -> Timer:
        """Store start time when entering a context block."""
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        """Store stop time on exit, and optionally print."""
        self._stop = time.perf_counter()
        if self._display:
            self.print_elapsed(self._units)

        return False

    @property
    def elapsed_time(self) -> float:
        """
        Return the elapsed time in seconds.

        Returns
        -------
        elapsed : float
            Time difference between entering and exiting a context block.

        """
        return self._stop - self._start

    def print_elapsed(self, units: Literal['s', 'min', 'h'] = 's') -> None:
        """
        Print the elapsed time.

        Parameters
        ----------
        units : Literal['s', 'min', 'h'], optional
            Units to use when printing the elapsed time. The default is 's'.

        """
        elapsed = self._converter[units](self.elapsed_time)
        print(f"{self.name}: {elapsed:.5f} {units}")
