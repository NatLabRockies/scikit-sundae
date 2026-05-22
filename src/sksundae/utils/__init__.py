"""General-purpose module for shared utilities across the package."""

from ._timer import Timer
from ._timeout import Timeout
from ._rich_result import RichResult

__all__ = ['Timer', 'Timeout', 'RichResult']
