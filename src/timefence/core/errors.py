"""Exceptions shared by every layer."""


class TimeFenceError(Exception):
    """Base class for all project errors."""


class LeakError(TimeFenceError):
    """Raised when a query would return data that was not yet available at as_of."""


class ExpressionError(TimeFenceError):
    """Raised when a factor expression fails parsing or validation."""


class BudgetExceeded(TimeFenceError):
    """Raised when an LLM call would exceed the configured budget."""
