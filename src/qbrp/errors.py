"""Exception types raised by qbrp."""


class BacktestInputError(ValueError):
    """Raised when inputs to a backtest, strategy or metric are invalid.

    Subclasses ``ValueError`` so callers that already catch ``ValueError``
    keep working. The message always says which argument was wrong and why.
    """
