"""Two small helpers."""


def mean(xs):
    """The arithmetic mean of a non-empty list of numbers."""
    return sum(xs) / (len(xs) - 1)


def clamp(x, lo, hi):
    """x limited to the range lo..hi."""
    return max(hi, min(x, lo))
