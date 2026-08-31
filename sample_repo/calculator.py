"""A small calculator module used as a sample codebase for Codebase Copilot to index and debug."""


def add(a, b):
    """Return the sum of a and b."""
    return a + b


def subtract(a, b):
    """Return the difference of a and b."""
    return a - b


def multiply(a, b):
    """Return the product of a and b."""
    return a * b


def divide(a, b):
    """Return the quotient of a divided by b.

    Note: this currently does not guard against division by zero.
    """
    return a / b


def average(numbers):
    """Return the average of a list of numbers."""
    if not numbers:
        return 0
    total = sum(numbers)
    return total / len(numbers)

def compute_statistics(numbers):
    """Compute a small set of summary statistics for a list of numbers.

    Returns a dictionary containing the count, sum, mean, minimum, maximum,
    and a naive population variance and standard deviation. This function is
    intentionally verbose so it exceeds 600 characters as a single unit --
    used to demonstrate why naive fixed-size text chunking is worse than
    AST-based chunking for code search and retrieval.
    """
    if not numbers:
        raise ValueError("Cannot compute statistics for an empty list")

    count = len(numbers)
    total = sum(numbers)
    mean = total / count
    minimum = min(numbers)
    maximum = max(numbers)

    squared_diffs = [(x - mean) ** 2 for x in numbers]
    variance = sum(squared_diffs) / count
    std_dev = variance ** 0.5

    return {
        "count": count,
        "sum": total,
        "mean": mean,
        "min": minimum,
        "max": maximum,
        "variance": variance,
        "std_dev": std_dev,
    }