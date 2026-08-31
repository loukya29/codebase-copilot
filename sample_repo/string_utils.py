"""String and list utilities. Several functions here are deliberately similar
to each other -- used to stress-test whether an embedding model can tell
apart genuinely different-but-related functions, not just unrelated ones."""


def reverse_string(text):
    """Return the input string with its characters in reverse order."""
    return text[::-1]


def reverse_list(items):
    """Return a new list with the elements in reverse order."""
    return items[::-1]


def is_palindrome(text):
    """Return True if the string reads the same forwards and backwards,
    ignoring case and spaces."""
    cleaned = text.lower().replace(" ", "")
    return cleaned == cleaned[::-1]


def deduplicate(items):
    """Return a new list with duplicate elements removed, preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def find_duplicates(items):
    """Return a list of elements that appear more than once in the input."""
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)


def chunk_list(items, size):
    """Split a list into sublists of at most `size` elements each."""
    return [items[i:i + size] for i in range(0, len(items), size)]


def flatten(nested_list):
    """Flatten a list of lists into a single list, one level deep."""
    return [item for sublist in nested_list for item in sublist]


def safe_divide(a, b, default=0):
    """Divide a by b, returning `default` instead of raising an error
    if b is zero. Unlike calculator.divide, this never raises."""
    if b == 0:
        return default
    return a / b


def clamp(value, minimum, maximum):
    """Restrict a value to lie within [minimum, maximum]."""
    return max(minimum, min(value, maximum))


def normalize_whitespace(text):
    """Collapse multiple consecutive whitespace characters into a single space
    and strip leading/trailing whitespace."""
    return " ".join(text.split())