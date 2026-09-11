"""
generate_bug_dataset.py
-------------------------
Builds a labeled dataset for the "does this function look risky" classifier
using mutation-based bug seeding: start from clean, correct functions, then
programmatically apply common real-world bug patterns to create labeled
"buggy" variants. This is the same underlying idea used in mutation testing
tools (mutmut, cosmic-ray) -- just applied here to generate training data
rather than to test a test suite's thoroughness.

Label 0 = clean / looks fine
Label 1 = buggy / risky

Run:
    python generate_bug_dataset.py bug_dataset1.jsonl
"""

import ast
import copy
import json
import random
import sys

random.seed(42)

# A bank of clean, correct utility functions covering common patterns.
# Deliberately varied -- string ops, math, list ops, dict ops, control flow --
# so mutations produce a genuinely varied set of bug types, not one pattern
# repeated 150 times.
CLEAN_FUNCTIONS = [
    "def add(a, b):\n    return a + b",
    "def subtract(a, b):\n    return a - b",
    "def is_even(n):\n    return n % 2 == 0",
    "def is_positive(n):\n    return n > 0",
    "def clamp(value, lo, hi):\n    return max(lo, min(value, hi))",
    "def average(numbers):\n    if not numbers:\n        raise ValueError('empty list')\n    return sum(numbers) / len(numbers)",
    "def find_max(numbers):\n    if not numbers:\n        raise ValueError('empty list')\n    return max(numbers)",
    "def reverse_string(s):\n    return s[::-1]",
    "def is_palindrome(s):\n    cleaned = s.lower().replace(' ', '')\n    return cleaned == cleaned[::-1]",
    "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
    "def safe_divide(a, b, default=0):\n    if b == 0:\n        return default\n    return a / b",
    "def get_first(items, default=None):\n    if not items:\n        return default\n    return items[0]",
    "def deduplicate(items):\n    seen = set()\n    result = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result",
    "def chunk_list(items, size):\n    if size <= 0:\n        raise ValueError('size must be positive')\n    return [items[i:i+size] for i in range(0, len(items), size)]",
    "def flatten(nested):\n    return [item for sub in nested for item in sub]",
    "def get_value(d, key, default=None):\n    if key not in d:\n        return default\n    return d[key]",
    "def merge_dicts(a, b):\n    result = dict(a)\n    result.update(b)\n    return result",
    "def count_occurrences(items, target):\n    return sum(1 for item in items if item == target)",
    "def is_sorted(items):\n    for i in range(len(items) - 1):\n        if items[i] > items[i+1]:\n            return False\n    return True",
    "def binary_search(items, target):\n    lo, hi = 0, len(items) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if items[mid] == target:\n            return mid\n        elif items[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
    "def factorial(n):\n    if n < 0:\n        raise ValueError('negative input')\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)",
    "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
    "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
    "def normalize_whitespace(s):\n    return ' '.join(s.split())",
    "def title_case(s):\n    return ' '.join(word.capitalize() for word in s.split())",
    "def sum_of_squares(numbers):\n    return sum(n * n for n in numbers)",
    "def is_leap_year(year):\n    if year % 4 != 0:\n        return False\n    if year % 100 == 0 and year % 400 != 0:\n        return False\n    return True",
    "def contains_duplicates(items):\n    return len(items) != len(set(items))",
    "def swap_case(s):\n    return s.swapcase()",
    "def is_within_range(value, lo, hi):\n    return lo <= value <= hi",
    "def sum_digits(n):\n    return sum(int(d) for d in str(abs(n)))",
    "def is_anagram(a, b):\n    return sorted(a.lower()) == sorted(b.lower())",
    "def rotate_list(items, n):\n    if not items:\n        return items\n    n = n % len(items)\n    return items[n:] + items[:n]",
    "def median(numbers):\n    if not numbers:\n        raise ValueError('empty list')\n    s = sorted(numbers)\n    mid = len(s) // 2\n    if len(s) % 2 == 0:\n        return (s[mid - 1] + s[mid]) / 2\n    return s[mid]",
    "def count_words(s):\n    if not s.strip():\n        return 0\n    return len(s.split())",
    "def has_upper(s):\n    return any(c.isupper() for c in s)",
    "def has_digit(s):\n    return any(c.isdigit() for c in s)",
    "def strip_punctuation(s):\n    return ''.join(c for c in s if c.isalnum() or c.isspace())",
    "def first_non_repeating(s):\n    for c in s:\n        if s.count(c) == 1:\n            return c\n    return None",
    "def is_subset(a, b):\n    return set(a).issubset(set(b))",
    "def intersection(a, b):\n    return list(set(a) & set(b))",
    "def union_lists(a, b):\n    return list(set(a) | set(b))",
    "def difference(a, b):\n    return list(set(a) - set(b))",
    "def most_frequent(items):\n    if not items:\n        raise ValueError('empty list')\n    return max(set(items), key=items.count)",
    "def is_valid_age(age):\n    return 0 <= age <= 150",
    "def percentage(part, whole):\n    if whole == 0:\n        raise ValueError('whole cannot be zero')\n    return (part / whole) * 100",
    "def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32",
    "def fahrenheit_to_celsius(f):\n    return (f - 32) * 5 / 9",
    "def is_perfect_square(n):\n    if n < 0:\n        return False\n    root = int(n ** 0.5)\n    return root * root == n",
    "def truncate(s, length):\n    if len(s) <= length:\n        return s\n    return s[:length] + '...'",
    "def pad_left(s, width, char='0'):\n    if len(s) >= width:\n        return s\n    return char * (width - len(s)) + s",
    "def zip_lists(a, b):\n    if len(a) != len(b):\n        raise ValueError('lists must be same length')\n    return list(zip(a, b))",
    "def all_positive(numbers):\n    return all(n > 0 for n in numbers)",
    "def any_negative(numbers):\n    return any(n < 0 for n in numbers)",
    "def running_total(numbers):\n    total = 0\n    result = []\n    for n in numbers:\n        total += n\n        result.append(total)\n    return result",
    "def is_balanced_parens(s):\n    depth = 0\n    for c in s:\n        if c == '(':\n            depth += 1\n        elif c == ')':\n            depth -= 1\n        if depth < 0:\n            return False\n    return depth == 0",
    "def compress_string(s):\n    if not s:\n        return s\n    result = []\n    count = 1\n    for i in range(1, len(s)):\n        if s[i] == s[i-1]:\n            count += 1\n        else:\n            result.append(s[i-1] + str(count))\n            count = 1\n    result.append(s[-1] + str(count))\n    return ''.join(result)",
    "def is_power_of_two(n):\n    if n <= 0:\n        return False\n    return n & (n - 1) == 0",
    "def euclidean_distance(p1, p2):\n    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5",
    "def is_valid_email(s):\n    if '@' not in s:\n        return False\n    parts = s.split('@')\n    return len(parts) == 2 and bool(parts[0]) and bool(parts[1])",
    "def last_n(items, n):\n    if n <= 0:\n        return []\n    return items[-n:]",
    "def first_n(items, n):\n    if n <= 0:\n        return []\n    return items[:n]",
    "def has_negative(numbers):\n    return any(n < 0 for n in numbers)",
    "def sum_positive(numbers):\n    return sum(n for n in numbers if n > 0)",
    "def word_frequency(s):\n    words = s.lower().split()\n    freq = {}\n    for w in words:\n        freq[w] = freq.get(w, 0) + 1\n    return freq",
]


class BugMutator(ast.NodeTransformer):
    """Applies one random bug-seeding mutation to a function's AST.
    Each mutation type mirrors a real, common bug pattern."""

    def __init__(self, mutation_type):
        self.mutation_type = mutation_type
        self.applied = False

    def visit_Compare(self, node):
        # Flip a comparison operator -- classic off-by-one / boundary bug
        if self.mutation_type == "flip_comparison" and not self.applied:
            flips = {
                ast.Lt: ast.LtE, ast.LtE: ast.Lt,
                ast.Gt: ast.GtE, ast.GtE: ast.Gt,
                ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
            }
            new_ops = []
            for op in node.ops:
                op_type = type(op)
                if op_type in flips:
                    new_ops.append(flips[op_type]())
                    self.applied = True
                else:
                    new_ops.append(op)
            node.ops = new_ops
        return self.generic_visit(node)

    def visit_If(self, node):
        # Remove a guard clause entirely -- classic missing-edge-case bug
        if self.mutation_type == "remove_guard" and not self.applied:
            self.applied = True
            return None  # deletes this If node from the function body
        return self.generic_visit(node)

    def visit_BinOp(self, node):
        # Swap an arithmetic operator -- classic logic-error bug
        if self.mutation_type == "swap_arithmetic" and not self.applied:
            swaps = {ast.Add: ast.Sub, ast.Sub: ast.Add,
                     ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult}
            op_type = type(node.op)
            if op_type in swaps:
                node.op = swaps[op_type]()
                self.applied = True
        return self.generic_visit(node)

    def visit_BoolOp(self, node):
        # Swap and/or -- classic boolean-logic bug
        if self.mutation_type == "swap_boolop" and not self.applied:
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
            self.applied = True
        return self.generic_visit(node)


MUTATION_TYPES = ["flip_comparison", "remove_guard", "swap_arithmetic", "swap_boolop"]


def mutate(source: str) -> str | None:
    """Try each mutation type until one actually applies (some functions
    don't contain every pattern, e.g. no BoolOp to swap). Returns None if
    no mutation could be applied to this function."""
    random.shuffle(MUTATION_TYPES)
    for mutation_type in MUTATION_TYPES:
        tree = ast.parse(source)
        mutator = BugMutator(mutation_type)
        new_tree = mutator.visit(copy.deepcopy(tree))
        if mutator.applied:
            ast.fix_missing_locations(new_tree)
            try:
                return ast.unparse(new_tree)
            except Exception:
                continue
    return None


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else "bug_dataset.jsonl"
    examples = []

    for source in CLEAN_FUNCTIONS:
        # Clean example, label 0
        examples.append({"code": source, "label": 0})

        # Generate several distinct buggy mutants per clean function,
        # so the dataset isn't limited to len(CLEAN_FUNCTIONS) buggy examples
        seen_mutants = set()
        attempts = 0
        while len(seen_mutants) < 5 and attempts < 20:
            attempts += 1
            mutant = mutate(source)
            if mutant and mutant not in seen_mutants and mutant != source:
                seen_mutants.add(mutant)
                examples.append({"code": mutant, "label": 1})

    random.shuffle(examples)

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    n_clean = sum(1 for e in examples if e["label"] == 0)
    n_buggy = sum(1 for e in examples if e["label"] == 1)
    print(f"Generated {len(examples)} examples ({n_clean} clean, {n_buggy} buggy)")
    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()