"""
generate_bug_dataset_v2.py
----------------------------
v2: broader domain diversity + new mutation types, built specifically to
address the out-of-distribution generalization failure found in v1 (which
predicted "buggy" for 8/8 hand-written examples from unfamiliar domains).

New mutation types, each targeting a bug CATEGORY the v1 OOD test exposed:
  - silent_wrong_default: converts a loud "raise on bad input" into a
    silent wrong-answer-on-bad-input -- the get_extension() failure pattern
  - wrong_constant_scale: removes a "/100" or "*100"-style unit conversion
    -- the calculate_discount() failure pattern
  - off_by_one: nudges a numeric constant by 1 -- a very common, general
    real-world bug class not covered at all in v1

New domains added to the clean-function bank: parsing/validation, unit
conversion, formatting -- not just math/list utilities.

Run:
    python generate_bug_dataset_v2.py bug_dataset_v2.jsonl
"""

import ast
import copy
import json
import random
import sys

random.seed(42)

# ---- Original domains (kept from v1) ----
MATH_LIST_FUNCTIONS = [
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
    "def is_within_range(value, lo, hi):\n    return lo <= value <= hi",
    "def median(numbers):\n    if not numbers:\n        raise ValueError('empty list')\n    s = sorted(numbers)\n    mid = len(s) // 2\n    if len(s) % 2 == 0:\n        return (s[mid - 1] + s[mid]) / 2\n    return s[mid]",
    "def most_frequent(items):\n    if not items:\n        raise ValueError('empty list')\n    return max(set(items), key=items.count)",
    "def zip_lists(a, b):\n    if len(a) != len(b):\n        raise ValueError('lists must be same length')\n    return list(zip(a, b))",
    "def get_middle_element(items):\n    if not items:\n        raise ValueError('empty list')\n    return items[len(items) // 2]",
    "def last_n(items, n):\n    if n <= 0:\n        return []\n    return items[-n:]",
    "def first_n(items, n):\n    if n <= 0:\n        return []\n    return items[:n]",
]

# ---- NEW: parsing / validation domain ----
PARSING_VALIDATION_FUNCTIONS = [
    "def parse_coordinates(s):\n    parts = s.split(',')\n    if len(parts) != 2:\n        raise ValueError('expected lat,lon')\n    return float(parts[0]), float(parts[1])",
    "def parse_time(s):\n    parts = s.split(':')\n    if len(parts) != 2:\n        raise ValueError('expected HH:MM')\n    return int(parts[0]), int(parts[1])",
    "def get_file_extension(filename):\n    if '.' not in filename:\n        return ''\n    return filename.rsplit('.', 1)[-1]",
    "def get_domain(url):\n    if '://' not in url:\n        raise ValueError('missing scheme')\n    return url.split('://')[1].split('/')[0]",
    "def parse_query_param(url, key):\n    if '?' not in url:\n        return None\n    query = url.split('?', 1)[1]\n    for pair in query.split('&'):\n        if '=' not in pair:\n            continue\n        k, v = pair.split('=', 1)\n        if k == key:\n            return v\n    return None",
    "def validate_phone(s):\n    digits = ''.join(c for c in s if c.isdigit())\n    return len(digits) == 10",
    "def validate_email_format(s):\n    if s.count('@') != 1:\n        return False\n    local, domain = s.split('@')\n    return bool(local) and '.' in domain",
    "def parse_csv_row(line):\n    return [field.strip() for field in line.split(',')]",
    "def parse_key_value(s):\n    if '=' not in s:\n        raise ValueError('expected key=value')\n    key, value = s.split('=', 1)\n    return key.strip(), value.strip()",
    "def strip_prefix(s, prefix):\n    if not s.startswith(prefix):\n        return s\n    return s[len(prefix):]",
    "def is_valid_hex_color(s):\n    if not s.startswith('#') or len(s) != 7:\n        return False\n    return all(c in '0123456789abcdefABCDEF' for c in s[1:])",
]

# ---- NEW: unit conversion / scaling domain (targets the "wrong constant" bug class) ----
CONVERSION_FUNCTIONS = [
    "def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32",
    "def fahrenheit_to_celsius(f):\n    return (f - 32) * 5 / 9",
    "def percentage(part, whole):\n    if whole == 0:\n        raise ValueError('whole cannot be zero')\n    return (part / whole) * 100",
    "def apply_discount(price, percent_off):\n    return price - (price * percent_off / 100)",
    "def apply_tax(price, tax_percent):\n    return price + (price * tax_percent / 100)",
    "def bytes_to_megabytes(b):\n    return b / (1024 * 1024)",
    "def minutes_to_seconds(m):\n    return m * 60",
    "def km_to_miles(km):\n    return km * 0.621371",
    "def compound_interest(principal, rate_percent, years):\n    return principal * ((1 + rate_percent / 100) ** years)",
    "def grade_to_gpa(percent):\n    return (percent / 100) * 4.0",
]

# ---- NEW: formatting / output domain ----
FORMATTING_FUNCTIONS = [
    "def truncate(s, length):\n    if len(s) <= length:\n        return s\n    return s[:length] + '...'",
    "def pad_left(s, width, char='0'):\n    if len(s) >= width:\n        return s\n    return char * (width - len(s)) + s",
    "def format_currency(amount):\n    return f'${amount:.2f}'",
    "def pluralize(word, count):\n    if count == 1:\n        return word\n    return word + 's'",
    "def format_duration(seconds):\n    minutes = seconds // 60\n    remaining = seconds % 60\n    return f'{minutes}m {remaining}s'",
]

CLEAN_FUNCTIONS = (
    MATH_LIST_FUNCTIONS + PARSING_VALIDATION_FUNCTIONS
    + CONVERSION_FUNCTIONS + FORMATTING_FUNCTIONS
)


class BugMutator(ast.NodeTransformer):
    def __init__(self, mutation_type):
        self.mutation_type = mutation_type
        self.applied = False

    def visit_Compare(self, node):
        if self.mutation_type == "flip_comparison" and not self.applied:
            flips = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE,
                     ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
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
        if self.mutation_type == "remove_guard" and not self.applied:
            self.applied = True
            return None

        # NEW: converts a validation-style "raise on bad input" into a
        # silent wrong answer instead -- the get_extension() failure
        # pattern. Only applies to If blocks that actually raise.
        if self.mutation_type == "silent_wrong_default" and not self.applied:
            has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(node))
            if has_raise:
                self.applied = True
                # Replace the whole validation block with a silent
                # `return None` -- loud failure becomes quiet wrong answer
                return ast.Return(value=ast.Constant(value=None))
        return self.generic_visit(node)

    def visit_BinOp(self, node):
        if self.mutation_type == "swap_arithmetic" and not self.applied:
            swaps = {ast.Add: ast.Sub, ast.Sub: ast.Add,
                     ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult}
            op_type = type(node.op)
            if op_type in swaps:
                node.op = swaps[op_type]()
                self.applied = True
            return self.generic_visit(node)

        # NEW: removes a unit-conversion scaling factor, e.g. "x * 9 / 5"
        # or "x / 100" -- the calculate_discount() failure pattern (forgot
        # to convert percent to a fraction).
        if self.mutation_type == "wrong_constant_scale" and not self.applied:
            if isinstance(node.op, (ast.Div, ast.Mult)) and isinstance(node.right, ast.Constant):
                self.applied = True
                return self.generic_visit(node.left)
            if isinstance(node.op, (ast.Div, ast.Mult)) and isinstance(node.left, ast.Constant):
                self.applied = True
                return self.generic_visit(node.right)
        return self.generic_visit(node)

    def visit_BoolOp(self, node):
        if self.mutation_type == "swap_boolop" and not self.applied:
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
            self.applied = True
        return self.generic_visit(node)

    def visit_Constant(self, node):
        # NEW: off-by-one -- nudges a numeric literal by 1. Covers a very
        # common, general real bug class not represented at all in v1.
        if self.mutation_type == "off_by_one" and not self.applied:
            if isinstance(node.value, int) and not isinstance(node.value, bool):
                self.applied = True
                return ast.Constant(value=node.value + 1)
        return node

    def visit_FunctionDef(self, node):
        # NEW: deletes an entire assignment or side-effecting statement from
        # the function body -- targets a bug class the v2 OOD test exposed
        # as a gap: a function that computes a value but never actually
        # stores/uses it (a missing statement, not an altered one).
        if self.mutation_type == "delete_statement" and not self.applied:
            candidates = [
                i for i, stmt in enumerate(node.body)
                if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.Expr))
            ]
            if candidates:
                idx = random.choice(candidates)
                new_body = node.body[:idx] + node.body[idx + 1:]
                if new_body:  # never leave a function with an empty body
                    node.body = new_body
                    self.applied = True
                    return node
        return self.generic_visit(node)


MUTATION_TYPES = [
    "flip_comparison", "remove_guard", "swap_arithmetic", "swap_boolop",
    "silent_wrong_default", "wrong_constant_scale", "off_by_one",
    # "delete_statement" removed -- tested and confirmed to regress both
    # held-out eval (0.664 -> 0.572 balanced accuracy) AND fresh OOD
    # generalization (5/8 -> 4/8), while failing to even fix the specific
    # case it targeted. Kept the BugMutator method above (harmless, unused)
    # as a documented record of a tried-and-rejected idea, rather than
    # deleting it outright.
]


def mutate(source: str) -> str | None:
    random.shuffle(MUTATION_TYPES)
    for mutation_type in MUTATION_TYPES:
        tree = ast.parse(source)
        mutator = BugMutator(mutation_type)
        new_tree = mutator.visit(copy.deepcopy(tree))
        if mutator.applied:
            ast.fix_missing_locations(new_tree)
            try:
                return ast.unparse(new_tree), mutation_type
            except Exception:
                continue
    return None, None


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else "bug_dataset_v2.jsonl"
    examples = []

    n_changed = 0
    for source_id, raw_source in enumerate(CLEAN_FUNCTIONS):
        # Pass clean code through the same formatter the mutants go through,
        # so formatting can no longer distinguish clean from buggy.
        source = ast.unparse(ast.parse(raw_source))
        if source != raw_source:
            n_changed += 1
        examples.append({"code": source, "label": 0, "source_id": source_id,
                         "mutation": "none"})
        seen_mutants = set()
        attempts = 0
        while len(seen_mutants) < 6 and attempts < 25:
            attempts += 1
            mutant, mutation_type = mutate(source)
            if mutant and mutant not in seen_mutants and mutant != source:
                seen_mutants.add(mutant)
                examples.append({"code": mutant, "label": 1, "source_id": source_id,
                                 "mutation": mutation_type})

    random.shuffle(examples)

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    n_clean = sum(1 for e in examples if e["label"] == 0)
    n_buggy = sum(1 for e in examples if e["label"] == 1)
    print(f"Generated {len(examples)} examples ({n_clean} clean, {n_buggy} buggy)")
    print(f"From {len(CLEAN_FUNCTIONS)} base functions across "
          f"{len(MUTATION_TYPES)} mutation types")
    print(f"Written to {output_path}")
    print(f"{n_changed}/{len(CLEAN_FUNCTIONS)} clean functions changed under normalization")


if __name__ == "__main__":
    main()