"""
generate_bug_dataset_v3.py
----------------------------
v3: fixes the real bottleneck behind weak mutation-type counts found by
5-fold CV on v2 (swap_boolop n=3, flip_comparison 41%, wrong_constant_scale
55%). The old BugMutator mutated only the FIRST matching node per function,
so a function with 3 comparisons still only ever produced 1 distinct
flip_comparison mutant -- more attempts just rediscovered the same mutant
and got deduplicated away.

Fix: mutate a specific OCCURRENCE INDEX, not just "the first match". A
function with 3 comparisons now yields up to 3 distinct mutants for
flip_comparison alone. Combined with new functions written specifically
to be rich in the weak categories' patterns (multiple and/or, multiple
constant-scaled arithmetic, multiple comparisons), this directly targets
the CV breakdown rather than diluting with unrelated new domains.

Run:
    python generate_bug_dataset_v3.py bug_dataset_v3.jsonl
"""

import ast
import copy
import json
import random
import sys

random.seed(42)

# ---- v2's original function bank, unchanged ----
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

FORMATTING_FUNCTIONS = [
    "def truncate(s, length):\n    if len(s) <= length:\n        return s\n    return s[:length] + '...'",
    "def pad_left(s, width, char='0'):\n    if len(s) >= width:\n        return s\n    return char * (width - len(s)) + s",
    "def format_currency(amount):\n    return f'${amount:.2f}'",
    "def pluralize(word, count):\n    if count == 1:\n        return word\n    return word + 's'",
    "def format_duration(seconds):\n    minutes = seconds // 60\n    remaining = seconds % 60\n    return f'{minutes}m {remaining}s'",
]

# ---- NEW: functions written specifically DENSE in the weak categories'
# patterns (multiple and/or, multiple constant-scaled arithmetic, multiple
# comparisons) -- directly targets the CV breakdown, not a random new domain.
BOOLOP_RICH_FUNCTIONS = [
    "def is_valid_triangle(a, b, c):\n    return a + b > c and b + c > a and a + c > b",
    "def is_eligible(age, has_license, has_insurance):\n    return age >= 18 and has_license and has_insurance",
    "def is_business_hours(hour, is_weekday):\n    return hour >= 9 and hour < 17 and is_weekday",
    "def can_checkout(cart_total, is_member, has_valid_payment):\n    return cart_total > 0 and has_valid_payment and (is_member or cart_total >= 50)",
    "def is_valid_username(s):\n    return len(s) >= 3 and len(s) <= 20 and s.isalnum()",
    "def should_alert(temperature, humidity):\n    return temperature > 90 or humidity > 80",
    "def is_weekend_or_holiday(day_name, is_holiday):\n    return day_name in ('Saturday', 'Sunday') or is_holiday",
    "def can_vote(age, is_citizen, is_registered):\n    return age >= 18 and is_citizen and is_registered",
]

CONSTANT_SCALE_RICH_FUNCTIONS = [
    "def total_with_tax_and_tip(bill, tax_percent, tip_percent):\n    tax = bill * tax_percent / 100\n    tip = bill * tip_percent / 100\n    return bill + tax + tip",
    "def convert_temperature_range(low_c, high_c):\n    return (low_c * 9 / 5 + 32, high_c * 9 / 5 + 32)",
    "def annual_to_monthly_rate(annual_rate_percent):\n    return annual_rate_percent / 100 / 12",
    "def scale_image_dimensions(width, height, scale_percent):\n    return (width * scale_percent / 100, height * scale_percent / 100)",
    "def calculate_commission(sale_amount, commission_percent):\n    return sale_amount * commission_percent / 100",
]

COMPARISON_RICH_FUNCTIONS = [
    "def classify_bmi(bmi):\n    if bmi < 18.5:\n        return 'underweight'\n    if bmi < 25:\n        return 'normal'\n    if bmi < 30:\n        return 'overweight'\n    return 'obese'",
    "def grade_from_score(score):\n    if score >= 90:\n        return 'A'\n    if score >= 80:\n        return 'B'\n    if score >= 70:\n        return 'C'\n    return 'F'",
    "def compare_versions(a, b):\n    if a > b:\n        return 1\n    if a < b:\n        return -1\n    return 0",
    "def tax_bracket(income):\n    if income <= 10000:\n        return 0.10\n    if income <= 40000:\n        return 0.12\n    if income <= 85000:\n        return 0.22\n    return 0.24",
]

CLEAN_FUNCTIONS = (
    MATH_LIST_FUNCTIONS + PARSING_VALIDATION_FUNCTIONS
    + CONVERSION_FUNCTIONS + FORMATTING_FUNCTIONS
    + BOOLOP_RICH_FUNCTIONS + CONSTANT_SCALE_RICH_FUNCTIONS
    + COMPARISON_RICH_FUNCTIONS
)

FLIP_OPS = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE,
            ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
ARITH_SWAPS = {ast.Add: ast.Sub, ast.Sub: ast.Add,
               ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult}


class BugMutator(ast.NodeTransformer):
    """Mutates the Nth occurrence (0-indexed, ast.walk order) matching
    `mutation_type`, leaving every other candidate site untouched. This is
    the actual fix for weak-category starvation: a function with 3
    comparisons can now yield 3 distinct flip_comparison mutants instead
    of always just 1."""

    def __init__(self, mutation_type, occurrence_index):
        self.mutation_type = mutation_type
        self.occurrence_index = occurrence_index
        self.counter = -1
        self.applied = False

    def _is_target(self):
        self.counter += 1
        return self.counter == self.occurrence_index and not self.applied

    def visit_Compare(self, node):
        if self.mutation_type == "flip_comparison" and any(type(op) in FLIP_OPS for op in node.ops):
            if self._is_target():
                node.ops = [FLIP_OPS.get(type(op), type(op))() for op in node.ops]
                self.applied = True
        return self.generic_visit(node)

    def visit_If(self, node):
        if self.mutation_type == "remove_guard":
            if self._is_target():
                self.applied = True
                return None
            return self.generic_visit(node)
        if self.mutation_type == "silent_wrong_default":
            has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(node))
            if has_raise and self._is_target():
                self.applied = True
                return ast.Return(value=ast.Constant(value=None))
            return self.generic_visit(node)
        return self.generic_visit(node)

    def visit_BinOp(self, node):
        if self.mutation_type == "swap_arithmetic" and type(node.op) in ARITH_SWAPS:
            if self._is_target():
                node.op = ARITH_SWAPS[type(node.op)]()
                self.applied = True
            return self.generic_visit(node)
        if self.mutation_type == "wrong_constant_scale" and isinstance(node.op, (ast.Div, ast.Mult)) \
                and (isinstance(node.left, ast.Constant) or isinstance(node.right, ast.Constant)):
            if self._is_target():
                self.applied = True
                keep = node.left if isinstance(node.right, ast.Constant) else node.right
                return self.generic_visit(keep)
            return self.generic_visit(node)
        return self.generic_visit(node)

    def visit_BoolOp(self, node):
        if self.mutation_type == "swap_boolop":
            if self._is_target():
                node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
                self.applied = True
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if self.mutation_type == "off_by_one" and isinstance(node.value, int) and not isinstance(node.value, bool):
            if self._is_target():
                self.applied = True
                return ast.Constant(value=node.value + 1)
        return node


MUTATION_TYPES = [
    "flip_comparison", "remove_guard", "swap_arithmetic", "swap_boolop",
    "silent_wrong_default", "wrong_constant_scale", "off_by_one",
]


def _count_candidates(source: str, mutation_type: str) -> int:
    """How many distinct mutable sites this function has for this type."""
    tree = ast.parse(source)
    counter = BugMutator(mutation_type, occurrence_index=10 ** 9)  # never hit
    counter.visit(copy.deepcopy(tree))
    return counter.counter + 1


def _mutate_occurrence(source: str, mutation_type: str, occurrence_index: int) -> str | None:
    tree = ast.parse(source)
    mutator = BugMutator(mutation_type, occurrence_index)
    new_tree = mutator.visit(copy.deepcopy(tree))
    if not mutator.applied:
        return None
    ast.fix_missing_locations(new_tree)
    try:
        return ast.unparse(new_tree)
    except Exception:
        return None


def all_mutants(source: str) -> list[tuple[str, str]]:
    """Every distinct (mutant_code, mutation_type) this function can produce
    across all mutation types and all occurrence sites -- not just one."""
    results = []
    for mutation_type in MUTATION_TYPES:
        n = _count_candidates(source, mutation_type)
        for idx in range(n):
            mutant = _mutate_occurrence(source, mutation_type, idx)
            if mutant and mutant != source:
                results.append((mutant, mutation_type))
    return results


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else "bug_dataset_v3.jsonl"
    examples = []

    for source_id, raw_source in enumerate(CLEAN_FUNCTIONS):
        source = ast.unparse(ast.parse(raw_source))  # normalize, same as v2
        examples.append({"code": source, "label": 0, "source_id": source_id,
                          "mutation": "none"})

        mutants = all_mutants(source)
        seen = set()
        for mutant_code, mutation_type in mutants:
            if mutant_code not in seen:
                seen.add(mutant_code)
                examples.append({"code": mutant_code, "label": 1,
                                  "source_id": source_id, "mutation": mutation_type})

    random.shuffle(examples)

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    n_clean = sum(1 for e in examples if e["label"] == 0)
    n_buggy = sum(1 for e in examples if e["label"] == 1)
    print(f"Generated {len(examples)} examples ({n_clean} clean, {n_buggy} buggy)")
    print(f"From {len(CLEAN_FUNCTIONS)} base functions across {len(MUTATION_TYPES)} mutation types\n")

    print("Examples per mutation type:")
    from collections import Counter
    counts = Counter(e["mutation"] for e in examples)
    for mtype, count in sorted(counts.items(), key=lambda kv: kv[1]):
        print(f"  {mtype:22s} {count}")

    print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    main()