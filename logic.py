"""Coherence calculations for several conjunctions of conditional events.

The extension algorithm implemented here is the finite, precise version of the
fundamental theorem algorithm described in the supplied manuscript.  It is
used for the four trivalent conjunctions.  The Gilio--Sanfilippo conjunction
is a conditional random quantity and is handled by its equivalent convex-hull
linear programme.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from itertools import combinations, product
from typing import Sequence

import numpy as np
from scipy.optimize import linprog


TOL = 1e-9


@dataclass(frozen=True)
class ConditionalEvent:
    event: str
    condition: str
    probability: float
    label: str


@dataclass(frozen=True)
class Constituent:
    statuses: tuple[str, ...]
    witnesses: tuple[dict[str, bool], ...]


@dataclass
class ExtensionCycle:
    direction: str
    family_labels: tuple[str, ...]
    constituents: tuple[Constituent, ...]
    endpoint: float
    endpoint_feasible: bool
    masses: tuple[float, ...] | None
    reduced_labels: tuple[str, ...] | None = None
    bound: float | None = None


@dataclass
class BoundResult:
    name: str
    symbol: str
    formula: str
    lower: float | None
    upper: float | None
    target: ConditionalEvent | None
    constituents: tuple[Constituent, ...]
    operation_values: tuple[float, ...] | None
    lower_cycles: list[ExtensionCycle]
    upper_cycles: list[ExtensionCycle]
    method: str
    error: str | None = None


class ExtensionFailure(ValueError):
    """An extension bound could not be obtained, but its construction exists."""

    def __init__(self, message: str, cycles: list[ExtensionCycle]):
        super().__init__(message)
        self.cycles = cycles


CONJUNCTIONS = {
    "K": {
        "title": "Kleene–Lukasiewicz–Heyting (de Finetti)",
        "formula": r"(A|H)\wedge_K(B|K)=AHBK\mid(AHBK\vee\overline{A}H\vee\overline{B}K)",
    },
    "L": {
        "title": "Łukasiewicz",
        "formula": r"(A|H)\wedge_L(B|K)=AHBK\mid(AHBK\vee\overline{A}H\vee\overline{B}K\vee\overline{H}\,\overline{K})",
    },
    "B": {
        "title": "Bochvar internal (Kleene weak)",
        "formula": r"(A|H)\wedge_B(B|K)=AHBK\mid HK",
    },
    "S": {
        "title": "Sobociński (quasi-conjunction)",
        "formula": r"(A|H)\wedge_S(B|K)=((AH\vee\overline{H})(BK\vee\overline{K}))\mid(H\vee K)",
    },
    "gs": {
        "title": "Gilio–Sanfilippo conditional random quantity",
        "formula": r"(A|H)\wedge_{gs}(B|K)=(AHBK+x\,\overline{H}BK+y\,AH\overline{K})\mid(H\vee K)",
    },
}


def _parse(formula: str) -> ast.AST:
    formula = formula.strip() or "True"
    try:
        tree = ast.parse(formula, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Invalid Boolean formula {formula!r}.") from exc
    allowed = (
        ast.Name,
        ast.Constant,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.Not,
        ast.Load,
    )
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        raise ValueError(
            f"Unsupported syntax in {formula!r}. Use and, or, not, and parentheses."
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
            raise ValueError(f"Only Boolean constants are allowed in {formula!r}.")
    return tree


def _names(formula: str) -> set[str]:
    return {node.id for node in ast.walk(_parse(formula)) if isinstance(node, ast.Name)}


def _evaluate(tree: ast.AST, assignment: dict[str, bool]) -> bool:
    if isinstance(tree, ast.Name):
        return assignment[tree.id]
    if isinstance(tree, ast.Constant):
        return bool(tree.value)
    if isinstance(tree, ast.UnaryOp) and isinstance(tree.op, ast.Not):
        return not _evaluate(tree.operand, assignment)
    if isinstance(tree, ast.BoolOp) and isinstance(tree.op, ast.And):
        return all(_evaluate(value, assignment) for value in tree.values)
    if isinstance(tree, ast.BoolOp) and isinstance(tree.op, ast.Or):
        return any(_evaluate(value, assignment) for value in tree.values)
    raise ValueError("Unsupported Boolean expression.")


def _implicant_covers(implicant: tuple[str, ...], minterm: tuple[str, ...]) -> bool:
    return all(symbol == "-" or symbol == value for symbol, value in zip(implicant, minterm))


def _combine_implicants(
    first: tuple[str, ...], second: tuple[str, ...]
) -> tuple[str, ...] | None:
    differences = [index for index, (left, right) in enumerate(zip(first, second)) if left != right]
    if len(differences) != 1:
        return None
    index = differences[0]
    if first[index] == "-" or second[index] == "-":
        return None
    combined = list(first)
    combined[index] = "-"
    return tuple(combined)


def _minimal_dnf(names: tuple[str, ...], true_minterms: set[tuple[str, ...]]) -> str:
    """Return a compact disjunctive normal form for a small Boolean function."""
    if not true_minterms:
        return "False"
    total = set(product(("0", "1"), repeat=len(names)))
    if true_minterms == total:
        return "True"

    current = set(true_minterms)
    primes: set[tuple[str, ...]] = set()
    while current:
        used: set[tuple[str, ...]] = set()
        next_round: set[tuple[str, ...]] = set()
        ordered = sorted(current)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                combined = _combine_implicants(left, right)
                if combined is not None:
                    used.add(left)
                    used.add(right)
                    next_round.add(combined)
        primes.update(current - used)
        current = next_round

    prime_list = sorted(primes, key=lambda item: (sum(value != "-" for value in item), item))
    covers = {
        implicant: {
            minterm for minterm in true_minterms if _implicant_covers(implicant, minterm)
        }
        for implicant in prime_list
    }
    uncovered = set(true_minterms)
    selected: list[tuple[str, ...]] = []
    while uncovered:
        candidate = max(
            (implicant for implicant in prime_list if implicant not in selected),
            key=lambda item: (len(covers[item] & uncovered), -sum(value != "-" for value in item)),
            default=None,
        )
        if candidate is None:
            break
        selected.append(candidate)
        uncovered -= covers[candidate]

    # For the small formulas used for conjunction objects, choose the shortest
    # exact cover. The greedy cover above is the fallback for larger tables.
    if len(prime_list) <= 24:
        for number in range(1, len(prime_list) + 1):
            exact = []
            for combination_of_primes in combinations(prime_list, number):
                covered = set().union(*(covers[item] for item in combination_of_primes))
                if covered == true_minterms:
                    exact.append(combination_of_primes)
            if exact:
                selected = min(
                    exact,
                    key=lambda item: sum(sum(value != "-" for value in term) for term in item),
                )
                break

    terms = []
    for implicant in selected:
        literals = [
            name if value == "1" else f"not ({name})"
            for name, value in zip(names, implicant)
            if value != "-"
        ]
        terms.append(" and ".join(literals) if literals else "True")
    if len(terms) == 1:
        return terms[0]
    return " or ".join(f"({term})" for term in terms)


def simplify_boolean(formula: str) -> str:
    """Simplify a Boolean formula used to display a conjunction object.

    A truth-table minimisation is used for formulas with at most eight primitive
    events. Larger user formulas are left unchanged so that simplification
    never causes an impractical enumeration.
    """
    tree = _parse(formula)
    names = tuple(sorted(_names(formula)))
    if len(names) > 8:
        return formula.strip() or "True"
    true_minterms = set()
    for values in product((False, True), repeat=len(names)):
        assignment = dict(zip(names, values))
        if _evaluate(tree, assignment):
            true_minterms.add(tuple("1" if value else "0" for value in values))
    return _minimal_dnf(names, true_minterms)


def _parse_relation(relation: str) -> ast.AST:
    relation = relation.strip()
    implication = re.fullmatch(r"(.+?)\s+implies\s+(.+)", relation, flags=re.I)
    if implication:
        left = _parse(implication.group(1))
        right = _parse(implication.group(2))
        return ast.BoolOp(
            op=ast.Or(),
            values=[ast.UnaryOp(op=ast.Not(), operand=left), right],
        )
    match = re.fullmatch(r"(.+?)\s*=\s*(True|False)", relation, flags=re.I)
    if match:
        tree = _parse(match.group(1))
        return tree if match.group(2).lower() == "true" else ast.UnaryOp(
            op=ast.Not(), operand=tree
        )
    return _parse(relation)


def _status(event: bool, condition: bool) -> str:
    if not condition:
        return "V"
    return "T" if event else "F"


def build_constituents(
    events: Sequence[ConditionalEvent],
    relations: Sequence[str] = (),
) -> tuple[Constituent, ...]:
    if not events:
        raise ValueError("No conditional events were supplied.")
    formulas = [formula for event in events for formula in (event.event, event.condition)]
    trees = [_parse(formula) for formula in formulas]
    relation_trees = [_parse_relation(r) for r in relations if r.strip()]
    primitive_names = set().union(*(_names(formula) for formula in formulas))
    primitive_names.update(
        node.id
        for tree in relation_trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    )

    groups: dict[tuple[str, ...], list[dict[str, bool]]] = {}
    for values in product((False, True), repeat=len(primitive_names)):
        assignment = dict(zip(sorted(primitive_names), values))
        if any(not _evaluate(tree, assignment) for tree in relation_trees):
            continue
        evaluated = [_evaluate(tree, assignment) for tree in trees]
        statuses = tuple(
            _status(evaluated[2 * index], evaluated[2 * index + 1])
            for index in range(len(events))
        )
        if all(item == "V" for item in statuses):
            continue  # C_0: all conditioning events are false.
        groups.setdefault(statuses, []).append(assignment)

    order = {"T": 0, "F": 1, "V": 2}
    return tuple(
        Constituent(statuses, tuple(witnesses))
        for statuses, witnesses in sorted(
            groups.items(), key=lambda pair: tuple(order[item] for item in pair[0])
        )
    )


def _event_row(constituents: Sequence[Constituent], index: int, probability: float) -> np.ndarray:
    return np.asarray(
        [
            (1.0 if item.statuses[index] == "T" else 0.0)
            - probability * (1.0 if item.statuses[index] != "V" else 0.0)
            for item in constituents
        ],
        dtype=float,
    )


def _condition_mass(constituents: Sequence[Constituent], index: int) -> np.ndarray:
    return np.asarray(
        [1.0 if item.statuses[index] != "V" else 0.0 for item in constituents],
        dtype=float,
    )


def _numerator_mass(constituents: Sequence[Constituent], index: int) -> np.ndarray:
    return np.asarray(
        [1.0 if item.statuses[index] == "T" else 0.0 for item in constituents],
        dtype=float,
    )


def _solve(
    objective: np.ndarray,
    A_eq: Sequence[np.ndarray],
    b_eq: Sequence[float],
    maximize: bool = False,
) -> tuple[bool, float | None, np.ndarray | None]:
    vector = -objective if maximize else objective
    result = linprog(
        vector,
        A_eq=np.asarray(A_eq, dtype=float),
        b_eq=np.asarray(b_eq, dtype=float),
        bounds=[(0.0, None)] * len(objective),
        method="highs",
    )
    if not result.success or result.x is None:
        return False, None, None
    value = float(-result.fun if maximize else result.fun)
    return True, value, result.x


def _initial_constraints(
    events: Sequence[ConditionalEvent], constituents: Sequence[Constituent]
) -> tuple[list[np.ndarray], list[float]]:
    rows = [_event_row(constituents, index, event.probability) for index, event in enumerate(events)]
    rows.append(np.ones(len(constituents)))
    return rows, [0.0] * len(events) + [1.0]


def initial_assessment_is_coherent(
    events: Sequence[ConditionalEvent], relations: Sequence[str] = ()
) -> tuple[bool, str]:
    """Check the initial assessment, including its proper subfamilies."""
    for size in range(1, len(events) + 1):
        for indices in combinations(range(len(events)), size):
            family = tuple(events[index] for index in indices)
            constituents = build_constituents(family, relations)
            if not constituents:
                return False, "A conditioning event is logically impossible."
            rows, rhs = _initial_constraints(family, constituents)
            feasible, _, _ = _solve(np.zeros(len(constituents)), rows, rhs)
            if not feasible:
                labels = ", ".join(event.label for event in family)
                return False, f"The assessment on {{{labels}}} is incoherent."
    return True, "The initial assessment is coherent."


def _target_events(
    first: ConditionalEvent, second: ConditionalEvent, kind: str
) -> ConditionalEvent:
    A, H, B, K = first.event, first.condition, second.event, second.condition
    if kind == "K":
        event = f"(({A}) and ({H}) and ({B}) and ({K}))"
        condition = (
            f"(({A}) and ({H}) and ({B}) and ({K})) or "
            f"(not ({A}) and ({H})) or (not ({B}) and ({K}))"
        )
    elif kind == "L":
        event = f"(({A}) and ({H}) and ({B}) and ({K}))"
        condition = (
            f"(({A}) and ({H}) and ({B}) and ({K})) or "
            f"(not ({A}) and ({H})) or (not ({B}) and ({K})) or "
            f"(not ({H}) and not ({K}))"
        )
    elif kind == "B":
        event = f"(({A}) and ({H}) and ({B}) and ({K}))"
        condition = f"(({H}) and ({K}))"
    elif kind == "S":
        event = f"((({A}) and ({H})) or (not ({H}))) and ((({B}) and ({K})) or (not ({K})))"
        condition = f"(({H}) or ({K}))"
    else:
        raise ValueError(f"Unknown trivalent conjunction {kind!r}.")
    return ConditionalEvent(
        simplify_boolean(event),
        simplify_boolean(condition),
        0.0,
        f"wedge_{kind}",
    )


def _endpoint_constraints(
    events: Sequence[ConditionalEvent],
    constituents: Sequence[Constituent],
    endpoint: float,
) -> tuple[list[np.ndarray], list[float]]:
    initial = events[:-1]
    rows, rhs = _initial_constraints(initial, constituents)
    target_index = len(events) - 1
    rows.insert(
        len(initial),
        _numerator_mass(constituents, target_index)
        - endpoint * _condition_mass(constituents, target_index),
    )
    rhs.insert(len(initial), 0.0)
    return rows, rhs


def _normalised_target_constraints(
    events: Sequence[ConditionalEvent],
    constituents: Sequence[Constituent],
) -> tuple[list[np.ndarray], list[float]]:
    initial = events[:-1]
    rows, rhs = _initial_constraints(initial, constituents)
    target_index = len(events) - 1
    rows.insert(len(initial), _condition_mass(constituents, target_index))
    rhs.insert(len(initial), 1.0)
    return rows, rhs


def _extension_bound(
    initial_events: Sequence[ConditionalEvent],
    target: ConditionalEvent,
    relations: Sequence[str],
    lower: bool,
) -> tuple[float, list[ExtensionCycle]]:
    active_indices = list(range(len(initial_events)))
    direction = "lower" if lower else "upper"
    cycles: list[ExtensionCycle] = []

    for _ in range(len(initial_events) + 2):
        current_initial = tuple(initial_events[index] for index in active_indices)
        current_events = current_initial + (target,)
        constituents = build_constituents(current_events, relations)
        target_index = len(current_events) - 1
        if not constituents or not any(
            item.statuses[target_index] != "V" for item in constituents
        ):
            raise ValueError("The target conditioning event is logically impossible.")

        endpoint = 0.0 if lower else 1.0
        endpoint_rows, endpoint_rhs = _endpoint_constraints(
            current_events, constituents, endpoint
        )
        feasible, _, _ = _solve(
            np.zeros(len(constituents)), endpoint_rows, endpoint_rhs
        )
        if feasible:
            masses: list[float] = []
            for index in range(len(current_events)):
                ok, value, _ = _solve(
                    _condition_mass(constituents, index),
                    endpoint_rows,
                    endpoint_rhs,
                    maximize=True,
                )
                if not ok or value is None:
                    value = 0.0
                masses.append(value)
            target_mass = masses[-1]
            zero_initial = [
                active_indices[position]
                for position, value in enumerate(masses[:-1])
                if value <= TOL
            ]
            if target_mass > TOL or not zero_initial:
                cycles.append(
                    ExtensionCycle(
                        direction,
                        tuple(event.label for event in current_events),
                        constituents,
                        endpoint,
                        True,
                        tuple(masses),
                        bound=endpoint,
                    )
                )
                return endpoint, cycles
            cycles.append(
                ExtensionCycle(
                    direction,
                    tuple(event.label for event in current_events),
                    constituents,
                    endpoint,
                    True,
                    tuple(masses),
                    reduced_labels=tuple(initial_events[index].label for index in zero_initial)
                    + (target.label,),
                )
            )
            active_indices = zero_initial
            continue

        rows, rhs = _normalised_target_constraints(current_events, constituents)
        objective = _numerator_mass(constituents, target_index)
        ok, value, _ = _solve(objective, rows, rhs, maximize=not lower)
        if not ok or value is None:
            raise ExtensionFailure("The extension system is unsolvable.", cycles)
        cycles.append(
            ExtensionCycle(
                direction,
                tuple(event.label for event in current_events),
                constituents,
                endpoint,
                False,
                None,
                bound=value,
            )
        )
        return value, cycles

    raise ExtensionFailure("The extension algorithm did not terminate.", cycles)


def _gs_value(statuses: tuple[str, str], x: float, y: float) -> float:
    if statuses == ("T", "T"):
        return 1.0
    if statuses == ("V", "T"):
        return x
    if statuses == ("T", "V"):
        return y
    return 0.0


def gs_bounds(
    first: ConditionalEvent,
    second: ConditionalEvent,
    relations: Sequence[str] = (),
) -> BoundResult:
    constituents = build_constituents((first, second), relations)
    if not constituents:
        raise ValueError("Every constituent is C_0: both conditioning events are impossible.")
    values = np.asarray(
        [_gs_value(item.statuses, first.probability, second.probability) for item in constituents]
    )
    coherent, message = initial_assessment_is_coherent((first, second), relations)
    if not coherent:
        return BoundResult(
            "gs",
            "z",
            CONJUNCTIONS["gs"]["formula"],
            None,
            None,
            None,
            constituents,
            tuple(float(value) for value in values),
            [],
            [],
            "Direct convex-hull linear programming for the five-valued conditional random quantity.",
            error=message,
        )
    rows, rhs = _initial_constraints((first, second), constituents)
    ok_lower, lower, _ = _solve(values, rows, rhs)
    ok_upper, upper, _ = _solve(values, rows, rhs, maximize=True)
    if not ok_lower or not ok_upper or lower is None or upper is None:
        return BoundResult(
            "gs",
            "z",
            CONJUNCTIONS["gs"]["formula"],
            None,
            None,
            None,
            constituents,
            tuple(float(value) for value in values),
            [],
            [],
            "Direct convex-hull linear programming for the five-valued conditional random quantity.",
            error="The convex-hull system is unsolvable.",
        )
    lower = max(0.0, min(1.0, lower))
    upper = max(0.0, min(1.0, upper))
    cycle = ExtensionCycle(
        "direct convex-hull LP",
        (first.label, second.label),
        constituents,
        0.0,
        True,
        None,
        bound=None,
    )
    return BoundResult(
        "gs",
        "z",
        CONJUNCTIONS["gs"]["formula"],
        lower,
        upper,
        None,
        constituents,
        tuple(float(value) for value in values),
        [cycle],
        [cycle],
        "Direct convex-hull linear programming for the five-valued conditional random quantity.",
    )


def conjunction_bounds(
    first: ConditionalEvent,
    second: ConditionalEvent,
    kind: str,
    relations: Sequence[str] = (),
) -> BoundResult:
    if kind == "gs":
        return gs_bounds(first, second, relations)
    if kind not in {"K", "L", "B", "S"}:
        raise ValueError(f"Unknown conjunction {kind!r}.")
    target = _target_events(first, second, kind)
    constituents = build_constituents((first, second, target), relations)
    values = tuple(
        1.0 if item.statuses[2] == "T" else 0.0 if item.statuses[2] == "F" else np.nan
        for item in constituents
    )
    coherent, message = initial_assessment_is_coherent((first, second), relations)
    if not coherent:
        return BoundResult(
            kind,
            "z",
            CONJUNCTIONS[kind]["formula"],
            None,
            None,
            target,
            constituents,
            values,
            [],
            [],
            "Fundamental-theorem extension algorithm for a further conditional event.",
            error=message,
        )

    if target.event == "False":
        return BoundResult(
            kind,
            "z",
            CONJUNCTIONS[kind]["formula"],
            0.0,
            0.0,
            target,
            constituents,
            values,
            [],
            [],
            "Exact value from coherence: the conjunction is the impossible event "
            r"\(\bot\), whose prevision is 0.",
        )

    lower: float | None = None
    upper: float | None = None
    lower_cycles: list[ExtensionCycle] = []
    upper_cycles: list[ExtensionCycle] = []
    errors: list[str] = []
    try:
        lower, lower_cycles = _extension_bound(
            (first, second), target, relations, lower=True
        )
        lower = max(0.0, min(1.0, lower))
    except ExtensionFailure as error:
        lower_cycles = error.cycles
        errors.append(f"Lower bound: {error}")
    try:
        upper, upper_cycles = _extension_bound(
            (first, second), target, relations, lower=False
        )
        upper = max(0.0, min(1.0, upper))
    except ExtensionFailure as error:
        upper_cycles = error.cycles
        errors.append(f"Upper bound: {error}")
    return BoundResult(
        kind,
        "z",
        CONJUNCTIONS[kind]["formula"],
        lower,
        upper,
        target,
        constituents,
        values,
        lower_cycles,
        upper_cycles,
        "Fundamental-theorem extension algorithm for a further conditional event.",
        error=" ".join(errors) if errors else None,
    )
