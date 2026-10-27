"""Streamlit interface for different conjunctions of conditional events."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logic import (  # noqa: E402
    CONJUNCTIONS,
    BoundResult,
    ConditionalEvent,
    ExtensionCycle,
    conjunction_bounds,
    simplify_boolean,
)


st.set_page_config(
    page_title="Different Conjunctions of Conditional Events",
    page_icon="∧",
    layout="wide",
)


def _number(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.8f}"


def _boolean_latex(formula: str) -> str:
    """Make the Boolean syntax readable in a displayed conditional object."""
    text = formula
    text = text.replace(" and ", r" \land ")
    text = text.replace(" or ", r" \lor ")
    text = text.replace("not ", r"\neg ")
    text = text.replace("True", r"\top").replace("False", r"\bot")
    return text


def _gs_object_latex(
    first: ConditionalEvent, second: ConditionalEvent
) -> str:
    """Display the concrete five-valued gs object for the entered formulas."""
    ab = simplify_boolean(
        f"({first.event}) and ({first.condition}) and "
        f"({second.event}) and ({second.condition})"
    )
    x_term = simplify_boolean(
        f"not ({first.condition}) and ({second.event}) and ({second.condition})"
    )
    y_term = simplify_boolean(
        f"({first.event}) and ({first.condition}) and not ({second.condition})"
    )
    condition = simplify_boolean(f"({first.condition}) or ({second.condition})")

    pieces = []
    if ab != "False":
        pieces.append("1" if ab == "True" else _boolean_latex(ab))
    if x_term != "False":
        pieces.append("x" if x_term == "True" else rf"x\,{_boolean_latex(x_term)}")
    if y_term != "False":
        pieces.append("y" if y_term == "True" else rf"y\,{_boolean_latex(y_term)}")
    random_quantity = "0" if not pieces else "+".join(pieces)
    return rf"\left({random_quantity}\right)\mid {_boolean_latex(condition)}"


def _status_text(status: str) -> str:
    return {"T": "true", "F": "false", "V": "void"}[status]


def _system_latex(
    cycle: ExtensionCycle,
    probabilities: tuple[float, float],
    endpoint_system: bool,
) -> str:
    count = len(cycle.constituents)
    rows: list[str] = []
    initial_count = len(cycle.family_labels) - 1
    for index in range(initial_count):
        true_terms = [
            rf"\lambda_{{{number}}}"
            for number, item in enumerate(cycle.constituents, start=1)
            if item.statuses[index] == "T"
        ]
        active_terms = [
            rf"\lambda_{{{number}}}"
            for number, item in enumerate(cycle.constituents, start=1)
            if item.statuses[index] != "V"
        ]
        left = " + ".join(true_terms) or "0"
        right = " + ".join(active_terms) or "0"
        rows.append(rf"{left}={probabilities[index]:.6g}({right})")

    target_index = initial_count
    target_true = [
        rf"\lambda_{{{number}}}"
        for number, item in enumerate(cycle.constituents, start=1)
        if item.statuses[target_index] == "T"
    ]
    target_active = [
        rf"\lambda_{{{number}}}"
        for number, item in enumerate(cycle.constituents, start=1)
        if item.statuses[target_index] != "V"
    ]
    target_left = " + ".join(target_true) or "0"
    target_right = " + ".join(target_active) or "0"
    if endpoint_system:
        rows.append(rf"{target_left}={cycle.endpoint:g}({target_right})")
    else:
        rows.append(rf"{target_right}=1")
    rows.append(rf"\lambda_1+\cdots+\lambda_{{{count}}}=1")
    rows.append(rf"\lambda_r\geq0,\quad r=1,\ldots,{count}")
    body = r" \\[3pt] ".join(rows)
    return rf"\left\{{\begin{{aligned}}{body}\end{{aligned}}\right."


def _constituent_table(result: BoundResult) -> None:
    rows = []
    for number, item in enumerate(result.constituents, start=1):
        row = {"constituent": f"C{number}"}
        if len(item.statuses) == 3:
            row["A|H"] = _status_text(item.statuses[0])
            row["B|K"] = _status_text(item.statuses[1])
            row["conjunction"] = _status_text(item.statuses[2])
            if result.operation_values is not None:
                value = result.operation_values[number - 1]
                row["value"] = "z" if value != value else value
        else:
            row["A|H"] = _status_text(item.statuses[0])
            row["B|K"] = _status_text(item.statuses[1])
            row["value"] = result.operation_values[number - 1] if result.operation_values else "—"
        row["witness"] = str(item.witnesses[0])
        rows.append(row)
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _show_cycle(
    cycle: ExtensionCycle,
    probabilities: tuple[float, float],
    cycle_number: int,
) -> None:
    st.markdown(
        f"**Cycle {cycle_number}.** Family: "
        + ", ".join(f"`{label}`" for label in cycle.family_labels)
    )
    st.caption(
        "The displayed statuses are T (true), F (false), and V (void); "
        "the all-void constituent C₀ is excluded."
    )
    rows = []
    for number, item in enumerate(cycle.constituents, start=1):
        rows.append(
            {
                "constituent": f"C{number}",
                **{
                    label: status
                    for label, status in zip(cycle.family_labels, item.statuses)
                },
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.markdown("The linear system used in this cycle is")
    st.latex(_system_latex(cycle, probabilities, cycle.endpoint_feasible))
    if cycle.endpoint_feasible and cycle.masses is not None:
        masses = ", ".join(
            f"M_{{{label}}}={value:.6g}"
            for label, value in zip(cycle.family_labels, cycle.masses)
        )
        st.markdown(f"Step 3 gives: ${masses}$. ")
        if cycle.reduced_labels:
            st.info(
                "The endpoint system is solvable, but the target condition and "
                "some initial conditioning events have zero maximal mass. "
                "The algorithm continues with: "
                + ", ".join(f"`{label}`" for label in cycle.reduced_labels)
                + "."
            )
        else:
            st.success(
                f"The endpoint z={cycle.endpoint:g} is admitted by the algorithm."
            )
    else:
        objective = "min" if cycle.direction == "lower" else "max"
        bound_symbol = r"z'" if cycle.direction == "lower" else r"z''"
        st.latex(
            rf"{bound_symbol}="
            rf"{objective}\;\sum_{{r:C_r\subseteq E_{{n+1}}H_{{n+1}}}}\lambda_r"
        )
        st.success(
            f"Step 2 returns {cycle.direction} bound "
            f"{cycle.bound:.8f}."
        )


def _render_result(
    result: BoundResult,
    first: ConditionalEvent,
    second: ConditionalEvent,
) -> None:
    st.subheader(f"{result.name} — {CONJUNCTIONS[result.name]['title']}")
    st.latex(result.formula)
    if result.lower is not None and result.upper is not None:
        st.success(
            f"Coherent interval for z: [{result.lower:.8f}, {result.upper:.8f}]"
        )
    else:
        st.warning(
            "The object is displayed below, but its coherent bounds could not "
            "be computed from the supplied assessment."
        )
    if result.error:
        st.info(f"Bounds status: {result.error}")
    st.markdown(f"**Method:** {result.method}")

    if result.target is not None:
        st.markdown("**Object for the entered events**")
        st.latex(
            rf"{_boolean_latex(result.target.event)}\mid"
            rf"{_boolean_latex(result.target.condition)}"
        )
        st.caption(
            "This logical object is shown even when the extension system is "
            "unsolvable; only the numerical bounds are unavailable."
        )
    elif result.name == "gs":
        st.markdown("**Object for the entered events**")
        st.latex(_gs_object_latex(first, second))
        st.caption(
            "This five-valued object is shown even when its convex-hull system "
            "is unsolvable."
        )

    with st.expander("Constituents and numerical values", expanded=False):
        _constituent_table(result)

    if result.name == "gs":
        st.latex(
            r"z=\min\sum_r v_r\lambda_r,\qquad "
            r"z''=\max\sum_r v_r\lambda_r"
        )
        st.caption(
            "Here vᵣ is 1, 0, x, or y according to the five-valued "
            "Gilio–Sanfilippo conjunction; C₀ is discarded."
        )
        return

    probabilities = (first.probability, second.probability)
    with st.expander("Extension algorithm for the lower bound z′", expanded=False):
        st.markdown(
            "Step 0 builds the constituents of the initial family plus the "
            "further conditional event. Step 1 tests z=0. Step 2 solves a "
            "linear programme if z=0 is impossible. Step 3 removes the "
            "conditioning events with zero maximal mass and repeats the cycle."
        )
        for number, cycle in enumerate(result.lower_cycles, start=1):
            _show_cycle(cycle, probabilities, number)
        if not result.lower_cycles:
            st.caption("No lower-bound extension cycle was available.")
    with st.expander("Extension algorithm for the upper bound z′′", expanded=False):
        st.markdown(
            "The same algorithm is applied with the endpoint z=1 and, when "
            "necessary, with the maximisation objective."
        )
        for number, cycle in enumerate(result.upper_cycles, start=1):
            _show_cycle(cycle, probabilities, number)
        if not result.upper_cycles:
            st.caption("No upper-bound extension cycle was available.")


st.title("Different Conjunctions of Conditional Events")
st.markdown(
    "This application compares four trivalent conjunctions with the "
    "Gilio–Sanfilippo conjunction of two conditional events. For each precise "
    "assessment x=P(A|H) and y=P(B|K), it computes the coherent lower and "
    "upper bounds for the prevision z of every selected conjunction."
)
st.latex(r"A|H\in\{1,0,x\},\quad B|K\in\{1,0,y\},\quad x=P(A|H),\ y=P(B|K)")
st.markdown(
    "The symbols K, L, B, and S denote the four trivalent logics discussed in "
    "the manuscript. The symbol gs denotes the five-valued conditional random "
    "quantity introduced by Gilio and Sanfilippo."
)

st.subheader("1. Enter the two conditional events")
left, right = st.columns(2)
with left:
    st.markdown("**First conditional event**")
    first_event = st.text_input("Event A", value="A")
    first_condition = st.text_input("Condition H", value="H")
    first_probability = st.number_input(
        "P(A|H)", min_value=0.0, max_value=1.0, value=0.6, step=0.05, format="%.4f"
    )
with right:
    st.markdown("**Second conditional event**")
    second_event = st.text_input("Event B", value="B")
    second_condition = st.text_input("Condition K", value="K")
    second_probability = st.number_input(
        "P(B|K)", min_value=0.0, max_value=1.0, value=0.4, step=0.05, format="%.4f"
    )

st.subheader("2. Optional logical relations")
relations_text = st.text_area(
    "One relation per line",
    value="",
    help="Use and, or, not, parentheses, and optionally = True or = False.",
)
st.caption(
    "Examples: `not (A and B)` means incompatibility; `A implies B` means "
    "A and not B is False. Relations are used when constructing the constituents."
)
relations = tuple(line.strip() for line in relations_text.splitlines() if line.strip())

st.subheader("3. Select conjunctions")
selected = st.multiselect(
    "Conjunctions to calculate",
    options=("K", "L", "B", "S", "gs"),
    default=("K", "L", "B", "S", "gs"),
    format_func=lambda key: f"{key}: {CONJUNCTIONS[key]['title']}",
)

if st.button("Compute coherent lower and upper bounds", type="primary"):
    first = ConditionalEvent(
        first_event, first_condition, float(first_probability), "A|H"
    )
    second = ConditionalEvent(
        second_event, second_condition, float(second_probability), "B|K"
    )
    if not selected:
        st.warning("Select at least one conjunction.")
    else:
        try:
            results = []
            calculation_errors = []
            for kind in selected:
                try:
                    results.append(conjunction_bounds(first, second, kind, relations))
                except Exception as error:
                    calculation_errors.append(f"{kind}: {error}")
            for message in calculation_errors:
                st.error(f"Input error: {message}")
            if not results:
                st.stop()
            st.subheader("Summary of conjunction objects and coherent extensions")
            st.dataframe(
                [
                    {
                        "conjunction": f"{result.name}: {CONJUNCTIONS[result.name]['title']}",
                        "lower bound z′": _number(result.lower),
                        "upper bound z′′": _number(result.upper),
                        "status": "available" if result.error is None else result.error,
                    }
                    for result in results
                ],
                hide_index=True,
                use_container_width=True,
            )
            for result in results:
                st.divider()
                _render_result(result, first, second)
        except Exception as error:
            st.error(f"Input error: {error}")
