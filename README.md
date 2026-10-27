# Different Conjunctions of Conditional Events

An English Streamlit application for comparing several conjunctions of two
conditional events (A|H) and (B|K).

For a precise assessment

\[
x=P(A|H),\qquad y=P(B|K),
\]

the application computes coherent lower and upper bounds for the prevision
(z) of:

- the Kleene--Lukasiewicz--Heyting/de Finetti conjunction \(\wedge_K\);
- the Łukasiewicz conjunction \(\wedge_L\);
- the Bochvar internal conjunction \(\wedge_B\);
- the Sobociński/quasi-conjunction \(\wedge_S\);
- the Gilio--Sanfilippo conditional-random-quantity conjunction \(\wedge_{gs}\).

The four trivalent conjunctions are treated as further conditional events. Their
intervals \([z',z'']\) are computed with the finite precise-assessment
extension algorithm based on constituents and linear programming. The app
displays the cycles of the algorithm, the endpoint tests, the (M_j) values,
and the systems in the \(\lambda_r\) variables.

The \(gs\) conjunction is five-valued. Its coherent bounds are obtained by
the equivalent convex-hull linear programme, using the values

\[
1,\quad 0,\quad x,\quad y,\quad z
\]

according to the constituent. The all-void constituent (C_0) is excluded.

## Run locally

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

Boolean relations can be entered one per line. For example,
`not (A and B)` expresses incompatibility and `A implies B` is interpreted as
`A and not B = False`.

If the supplied assessment is incoherent or an extension system is unsolvable,
the app still displays every logically constructible conjunction and its
constituents. It reports the numerical bounds as unavailable instead of
suppressing the object. For example, if the same event is entered twice with
incompatible probabilities, the intersection object is still shown while the
inconsistent assessment is reported.
