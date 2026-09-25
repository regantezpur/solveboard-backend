"""
SolveBoard backend — Phase 2
-----------------------------
A tiny, free, open-source API that solves a math problem typed in as text
(e.g. "2x + 5 = 17" or "1/4 + 1/2") and returns a step-by-step "script"
in the same {display, spoken} format the SolveBoard whiteboard already
understands.

No AI model is used here — every number comes from SymPy, an open-source
symbolic math library, so answers are always exactly correct. This keeps
the whole thing free to run (no per-call API cost).

Endpoints:
  GET  /            -> health check
  POST /solve       -> { "problem": "2x + 5 = 17" }  ->  step-script JSON
"""

import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
)

app = FastAPI(title="SolveBoard Solver")

# Allow the Netlify-hosted frontend (or any site) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
X = sp.symbols("x")


class ProblemIn(BaseModel):
    problem: str


def parse_side(text: str):
    text = text.replace("^", "**")
    return parse_expr(text, transformations=TRANSFORMS)


def fmt(n):
    """Pretty-print a sympy number: whole numbers without .0, fractions as a/b."""
    n = sp.nsimplify(n)
    if n.is_Integer:
        return str(n)
    if n.is_Rational:
        return f"{n.p}/{n.q}"
    return str(sp.nsimplify(n, rational=False).evalf(4))


def solve_equation(problem: str):
    lhs_text, rhs_text = problem.split("=")
    lhs, rhs = parse_side(lhs_text), parse_side(rhs_text)
    expr = sp.expand(lhs - rhs)
    poly = sp.Poly(expr, X)
    degree = poly.degree()

    steps = [{
        "d": f"{sp.sstr(lhs)} = {sp.sstr(rhs)}",
        "s": f"Let's solve {sp.sstr(lhs)} equals {sp.sstr(rhs)}, for x.",
    }]

    if degree == 1:
        a, b = poly.all_coeffs()  # a*x + b = 0
        steps.append({"d": f"{fmt(a)}x = {fmt(-b)}",
                       "s": "Move the constant term to the other side."})
        root = sp.nsimplify(-b / a)
        steps.append({"d": f"x = {fmt(-b)} / {fmt(a)}",
                       "s": f"Divide both sides by {fmt(a)}."})
        steps.append({"d": f"x = {fmt(root)}",
                       "s": f"So x equals {fmt(root)}."})
        return steps

    if degree == 2:
        a, b, c = poly.all_coeffs()
        steps.append({"d": f"{fmt(a)}x² + {fmt(b)}x + {fmt(c)} = 0",
                       "s": "Rewrite it in standard quadratic form."})
        disc = sp.simplify(b**2 - 4*a*c)
        steps.append({"d": f"Discriminant = {fmt(b)}² - 4({fmt(a)})({fmt(c)}) = {fmt(disc)}",
                       "s": f"The discriminant works out to {fmt(disc)}."})
        if disc < 0:
            steps.append({"d": "No real solutions",
                           "s": "Since the discriminant is negative, there are no real solutions."})
            return steps
        roots = sorted(sp.solve(sp.Eq(expr, 0), X), key=lambda r: sp.N(r))
        r1, r2 = roots[0], roots[1] if len(roots) > 1 else roots[0]
        steps.append({"d": f"x = (-{fmt(b)} ± √{fmt(disc)}) / (2·{fmt(a)})",
                       "s": "Apply the quadratic formula."})
        if r1 == r2:
            steps.append({"d": f"x = {fmt(r1)}", "s": f"So x equals {fmt(r1)}."})
        else:
            steps.append({"d": f"x = {fmt(r1)}  or  x = {fmt(r2)}",
                           "s": f"So x equals {fmt(r1)}, or x equals {fmt(r2)}."})
        return steps

    raise HTTPException(400, "This solver currently handles linear and quadratic equations only.")


def solve_fraction_add(problem: str):
    nums = re.findall(r"(-?\d+)\s*/\s*(-?\d+)", problem)
    if len(nums) != 2:
        raise HTTPException(400, "Couldn't read two fractions from that.")
    (a, b), (c, d) = [(int(x), int(y)) for x, y in nums]
    b, d = sp.Integer(b), sp.Integer(d)
    lcd = sp.lcm(b, d)
    a2, c2 = a * (lcd // b), c * (lcd // d)
    total = sp.Rational(a2, lcd) + sp.Rational(c2, lcd)
    steps = [
        {"d": f"{a}/{b} + {c}/{d} = ?", "s": f"Let's add {a} {b}ths and {c} {d}ths."},
        {"d": f"LCD of {b} and {d} is {lcd}", "s": f"The lowest common denominator of {b} and {d} is {lcd}."},
        {"d": f"{a}/{b} = {a2}/{lcd},  {c}/{d} = {c2}/{lcd}",
         "s": "Convert both fractions to that denominator."},
        {"d": f"{a2}/{lcd} + {c2}/{lcd} = {fmt(total)}", "s": f"Adding them gives {fmt(total)}."},
        {"d": f"Answer: {fmt(total)}", "s": f"So the answer is {fmt(total)}."},
    ]
    return steps


def solve_arithmetic(problem: str):
    expr = parse_side(problem)
    result = sp.nsimplify(expr)
    steps = [
        {"d": f"{problem.strip()} = ?", "s": f"Let's work out {problem.strip()}."},
        {"d": f"{problem.strip()} = {fmt(result)}", "s": f"That comes to {fmt(result)}."},
    ]
    return steps


@app.get("/")
def health():
    return {"status": "ok", "message": "SolveBoard solver is running."}


@app.post("/solve")
def solve(payload: ProblemIn):
    problem = payload.problem.strip()
    if not problem:
        raise HTTPException(400, "Please provide a problem.")

    try:
        if "=" in problem:
            steps = solve_equation(problem)
            topic = "Equation"
        elif re.search(r"\d\s*/\s*\d.*[+\-].*\d\s*/\s*\d", problem):
            steps = solve_fraction_add(problem)
            topic = "Fractions"
        else:
            steps = solve_arithmetic(problem)
            topic = "Arithmetic"
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "Couldn't understand that problem — try a simpler equation "
                                  "like '2x + 5 = 17', a fraction sum like '1/4 + 1/2', "
                                  "or basic arithmetic like '6 * 4'.")

    return {"topic": topic, "problem": problem, "steps": steps}
