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


SYMBOL_MAP = {
    "√": "sqrt", "π": "pi", "×": "*", "÷": "/", "−": "-", "·": "*",
    "≤": "<=", "≥": ">=", "≠": "!=",
    "⁰": "^0", "¹": "^1", "²": "^2", "³": "^3", "⁴": "^4",
    "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9",
}


TRIG_POWER_RE = re.compile(r"\b(sin|cos|tan|sec|csc|cot)\^(\d+)\(([^()]*)\)", re.IGNORECASE)


def normalize(text: str) -> str:
    """Turn symbols from the on-screen math toolbar into parseable text."""
    for k, v in SYMBOL_MAP.items():
        text = text.replace(k, v)
    # tan^2(30) -> (tan(30))^2, so the power applies to the whole function result
    text = TRIG_POWER_RE.sub(r"(\1(\3))^\2", text)
    return text


def strip_outer_parens(s: str) -> str:
    """Remove a genuinely wrapping pair of parens, e.g. '(x+1)' -> 'x+1',
    WITHOUT touching a function call's own parens like 'sin(x)'."""
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        depth = 0
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0 and i < len(s) - 1:
                return s  # closes before the very end -> not a full outer wrap
        return s[1:-1]
    return s


def apply_degrees(expr):
    """K-12 trig problems give angles in degrees (e.g. tan(30)), but SymPy's
    sin/cos/tan assume radians. Convert any trig call whose argument is a plain
    number (not a symbolic variable like x) so the numbers come out correct."""
    for fn in (sp.sin, sp.cos, sp.tan, sp.sec, sp.csc, sp.cot):
        expr = expr.replace(
            lambda e, fn=fn: e.func == fn and e.args[0].is_number,
            lambda e, fn=fn: fn(e.args[0] * sp.pi / 180),
        )
    return expr


def parse_side(text: str, degrees: bool = True):
    text = normalize(text).replace("^", "**")
    text = text.lower()  # unify variable case (x vs X) and function names
    expr = parse_expr(text, transformations=TRANSFORMS)
    return apply_degrees(expr) if degrees else expr


def pretty(expr):
    """Turn a sympy expression into display text without stray *'s, e.g. '54*x' -> '54x'."""
    s = sp.sstr(expr)
    s = re.sub(r"(\d)\*([a-zA-Z])", r"\1\2", s)   # 54*x -> 54x
    s = re.sub(r"\b1([a-zA-Z])\b", r"\1", s)      # 1x -> x
    s = s.replace("**2", "²").replace("**3", "³").replace("**", "^")
    return s


def fmt(n):
    """Pretty-print a sympy number: whole numbers without .0, fractions as a/b."""
    n = sp.nsimplify(n)
    if n.is_Integer:
        return str(n)
    if n.is_Rational:
        return f"{n.p}/{n.q}"
    return str(sp.nsimplify(n, rational=False).evalf(4))


def term_str(coeff, var="", first=False):
    """Format one polynomial term with correct sign and no '1x'/'​-1x' clutter."""
    coeff = sp.nsimplify(coeff)
    sign = "" if (first and coeff >= 0) else ("+ " if coeff >= 0 else "- ")
    mag = abs(coeff)
    if var and mag == 1:
        body = var
    else:
        body = f"{fmt(mag)}{var}"
    return f"{sign}{body}"


def poly_str(a, b, c):
    return f"{term_str(a, 'x²', first=True)} {term_str(b, 'x')} {term_str(c)}"


def _find_split_pair(ac, b):
    """For factoring by splitting the middle term: find integers p,q with p*q=ac, p+q=b."""
    ac = int(ac)
    if ac == 0:
        return None
    for i in range(1, abs(ac) + 1):
        if ac % i:
            continue
        j = ac // i
        for p, q in ((i, j), (-i, -j), (i, -j), (-i, j)):
            if p + q == b:
                return p, q
    return None


def solve_linear(given_lhs, given_rhs, lhs, rhs, a, b):
    """Standard sequence: given -> transpose -> isolate -> solve -> verify."""
    steps = [{"d": f"Given:  {given_lhs} = {given_rhs}",
               "s": f"We're given the equation {pretty(lhs)} equals {pretty(rhs)}. Let's solve for x."}]

    steps.append({"d": f"Transposing the constant term:  {fmt(a)}x = {fmt(-b)}",
                   "s": "Move the constant term to the other side of the equation, "
                        "changing its sign — this is called transposing."})

    root = sp.nsimplify(-b / a)
    if a != 1:
        steps.append({"d": f"Dividing both sides by {fmt(a)}:  x = {fmt(-b)} / {fmt(a)}",
                       "s": f"Divide both sides by {fmt(a)} to isolate x."})

    steps.append({"d": f"x = {fmt(root)}", "s": f"So x equals {fmt(root)}."})

    lhs_check = sp.nsimplify(lhs.subs(X, root))
    rhs_check = sp.nsimplify(rhs.subs(X, root))
    steps.append({"d": f"Check:  substitute x = {fmt(root)} back in",
                   "s": "Let's verify the answer by substituting it back into the original equation."})
    steps.append({"d": f"{pretty(lhs)} → {fmt(lhs_check)}   and   {pretty(rhs)} → {fmt(rhs_check)}  ✓",
                   "s": "Both sides come out equal, so the solution checks out."})
    return steps


def solve_quadratic(given_lhs, given_rhs, lhs, rhs, a, b, c, expr):
    steps = [{"d": f"Given:  {given_lhs} = {given_rhs}",
               "s": f"We're given {pretty(lhs)} equals {pretty(rhs)}. Let's solve for x."}]
    steps.append({"d": f"Standard form:  {poly_str(a, b, c)} = 0",
                   "s": "Rewrite it in standard quadratic form, a x squared plus b x plus c equals zero."})
    steps.append({"d": f"a = {fmt(a)},  b = {fmt(b)},  c = {fmt(c)}",
                   "s": f"Comparing with the standard form: a is {fmt(a)}, b is {fmt(b)}, c is {fmt(c)}."})

    disc = sp.simplify(b**2 - 4*a*c)
    is_int_coeffs = all(v.is_Integer for v in (a, b, c))
    pair = _find_split_pair(a * c, b) if is_int_coeffs else None

    if pair:
        p, q = pair
        steps.append({"d": "Method: factoring by splitting the middle term",
                       "s": "Since this factors neatly, let's use the splitting-the-middle-term method."})
        steps.append({"d": f"Find two numbers with product a×c = {fmt(a*c)} and sum b = {fmt(b)}",
                       "s": f"We need two numbers whose product is {fmt(a*c)} and whose sum is {fmt(b)}."})
        steps.append({"d": f"Those numbers are {fmt(p)} and {fmt(q)}",
                       "s": f"Those numbers are {fmt(p)} and {fmt(q)}."})
        split_line = f"{term_str(a, 'x²', first=True)} {term_str(p, 'x')} {term_str(q, 'x')} {term_str(c)}"
        steps.append({"d": f"Split the middle term:  {split_line} = 0",
                       "s": "Split the middle term into these two parts."})
        factored = sp.factor(expr)
        steps.append({"d": f"Factor by grouping:  {pretty(factored)} = 0",
                       "s": "Group the terms in pairs and factor each pair, then factor out the common bracket."})
        roots = sorted(sp.solve(sp.Eq(expr, 0), X), key=lambda r: sp.N(r))
        r1 = roots[0]
        r2 = roots[1] if len(roots) > 1 else roots[0]
        steps.append({"d": "Set each factor to zero", "s": "Each factor, set equal to zero, gives a solution."})
    else:
        steps.append({"d": "Method: the quadratic formula",
                       "s": "This doesn't factor neatly with whole numbers, so let's use the quadratic formula."})
        steps.append({"d": f"Discriminant D = b² - 4ac = {fmt(b)}² - 4({fmt(a)})({fmt(c)}) = {fmt(disc)}",
                       "s": f"The discriminant works out to {fmt(disc)}."})
        if disc < 0:
            steps.append({"d": "D < 0  →  no real solutions",
                           "s": "Since the discriminant is negative, there are no real solutions."})
            return steps
        steps.append({"d": f"x = (-b ± √D) / (2a) = (-{fmt(b)} ± √{fmt(disc)}) / (2×{fmt(a)})",
                       "s": "Substitute the values into the quadratic formula."})
        roots = sorted(sp.solve(sp.Eq(expr, 0), X), key=lambda r: sp.N(r))
        r1 = roots[0]
        r2 = roots[1] if len(roots) > 1 else roots[0]

    if r1 == r2:
        steps.append({"d": f"x = {fmt(r1)}  (repeated root)", "s": f"So x equals {fmt(r1)}, a repeated root."})
    else:
        steps.append({"d": f"x = {fmt(r1)}   or   x = {fmt(r2)}",
                       "s": f"So x equals {fmt(r1)}, or x equals {fmt(r2)}."})
    return steps


def solve_equation(problem: str):
    lhs_text, rhs_text = problem.split("=")
    def display_form(t):
        t = normalize(t).strip()
        return t.replace("^2", "²").replace("^3", "³")

    given_lhs = display_form(lhs_text)
    given_rhs = display_form(rhs_text)
    lhs, rhs = parse_side(lhs_text), parse_side(rhs_text)
    expr = sp.expand(lhs - rhs)
    poly = sp.Poly(expr, X)
    degree = poly.degree()

    if degree == 1:
        a, b = poly.all_coeffs()
        return solve_linear(given_lhs, given_rhs, lhs, rhs, a, b)
    if degree == 2:
        a, b, c = poly.all_coeffs()
        return solve_quadratic(given_lhs, given_rhs, lhs, rhs, a, b, c, expr)

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


def _infix_display(node, arg_vals):
    """Render one node's operation (pre-evaluation) for display, e.g. '2 × 3', '6 + 32'."""
    if node.func == sp.Add:
        parts = [term_str(v, first=(i == 0)) for i, v in enumerate(arg_vals)]
        return " ".join(parts)
    if node.func == sp.Mul:
        return " × ".join(fmt(v) for v in arg_vals)
    if node.func == sp.Pow:
        if arg_vals[1] == sp.Rational(1, 2):
            return f"√{fmt(arg_vals[0])}"
        return f"{fmt(arg_vals[0])}^{fmt(arg_vals[1])}"
    if node.func in (sp.sin, sp.cos, sp.tan, sp.sec, sp.csc, sp.cot):
        # arg_vals[0] is already in radians (converted); show the original degree number
        deg = sp.nsimplify(arg_vals[0] * 180 / sp.pi)
        return f"{node.func.__name__}({fmt(deg)}°)"
    if node.func == sp.log:
        return f"log({fmt(arg_vals[0])})"
    return pretty(node.func(*arg_vals))


def _arithmetic_steps(expr):
    """Walk the expression bottom-up (post-order), emitting one step per
    operation as it becomes fully numeric — this naturally follows BODMAS/PEMDAS
    order since inner sub-expressions are always resolved before outer ones."""
    steps = []

    def rec(node):
        if node.is_Atom:
            return node
        arg_vals = [rec(a) for a in node.args]
        if all(v.is_number for v in arg_vals):
            disp = _infix_display(node, arg_vals)
            evaluated = node.func(*arg_vals)
            val = sp.nsimplify(evaluated)
            if not (val.is_Integer or val.is_Rational):
                val = sp.nsimplify(evaluated.evalf(6), rational=False)
            steps.append({"d": f"{disp} = {fmt(val)}", "s": f"{disp} equals {fmt(val)}."})
            return val
        return node.func(*arg_vals)

    result = rec(expr)
    return steps, result


def solve_arithmetic(problem: str):
    raw = problem.strip()
    expr = parse_side(raw, degrees=True)
    # Parse again, unevaluated, so we can walk the original structure step by step —
    # SymPy auto-simplifies plain numeric expressions the instant they're built otherwise.
    unevaluated = parse_expr(
        normalize(raw).replace("^", "**").lower(), transformations=TRANSFORMS, evaluate=False
    )
    unevaluated = apply_degrees(unevaluated)

    steps = [{"d": f"{raw} = ?", "s": f"Let's work this out, following the order of operations."}]
    sub_steps, result = _arithmetic_steps(unevaluated)
    steps.extend(sub_steps)
    if len(sub_steps) != 1:
        steps.append({"d": f"{raw} = {fmt(result)}", "s": f"So altogether, that's {fmt(result)}."})
    return steps


def solve_percentage(problem: str):
    low = problem.lower()

    m = re.search(r"increase\s+(-?\d+\.?\d*)\s+by\s+(-?\d+\.?\d*)\s*%", low)
    if m:
        base, pct = sp.nsimplify(m.group(1)), sp.nsimplify(m.group(2))
        change = base * pct / 100
        result = base + change
        return [
            {"d": f"Increase {fmt(base)} by {fmt(pct)}%", "s": f"Let's increase {fmt(base)} by {fmt(pct)} percent."},
            {"d": f"{fmt(pct)}% of {fmt(base)} = {fmt(base)} × {fmt(pct)}/100 = {fmt(change)}",
             "s": f"First find {fmt(pct)} percent of {fmt(base)}, which is {fmt(change)}."},
            {"d": f"{fmt(base)} + {fmt(change)} = {fmt(result)}",
             "s": f"Add that to the original amount: {fmt(result)}."},
            {"d": f"Answer: {fmt(result)}", "s": f"So the increased value is {fmt(result)}."},
        ]

    m = re.search(r"decrease\s+(-?\d+\.?\d*)\s+by\s+(-?\d+\.?\d*)\s*%", low)
    if m:
        base, pct = sp.nsimplify(m.group(1)), sp.nsimplify(m.group(2))
        change = base * pct / 100
        result = base - change
        return [
            {"d": f"Decrease {fmt(base)} by {fmt(pct)}%", "s": f"Let's decrease {fmt(base)} by {fmt(pct)} percent."},
            {"d": f"{fmt(pct)}% of {fmt(base)} = {fmt(base)} × {fmt(pct)}/100 = {fmt(change)}",
             "s": f"First find {fmt(pct)} percent of {fmt(base)}, which is {fmt(change)}."},
            {"d": f"{fmt(base)} - {fmt(change)} = {fmt(result)}",
             "s": f"Subtract that from the original amount: {fmt(result)}."},
            {"d": f"Answer: {fmt(result)}", "s": f"So the decreased value is {fmt(result)}."},
        ]

    m = re.search(r"(-?\d+\.?\d*)\s*%\s*of\s*(-?\d+\.?\d*)", low)
    if m:
        pct, base = sp.nsimplify(m.group(1)), sp.nsimplify(m.group(2))
        result = base * pct / 100
        return [
            {"d": f"{fmt(pct)}% of {fmt(base)} = ?", "s": f"Let's find {fmt(pct)} percent of {fmt(base)}."},
            {"d": f"{fmt(pct)}% = {fmt(pct)}/100", "s": f"{fmt(pct)} percent means {fmt(pct)} over 100."},
            {"d": f"{fmt(pct)}/100 × {fmt(base)} = {fmt(result)}",
             "s": f"Multiply that by {fmt(base)} to get {fmt(result)}."},
            {"d": f"Answer: {fmt(result)}", "s": f"So the answer is {fmt(result)}."},
        ]

    raise HTTPException(400, "Try a percentage problem like '20% of 150', "
                              "'increase 150 by 20%', or 'decrease 150 by 20%'.")


def num_after(keyword, text):
    m = re.search(keyword + r"[^\d]{0,15}?(-?\d+\.?\d*)", text)
    return sp.nsimplify(m.group(1)) if m else None


def solve_geometry(problem: str):
    low = problem.lower()
    want_perimeter = "perimeter" in low or "circumference" in low
    want_area = "area" in low or not want_perimeter

    if "circle" in low:
        r = num_after("radius", low)
        if r is None:
            raise HTTPException(400, "Give the radius, e.g. 'area of a circle with radius 7'.")
        if want_perimeter:
            result = 2 * sp.pi * r
            return [
                {"d": f"Circumference = 2πr, r = {fmt(r)}", "s": f"The circumference formula is 2 pi r, with radius {fmt(r)}."},
                {"d": f"= 2 × π × {fmt(r)} = {fmt(result)}", "s": f"That works out to about {fmt(result)}."},
            ]
        result = sp.pi * r**2
        return [
            {"d": f"Area = πr², r = {fmt(r)}", "s": f"The area formula for a circle is pi r squared, with radius {fmt(r)}."},
            {"d": f"= π × {fmt(r)}² = {fmt(result)}", "s": f"That comes to about {fmt(result)}."},
        ]

    if "rectangle" in low:
        l = num_after("length", low)
        w = num_after("width", low) or num_after("breadth", low)
        if l is None or w is None:
            raise HTTPException(400, "Give both length and width, e.g. 'area of a rectangle with length 8 and width 5'.")
        if want_perimeter:
            result = 2 * (l + w)
            return [
                {"d": f"Perimeter = 2(l + w), l={fmt(l)}, w={fmt(w)}", "s": f"The perimeter formula is 2 times length plus width."},
                {"d": f"= 2({fmt(l)} + {fmt(w)}) = {fmt(result)}", "s": f"That gives {fmt(result)}."},
            ]
        result = l * w
        return [
            {"d": f"Area = l × w, l={fmt(l)}, w={fmt(w)}", "s": "The area formula is length times width."},
            {"d": f"= {fmt(l)} × {fmt(w)} = {fmt(result)}", "s": f"That gives {fmt(result)}."},
        ]

    if "square" in low:
        s = num_after("side", low)
        if s is None:
            raise HTTPException(400, "Give the side length, e.g. 'area of a square with side 6'.")
        if want_perimeter:
            result = 4 * s
            return [
                {"d": f"Perimeter = 4 × side = 4 × {fmt(s)}", "s": "The perimeter formula is four times the side."},
                {"d": f"= {fmt(result)}", "s": f"That gives {fmt(result)}."},
            ]
        result = s * s
        return [
            {"d": f"Area = side² = {fmt(s)}²", "s": "The area formula is the side squared."},
            {"d": f"= {fmt(result)}", "s": f"That gives {fmt(result)}."},
        ]

    if "triangle" in low:
        b = num_after("base", low)
        h = num_after("height", low)
        if b is None or h is None:
            raise HTTPException(400, "Give base and height, e.g. 'area of a triangle with base 10 and height 4'.")
        result = sp.Rational(1, 2) * b * h
        return [
            {"d": f"Area = ½ × base × height, base={fmt(b)}, height={fmt(h)}",
             "s": "The area formula is one half times base times height."},
            {"d": f"= ½ × {fmt(b)} × {fmt(h)} = {fmt(result)}", "s": f"That gives {fmt(result)}."},
        ]

    raise HTTPException(400, "This solver currently handles circle, rectangle, square, and triangle "
                              "area/perimeter problems.")


def extract_calc_expr(problem: str):
    text = re.sub(r"(?i)differentiate|derivative of|d/dx|integrate|with respect to x|\by\s*=", "", problem)
    text = strip_outer_parens(text.strip())
    # Calculus is done symbolically in x, in radians (the standard convention) —
    # degree-conversion is only for evaluating a specific numeric angle.
    return parse_side(text, degrees=False)


def solve_derivative(problem: str):
    expr = sp.expand(extract_calc_expr(problem))
    terms = sp.Add.make_args(expr)
    steps = [{"d": f"Differentiate: {pretty(expr)}",
               "s": f"Let's differentiate {pretty(expr)}, with respect to x."}]
    for term in terms:
        d = sp.diff(term, X)
        steps.append({"d": f"d/dx({pretty(term)}) = {pretty(d)}",
                       "s": f"The derivative of {pretty(term)} is {pretty(d)}."})
    total = sp.diff(expr, X)
    steps.append({"d": f"dy/dx = {pretty(total)}", "s": f"Adding those, dy by dx equals {pretty(total)}."})
    return steps


def solve_integral(problem: str):
    expr = sp.expand(extract_calc_expr(problem))
    terms = sp.Add.make_args(expr)
    steps = [{"d": f"Integrate: {pretty(expr)}",
               "s": f"Let's integrate {pretty(expr)}, with respect to x."}]
    for term in terms:
        i = sp.integrate(term, X)
        steps.append({"d": f"∫{pretty(term)} dx = {pretty(i)}",
                       "s": f"The integral of {pretty(term)} is {pretty(i)}."})
    total = sp.integrate(expr, X)
    steps.append({"d": f"= {pretty(total)} + C", "s": "Adding those together, and remembering the constant of integration, C."})
    return steps


@app.get("/")
def health():
    return {"status": "ok", "message": "SolveBoard solver is running."}


@app.post("/solve")
def solve(payload: ProblemIn):
    problem = payload.problem.strip()
    if not problem:
        raise HTTPException(400, "Please provide a problem.")

    low = problem.lower()
    try:
        if "%" in problem:
            steps = solve_percentage(problem)
            topic = "Percentage"
        elif any(w in low for w in ["area", "perimeter", "circumference"]) and \
                any(w in low for w in ["circle", "rectangle", "square", "triangle"]):
            steps = solve_geometry(problem)
            topic = "Geometry"
        elif re.search(r"(?i)differentiate|derivative|d/dx", problem):
            steps = solve_derivative(problem)
            topic = "Calculus (Differentiation)"
        elif re.search(r"(?i)integrate|∫", problem):
            steps = solve_integral(problem)
            topic = "Calculus (Integration)"
        elif "=" in problem:
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
        raise HTTPException(400, "Couldn't understand that problem — try things like "
                                  "'2x + 5 = 17', '1/4 + 1/2', '20% of 150', "
                                  "'area of a circle with radius 7', or 'differentiate x^2 + 3x'.")

    return {"topic": topic, "problem": problem, "steps": steps}
