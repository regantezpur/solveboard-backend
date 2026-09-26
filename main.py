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


FUNC_NAMES = "sin|cos|tan|sec|csc|cot|sqrt|log|ln"
# func^n(arg) or func^narg  ->  (func(arg))^n   e.g. tan^2(30), sin^2x
ARG = r"(?:\d+|x|pi|π)"  # the only bare arguments we accept — deliberately NOT a general word
# Two separate, unambiguous patterns rather than one with an optional paren on both ends —
# an optional-paren-plus-trailing-\b regex can match without consuming the closing paren,
# leaving it stranded in the output (a real bug this shipped with once: 'tan^2(30)' -> '(tan(30))^2)').
FUNC_POWER_PAREN_RE = re.compile(rf"(?<![a-zA-Z])({FUNC_NAMES})\^(\d+)\(({ARG})\)", re.IGNORECASE)
FUNC_POWER_BARE_RE = re.compile(rf"(?<![a-zA-Z])({FUNC_NAMES})\^(\d+)({ARG})\b(?!\()", re.IGNORECASE)
# bare func name with no parens at all, e.g. sinx, sinpi, cos30, 2sinx  ->  func(arg)
# (?<![a-zA-Z]) lets a digit precede it (2sinx) but not a letter (avoids matching inside "using")
# (?!\() skips names already followed by '(' — those are already fine
# The whitelist ARG (digits/x/pi only, not any letters) is what keeps this from mangling
# ordinary words like "cost" or "cosine" that happen to start with a function name.
BARE_FUNC_RE = re.compile(rf"(?<![a-zA-Z])({FUNC_NAMES})(?!\()({ARG})\b", re.IGNORECASE)


def normalize(text: str) -> str:
    """Turn symbols from the on-screen math toolbar into parseable text, and rescue
    function calls a student typed without parentheses (e.g. 'sinx', 'sinpi') before
    the parser can misread them as separate letters multiplied together."""
    for k, v in SYMBOL_MAP.items():
        text = text.replace(k, v)
    text = FUNC_POWER_PAREN_RE.sub(r"(\1(\3))^\2", text)
    text = FUNC_POWER_BARE_RE.sub(r"(\1(\3))^\2", text)
    text = BARE_FUNC_RE.sub(r"\1(\2)", text)
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


def solve_complex(problem: str):
    text = normalize(problem).replace("^", "**").lower()
    expr = parse_expr(text, transformations=TRANSFORMS, local_dict={"i": sp.I})

    steps = [{"d": f"{problem.strip()}", "s": f"Let's simplify {problem.strip()}, where i is the imaginary unit."}]

    def ipretty(e):
        return pretty(e).replace("I", "i")

    combined = sp.together(expr)
    num, den = sp.fraction(combined)
    num_exp, den_exp = sp.expand(num), sp.expand(den)
    if den != 1:
        steps.append({"d": f"Combine into a single fraction:  ({ipretty(num)}) / ({ipretty(den)})",
                       "s": "Combine everything into a single fraction, multiplying by conjugates where needed."})
        steps.append({"d": f"Numerator = {ipretty(num_exp)}   Denominator = {ipretty(den_exp)}",
                       "s": "Expand the numerator and denominator."})

    result = sp.simplify(expr)
    result = sp.nsimplify(result)
    re_part = sp.nsimplify(sp.re(result))
    im_part = sp.nsimplify(sp.im(result))
    steps.append({"d": "Using i² = -1, simplify",
                   "s": "Remember that i squared equals negative one, and simplify."})
    if im_part == 0:
        steps.append({"d": f"= {fmt(re_part)}", "s": f"That simplifies to {fmt(re_part)}."})
        steps.append({"d": f"∴  a = {fmt(re_part)}  and  b = 0",
                       "s": f"So in the form a plus b i, a is {fmt(re_part)} and b is 0."})
    else:
        im_str = f"{fmt(im_part)}i" if im_part != 1 else "i"
        if im_part == -1:
            im_str = "i"
        sign = "+" if im_part >= 0 else "-"
        steps.append({"d": f"= {fmt(re_part)} {sign} {fmt(abs(im_part))}i",
                       "s": f"That simplifies to {fmt(re_part)} {'plus' if im_part>=0 else 'minus'} {fmt(abs(im_part))} i."})
        steps.append({"d": f"∴  a = {fmt(re_part)}  and  b = {fmt(im_part)}",
                       "s": f"So a is {fmt(re_part)} and b is {fmt(im_part)}."})
    return steps


def parse_vector(text):
    """Extract (i, j, k) coefficients from free-form vector notation like 'i - 2j'."""
    comps = []
    for letter in ("i", "j", "k"):
        m = re.search(rf"([+-]?\s*\d*\.?\d*)\s*{letter}\b", text)
        if m and m.group(0).strip():
            coeff_str = m.group(1).replace(" ", "")
            if coeff_str in ("", "+"):
                coeff = sp.Integer(1)
            elif coeff_str == "-":
                coeff = sp.Integer(-1)
            else:
                coeff = sp.nsimplify(coeff_str)
            comps.append(coeff)
        else:
            comps.append(sp.Integer(0))
    return comps


def sqrt_str(n):
    """Exact display of a square root — '5' -> '√5', '9' -> '3' (kept exact, never decimalized)."""
    n = sp.nsimplify(n)
    r = sp.sqrt(n)
    if r.is_Integer or r.is_Rational:
        return fmt(r)
    return f"√{fmt(n)}"


def sq_term(c):
    """'(-2)²' for negatives, '3²' for positives — avoids the ugly '-2²'."""
    return f"({fmt(c)})²" if c < 0 else f"{fmt(c)}²"


def vector_str(ci, cj, ck, denom_display=None):
    parts = []
    labels = ("î", "ĵ", "k̂")
    for c, lab in zip((ci, cj, ck), labels):
        if c == 0:
            continue
        parts.append(term_str(c, lab, first=(len(parts) == 0)))
    body = " ".join(parts) if parts else "0"
    if denom_display is not None:
        return f"({body}) / {denom_display}"
    return body


def solve_vector(problem: str):
    low = problem.lower()
    vec_text = problem[low.rfind(" of ") + 4:] if " of " in low else problem
    ci, cj, ck = parse_vector(vec_text)

    steps = [{"d": f"a = {vector_str(ci, cj, ck)}", "s": f"We're given the vector a equals {vector_str(ci, cj, ck)}."}]

    mag_sq = ci**2 + cj**2 + ck**2
    mag_display = sqrt_str(mag_sq)
    sq_terms = " + ".join(sq_term(c) for c in (ci, cj, ck) if c != 0)
    sqrt_disp = f"√{fmt(mag_sq)}"
    mid = f"= {sqrt_disp} = {mag_display}" if mag_display != sqrt_disp else f"= {mag_display}"
    steps.append({"d": f"|a| = √({sq_terms}) {mid}",
                   "s": f"The magnitude is the square root of the sum of the squares of the components, which is {mag_display}."})

    if "unit" in low or "direction" in low or "magnitude" in low:
        steps.append({"d": f"â = a / |a| = {vector_str(ci, cj, ck, denom_display=mag_display)}",
                       "s": "The unit vector in that direction is a divided by its magnitude."})

    m = re.search(r"magnitude\s+of\s+(\d+\.?\d*)|magnitude\s+(\d+\.?\d*)", low)
    if m and "direction" in low:
        target = sp.nsimplify(m.group(1) or m.group(2))
        steps.append({"d": f"Vector of magnitude {fmt(target)} = {fmt(target)} × â = {vector_str(target*ci, target*cj, target*ck, denom_display=mag_display)}",
                       "s": f"A vector of magnitude {fmt(target)} in that direction is {fmt(target)} times the unit vector."})
        mag_is_exact_int = sp.sqrt(mag_sq).is_Integer
        pieces = []
        for c, lab in zip((ci, cj, ck), ("î", "ĵ", "k̂")):
            if c == 0:
                continue
            coeff = target * c
            if mag_is_exact_int:
                coeff = sp.nsimplify(coeff / sp.sqrt(mag_sq))
                mag_frag = fmt(coeff)
            else:
                mag_frag = f"{fmt(abs(coeff))}/{mag_display}"
                mag_frag = f"-{mag_frag}" if coeff < 0 else mag_frag
            pieces.append((coeff, f"{mag_frag} {lab}"))
        disp_terms = []
        for i, (coeff, frag) in enumerate(pieces):
            if i == 0:
                disp_terms.append(frag)
            else:
                disp_terms.append(f"+ {frag}" if coeff >= 0 else f"- {frag.lstrip('-')}")
        steps.append({"d": f"= {' '.join(disp_terms)}", "s": "That's the final vector, in exact form."})
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
        if re.search(r"(?<![a-zA-Z])i(?![a-zA-Z])", problem) and re.search(r"\d", problem) and \
                ("/" in problem or "+" in problem or "-" in problem) and "sin" not in low and "cos" not in low:
            steps = solve_complex(problem)
            topic = "Complex Numbers"
        elif re.search(r"(?<![a-zA-Z])[ijk](?![a-zA-Z])", problem) and \
                any(w in low for w in ["vector", "magnitude", "direction", "unit"]):
            steps = solve_vector(problem)
            topic = "Vectors"
        elif "%" in problem:
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
