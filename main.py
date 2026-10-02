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


def find_matching_brace(text, open_idx):
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


BACKSLASH = "\\"  # built once, named, to keep every reference to it unambiguous


def latex_to_plain(text: str) -> str:
    """Convert common LaTeX math commands into our plain-text syntax, so a problem
    pasted or OCR'd from a textbook/PDF (which often comes out as LaTeX) still
    parses. Not a full LaTeX engine — covers the constructs K-12 problems actually
    use: \\frac, \\sqrt, \\left/\\right, \\cdot, \\pi, trig names, and ^{...}."""
    b = BACKSLASH
    if b not in text:
        return text  # fast path — nothing to do for ordinary input

    changed = True
    while changed:
        changed = False
        idx = text.find(b + "frac{")
        if idx != -1:
            open1 = idx + 5  # position of the '{' itself (\frac + { = 5 chars in)
            close1 = find_matching_brace(text, open1)
            if close1 != -1 and close1 + 1 < len(text) and text[close1 + 1] == "{":
                open2 = close1 + 1
                close2 = find_matching_brace(text, open2)
                if close2 != -1:
                    num, den = text[open1 + 1:close1], text[open2 + 1:close2]
                    text = text[:idx] + f"(({num})/({den}))" + text[close2 + 1:]
                    changed = True
                    continue
        idx2 = text.find(b + "sqrt{")
        if idx2 != -1:
            open3 = idx2 + 5
            close3 = find_matching_brace(text, open3)
            if close3 != -1:
                inner = text[open3 + 1:close3]
                text = text[:idx2] + f"sqrt({inner})" + text[close3 + 1:]
                changed = True
                continue

    text = text.replace(b + "left(", "(").replace(b + "right)", ")")
    text = text.replace(b + "left|", "|").replace(b + "right|", "|")
    text = text.replace(b + "left[", "[").replace(b + "right]", "]")
    text = text.replace(b + "cdot", "*").replace(b + "times", "*")
    text = text.replace(b + "pi", "pi")
    for fn in ("sin", "cos", "tan", "sec", "csc", "cot", "log", "ln"):
        text = text.replace(b + fn, fn)
    text = text.replace(b + ",", " ").replace(b + ";", " ").replace(b + "!", " ")
    text = text.replace(b + "quad", " ").replace(b + "qquad", " ")

    # ^{n} -> ^(n)
    out, i = [], 0
    while i < len(text):
        if text[i] == "^" and i + 1 < len(text) and text[i + 1] == "{":
            close = find_matching_brace(text, i + 1)
            if close != -1:
                out.append(f"^({text[i+2:close]})")
                i = close + 1
                continue
        out.append(text[i])
        i += 1
    text = "".join(out)

    text = text.replace("{", "(").replace("}", ")")

    # strip any remaining backslash-commands we didn't explicitly handle
    out, i = [], 0
    while i < len(text):
        if text[i] == b:
            i += 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def normalize(text: str) -> str:
    """Turn symbols from the on-screen math toolbar into parseable text, rescue
    function calls a student typed without parentheses (e.g. 'sinx', 'sinpi') before
    the parser can misread them as separate letters multiplied together, and convert
    any LaTeX commands (pasted from a textbook/PDF) into our plain-text syntax."""
    text = latex_to_plain(text)
    for k, v in SYMBOL_MAP.items():
        text = text.replace(k, v)
    text = FUNC_POWER_PAREN_RE.sub(r"(\1(\3))^\2", text)
    text = FUNC_POWER_BARE_RE.sub(r"(\1(\3))^\2", text)
    text = BARE_FUNC_RE.sub(r"\1(\2)", text)
    # A '.' directly before a letter or '(' is multiplication (how a "·" often survives
    # copy-paste). Decimals (3.5) and sentence-ending periods never match this.
    text = re.sub(r"\.(?=[A-Za-z(])", "*", text)
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


def sqrtify(s: str) -> str:
    """Convert sqrt(...) to √(...), correctly handling nested sqrt(sqrt(...)) —
    a plain regex can't do this because it can't match balanced/nested parens."""
    out = []
    i = 0
    while i < len(s):
        if s[i:i + 5] == "sqrt(":
            depth = 1
            j = i + 5
            while j < len(s) and depth > 0:
                if s[j] == "(":
                    depth += 1
                elif s[j] == ")":
                    depth -= 1
                j += 1
            out.append(f"√({sqrtify(s[i + 5:j - 1])})")
            i = j
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def pretty(expr):
    """Turn a sympy expression into display text without stray *'s, e.g. '54*x' -> '54x'."""
    s = sp.sstr(expr)
    s = re.sub(r"(\d)\*([a-zA-Z])", r"\1\2", s)   # 54*x -> 54x
    s = re.sub(r"\b1([a-zA-Z])\b", r"\1", s)      # 1x -> x
    s = s.replace("**2", "²").replace("**3", "³").replace("**", "^")
    s = s.replace("*", "·")   # any remaining multiplication reads as a dot, not a stray asterisk
    s = sqrtify(s)
    return s


def fmt(n):
    """Pretty-print a sympy number: whole numbers without .0, fractions as a/b,
    and expressions involving e (like 1/e or 2e) kept exact rather than decimalized —
    same philosophy as keeping √5 instead of 2.236 everywhere else in this app."""
    n = sp.nsimplify(n)
    if n.is_Integer:
        return str(n)
    if n.is_Rational:
        return f"{n.p}/{n.q}"
    if n.func == sp.exp:
        # SymPy auto-canonicalizes E**k back into exp(k) internally — it won't stay
        # rewritten as a power — so format this Function-call form directly instead.
        k = n.args[0]
        if k == 1:
            return "e"
        if k == -1:
            return "1/e"
        if k.is_negative:
            return f"1/e^{fmt(-k)}"
        return f"e^{fmt(k)}"
    if n.has(sp.E):
        return pretty(n).replace("E", "e")
    return str(sp.nsimplify(n, rational=False).evalf(4))


def term_str(coeff, var="", first=False):
    """Format one polynomial term with correct sign and no '1x'/'​-1x' clutter."""
    coeff = sp.nsimplify(coeff)
    sign = "" if (first and coeff >= 0) else ("-" if first else ("+ " if coeff >= 0 else "- "))
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
    if expr.free_symbols:
        # Something in there isn't a number — usually ordinary words (which the parser
        # would happily read as letters multiplied together) or an unknown like x with no
        # equation. Presenting that as an "answer" would be confident nonsense.
        letters = ", ".join(sorted(str(s) for s in expr.free_symbols))
        raise HTTPException(
            400,
            f"I couldn't turn that into a calculation — it contains {letters}, which I can't "
            f"evaluate as numbers. If it's an equation, include an '=' (e.g. '2x + 5 = 17'); "
            f"if it's a word problem, I currently understand motion problems "
            f"(position → velocity/acceleration).",
        )
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
    # LaTeX first (\(...\), \sin, etc.), so phrase-stripping below sees clean text,
    # not backslashes.
    text = latex_to_plain(problem.replace("$", ""))
    # Generic question phrasing: "Find/What is/Calculate/Determine/Compute (the) ... of"
    text = re.sub(r"(?i)^\s*(find|what\s+is|calculate|determine|compute|evaluate)\s+(the\s+)?"
                  r"(derivative|integral|slope|gradient)?\s*(of)?\s*", "", text)
    text = re.sub(r"(?i)differentiate|derivative of|d/dx|integrate|with respect to x|\by\s*=", "", text)
    text = text.strip().rstrip(".?!").strip()
    # Strip a wrapping paren now (often left over from \( ... \) LaTeX delimiters turning
    # into literal parens around the WHOLE expression) — otherwise "(f(x) = ...)" doesn't
    # match the "f(x) =" prefix pattern below, since the string doesn't start with a letter.
    text = strip_outer_parens(text)

    if "=" in text:
        left, _, right = text.partition("=")
        # "f(x) = x^2 + 1" or "g = ..." -> keep only the right-hand side
        if re.fullmatch(r"\s*[a-zA-Z]\s*(\(\s*x\s*\))?\s*", left) and "=" not in right:
            text = right.strip()
        else:
            # A stray '=' in the middle of an expression is almost always a typo: '=' and '+'
            # share a key on most keyboards, so "x=2" is usually "x+2" without Shift.
            # Don't silently "fix" it — that could answer a different question than the
            # student asked. Say what we noticed instead.
            eq = text.index("=")
            lo, hi = text.rfind("(", 0, eq), text.find(")", eq)
            ctx = text[lo:hi + 1] if lo != -1 and hi != -1 else text[max(0, eq - 4): eq + 5]
            as_plus, as_minus = ctx.replace("=", "+"), ctx.replace("=", "-")
            raise HTTPException(
                400,
                f"I found an '=' inside the expression, in \"{ctx}\". Did you mean "
                f"\"{as_plus}\" or \"{as_minus}\"? (On many keyboards '=' and '+' share a key.) "
                f"Please correct it and try again.",
            )
    text = strip_outer_parens(text)
    # Calculus is done symbolically in x, in radians (the standard convention) —
    # degree-conversion is only for evaluating a specific numeric angle.
    return parse_side(text, degrees=False)


def paren(e):
    """Wrap a sum in parentheses when it's used as a factor, so 'x + 2' times sin(x)
    is written (x + 2)·sin(x) — never x + 2·sin(x), which means something else."""
    return f"({pretty(e)})" if isinstance(e, sp.Add) else pretty(e)


TRIG_DERIVS = {
    sp.sin: lambda u: sp.cos(u),
    sp.cos: lambda u: -sp.sin(u),
    sp.tan: lambda u: sp.sec(u) ** 2,
}


def diff_with_steps(expr, steps, top=True):
    """Differentiate expr wrt X, appending a human-readable chain/product-rule step
    for every composite (non-trivial) operation encountered. Simple monomials like
    x**2 or 3*x are resolved silently, exactly as before, to avoid noisy over-explaining."""
    if not expr.has(X):
        return sp.Integer(0)
    if expr == X:
        return sp.Integer(1)

    if isinstance(expr, sp.Add):
        return sp.Add(*[diff_with_steps(a, steps, top=False) for a in expr.args])

    if isinstance(expr, sp.Mul):
        const_factors = [a for a in expr.args if not a.has(X)]
        var_parts = [a for a in expr.args if a.has(X)]
        const_part = sp.Mul(*const_factors) if const_factors else sp.Integer(1)

        # A genuine quotient (denominator involves x in a non-monomial way, e.g. x+3):
        # use the quotient rule, the way it's taught, instead of treating 1/(x+3) as a
        # power-rule factor inside a product.
        num, den = sp.fraction(sp.Mul(*var_parts)) if var_parts else (sp.Integer(1), sp.Integer(1))
        den_is_monomial = den == X or (isinstance(den, sp.Pow) and den.args[0] == X)
        if den.has(X) and not den_is_monomial:
            steps.append({"d": f"Quotient rule:  d/dx(N/D) = (D·d/dx(N) - N·d/dx(D)) / D²,   "
                                f"where N = {pretty(num)}  and  D = {pretty(den)}",
                           "s": "Since this is a fraction with x in the denominator, use the quotient rule: "
                                "the denominator times the derivative of the numerator, minus the numerator "
                                "times the derivative of the denominator, all over the denominator squared."})
            dn = diff_with_steps(num, steps, top=False)
            if num.has(X) and num != X:
                steps.append({"d": f"d/dx({pretty(num)}) = {pretty(dn)}",
                               "s": f"The derivative of the numerator is {pretty(dn)}."})
            dd = diff_with_steps(den, steps, top=False)
            if den != X:
                steps.append({"d": f"d/dx({pretty(den)}) = {pretty(dd)}",
                               "s": f"The derivative of the denominator is {pretty(dd)}."})
            steps.append({"d": f"= (({pretty(den)})({pretty(dn)}) - ({pretty(num)})({pretty(dd)})) / ({pretty(den)})²",
                           "s": "Substitute these into the quotient rule."})
            return const_part * sp.simplify((den * dn - num * dd) / den ** 2)

        if len(var_parts) <= 1:
            u = var_parts[0] if var_parts else sp.Integer(1)
            return const_part * diff_with_steps(u, steps, top=False)
        # product rule across all variable factors, applied pairwise
        u = var_parts[0]
        v = sp.Mul(*var_parts[1:])
        steps.append({"d": f"Product rule:  d/dx({paren(u)}·{paren(v)}) = {paren(u)}·d/dx({pretty(v)}) + {paren(v)}·d/dx({pretty(u)})",
                       "s": "Since this is a product of two expressions involving x, use the product rule."})
        du = diff_with_steps(u, steps, top=False)
        dv = diff_with_steps(v, steps, top=False)
        return const_part * sp.simplify(u * dv + v * du)

    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        if exp.has(X):
            return sp.diff(expr, X)  # variable exponent — outside current scope, fall back silently
        if base == X:
            return exp * X ** (exp - 1)
        if exp == sp.Rational(1, 2):
            steps.append({"d": f"d/dx(√({pretty(base)})) = 1/(2√({pretty(base)})) · d/dx({pretty(base)})",
                           "s": f"This is a square root of an expression involving x, so use the chain rule: "
                                f"one over twice the square root of the inside, times the derivative of the inside."})
        else:
            steps.append({"d": f"d/dx(({pretty(base)})^{fmt(exp)}) = {fmt(exp)}({pretty(base)})^{fmt(exp - 1)} · d/dx({pretty(base)})",
                           "s": f"Use the chain rule: bring down the exponent {fmt(exp)}, reduce the power by one, "
                                f"and multiply by the derivative of the inside."})
        inner_deriv = diff_with_steps(base, steps, top=False)
        if base != X:
            steps.append({"d": f"d/dx({pretty(base)}) = {pretty(inner_deriv)}",
                           "s": f"The derivative of the inside, {pretty(base)}, is {pretty(inner_deriv)}."})
        if exp == sp.Rational(1, 2):
            return sp.simplify(inner_deriv / (2 * sp.sqrt(base)))
        return sp.simplify(exp * base ** (exp - 1) * inner_deriv)

    if expr.func in TRIG_DERIVS:
        inner = expr.args[0]
        outer_deriv = TRIG_DERIVS[expr.func](inner)
        if inner == X:
            steps.append({"d": f"d/dx({expr.func.__name__}(x)) = {pretty(outer_deriv)}",
                           "s": f"The derivative of {expr.func.__name__} x is {pretty(outer_deriv)}."})
        else:
            steps.append({"d": f"d/dx({expr.func.__name__}({pretty(inner)})) = {pretty(outer_deriv)} · d/dx({pretty(inner)})",
                           "s": f"Use the chain rule for {expr.func.__name__}: differentiate the outside, "
                                f"then multiply by the derivative of the inside."})
        inner_deriv = diff_with_steps(inner, steps, top=False)
        if inner != X:
            steps.append({"d": f"d/dx({pretty(inner)}) = {pretty(inner_deriv)}",
                           "s": f"The derivative of the inside, {pretty(inner)}, is {pretty(inner_deriv)}."})
        return sp.simplify(outer_deriv * inner_deriv)

    return sp.diff(expr, X)  # fallback for anything not explicitly handled above


def solve_derivative(problem: str):
    parsed = extract_calc_expr(problem)
    is_simple = sp.expand(parsed).is_polynomial(X)
    # Only expand polynomials (to split them term by term). Expanding anything else
    # destroys the structure the student wrote and produces confusing, repeated steps.
    expr = sp.expand(parsed) if is_simple else parsed
    terms = sp.Add.make_args(expr) if is_simple else [expr]

    steps = [{"d": f"Differentiate: {pretty(expr)}",
               "s": f"Let's differentiate {pretty(expr)}, with respect to x."}]
    for term in terms:
        d = diff_with_steps(term, steps)
        if is_simple:
            steps.append({"d": f"d/dx({pretty(term)}) = {pretty(d)}",
                           "s": f"The derivative of {pretty(term)} is {pretty(d)}."})
    total = sp.simplify(sp.diff(expr, X))
    steps.append({"d": f"dy/dx = {pretty(total)}", "s": f"So dy by dx equals {pretty(total)}."})
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


I_NAMES = {0: "", 1: "i", 2: "i²", 3: "i³"}


def ci_str(expr, I, sub_i2: bool = False) -> str:
    """Write an expression in i the way a textbook does: ascending powers, no stray
    asterisks — '3 + 10i + 8i²', not '8*i**2 + 10*i + 3'. With sub_i2=True, i² is
    shown already replaced by (-1), e.g. '12 + 20(-1)', as the PDF's worked solutions do."""
    expr = sp.expand(expr)
    if not expr.has(I):
        return fmt(expr)
    coeffs = sp.Poly(expr, I).all_coeffs()[::-1]
    parts = []
    for power, c in enumerate(coeffs):
        if c == 0:
            continue
        var = "(-1)" if (sub_i2 and power == 2) else I_NAMES.get(power, f"i^{power}")
        parts.append(term_str(c, var, first=(not parts)))
    return " ".join(parts) if parts else "0"


def reduce_i(expr, I):
    """Reduce every power of i using i² = -1 (so i³ = -i, i⁴ = 1 as well) — done as a
    polynomial remainder modulo i²+1, which is exact for any power."""
    expr = sp.expand(expr)
    if not expr.has(I):
        return expr
    return sp.rem(sp.Poly(expr, I), sp.Poly(I ** 2 + 1, I)).as_expr()


def mono_str(term, I, first: bool) -> str:
    """One un-collected product term such as 10i² or -4i, as it appears mid-expansion."""
    k = int(sp.degree(term, I)) if term.has(I) else 0
    c = sp.nsimplify(sp.simplify(term / I ** k))
    return term_str(c, I_NAMES.get(k, f"i^{k}"), first)


def extract_math_run(text: str) -> str:
    """Pull the mathematical expression out of a sentence — the longest run made only of
    digits, i, operators, brackets and spaces — so 'Simplify (3+4i)/(1-2i) where i is the
    imaginary unit' works instead of failing on the words."""
    runs = re.findall(r"[0-9i+\-*/().^\s]+", text)
    runs = [r.strip() for r in runs if re.search(r"\d", r) and re.search(r"[+\-*/()]", r)]
    if not runs:
        raise HTTPException(400, "I couldn't find a calculation in that. Try something like "
                                  "'(3+2i)/(2-5i) + (3-2i)/(2+5i)'.")
    return max(runs, key=len)


def solve_complex(problem: str):
    if re.search(r"[α-ωΑ-Ω]", problem):
        raise HTTPException(400, "This has unknowns written as Greek letters (like λ and μ). Solving for "
                                  "unknowns inside complex equations isn't supported yet — I can simplify "
                                  "complex expressions with numbers, like '(3+7i)(2+i)'.")
    wants_modulus = any(w in problem.lower() for w in ("modulus", "magnitude", "absolute value"))
    math_text = extract_math_run(normalize(problem).lower())
    text = math_text.replace("^", "**")
    I = sp.Symbol("i")  # a PLAIN symbol, not sp.I — this is what lets us control exactly
    # when i**2 becomes -1, instead of the math library doing it automatically and
    # collapsing several teaching steps into one.
    expr = parse_expr(text, transformations=TRANSFORMS, local_dict={"i": I})
    stray = expr.free_symbols - {I}
    if stray:
        raise HTTPException(400, f"I found {', '.join(sorted(map(str, stray)))} in that, which I can't "
                                  f"treat as a number. (Solving for unknowns like λ and μ in complex "
                                  f"equations isn't supported yet.)")

    steps = [{"d": math_text, "s": "Let's simplify this, where i is the imaginary unit."}]

    add_args = sp.Add.make_args(expr)
    num_expr, den_expr = None, None

    if len(add_args) == 2 and all(sp.fraction(a)[1] != 1 for a in add_args):
        # Exactly two fractions being added — the textbook cross-multiplication method,
        # shown line by line as in a worked solution.
        (numA, denB), (numC, denD) = (sp.fraction(a) for a in add_args)
        steps.append({"d": f"Combine using conjugates:  (({ci_str(numA, I)})({ci_str(denD, I)}) + ({ci_str(denB, I)})({ci_str(numC, I)})) / (({ci_str(denB, I)})({ci_str(denD, I)}))",
                       "s": "Combine into a single fraction by cross-multiplying."})
        num_expr = sp.expand(numA * denD + denB * numC)
        den_expr = sp.expand(denB * denD)

        # distribute term by term, WITHOUT collecting like terms yet (6+15i+4i+10i²+...)
        raw_terms = []
        for A, B in ((numA, denD), (denB, numC)):
            for ta in sp.Add.make_args(sp.expand(A)):
                for tb in sp.Add.make_args(sp.expand(B)):
                    raw_terms.append(sp.expand(ta * tb))
        raw_num = " ".join(mono_str(t, I, first=(n == 0)) for n, t in enumerate(raw_terms))
        steps.append({"d": f"Expand each product:  ({raw_num}) / ({ci_str(den_expr, I)})",
                       "s": "Multiply out each pair of brackets, term by term."})
        if raw_num != ci_str(num_expr, I):
            steps.append({"d": f"Collect like terms:  ({ci_str(num_expr, I)}) / ({ci_str(den_expr, I)})",
                           "s": "Now combine the like terms."})
    else:
        combined = sp.together(expr)
        num_expr, den_expr = sp.fraction(combined)
        if den_expr.has(I):
            # Rationalize: multiply top and bottom by the denominator's conjugate.
            # (Substituting i -> -i gives the conjugate for anything linear in i.)
            den_conj = den_expr.subs(I, -I)
            steps.append({"d": f"Multiply numerator and denominator by the conjugate:  ({ci_str(den_conj, I)})/({ci_str(den_conj, I)})",
                           "s": "Multiply both the numerator and denominator by the denominator's conjugate, to clear i from the denominator."})
            num_expr = sp.expand(num_expr * den_conj)
            den_expr = sp.expand(den_expr * den_conj)
            steps.append({"d": f"= ({ci_str(num_expr, I)}) / ({ci_str(den_expr, I)})",
                           "s": "Expand the numerator and denominator."})
        else:
            num_expr, den_expr = sp.expand(num_expr), sp.expand(den_expr)
            if den_expr != 1:
                steps.append({"d": f"Combine into a single fraction:  ({ci_str(num_expr, I)}) / ({ci_str(den_expr, I)})",
                               "s": "Combine everything into a single fraction."})

    # Explicit "i² = -1" step, shown the way the PDF does: 12 + 20(-1)  /  4 - 25(-1)
    num_sub = reduce_i(num_expr, I)
    den_sub = reduce_i(den_expr, I) if den_expr != 1 else den_expr
    if num_sub != num_expr or den_sub != den_expr:
        if den_expr != 1:
            steps.append({"d": f"Using i² = -1:  ({ci_str(num_expr, I, sub_i2=True)}) / ({ci_str(den_expr, I, sub_i2=True)})",
                           "s": "Replace i squared with negative one."})
            if num_sub.has(I) or den_sub.has(I):
                steps.append({"d": f"Simplify:  ({ci_str(num_sub, I)}) / ({ci_str(den_sub, I)})",
                               "s": "Simplify the numerator and the denominator."})
        else:
            steps.append({"d": f"Using i² = -1:  {ci_str(num_expr, I, sub_i2=True)}",
                           "s": "Replace i squared with negative one."})

    # If the denominator STILL contains i (it only comes out real when the two denominators
    # happen to be conjugates, as in the PDF's example), rationalize it too — exactly what a
    # student would do next.
    if den_sub != 1 and den_sub.has(I):
        conj = den_sub.subs(I, -I)
        steps.append({"d": f"The denominator still has i — multiply top and bottom by its conjugate:  ({ci_str(conj, I)})/({ci_str(conj, I)})",
                       "s": "The denominator still contains i, so multiply the numerator and denominator by its conjugate."})
        n2, d2 = sp.expand(num_sub * conj), sp.expand(den_sub * conj)
        steps.append({"d": f"= ({ci_str(n2, I)}) / ({ci_str(d2, I)})", "s": "Expand the numerator and denominator."})
        n3, d3 = reduce_i(n2, I), reduce_i(d2, I)
        steps.append({"d": f"Using i² = -1:  ({ci_str(n2, I, sub_i2=True)}) / ({ci_str(d2, I, sub_i2=True)})",
                       "s": "Replace i squared with negative one."})
        steps.append({"d": f"Simplify:  ({ci_str(n3, I)}) / ({ci_str(d3, I)})", "s": "Simplify."})
        num_sub, den_sub = n3, d3
    if den_sub.has(I):
        raise HTTPException(400, "I couldn't clear i from the denominator for that expression.")

    result = sp.expand(num_sub / den_sub) if den_sub != 1 else num_sub
    poly = sp.Poly(result, I) if result.has(I) else None
    if poly and poly.degree() >= 1:
        coeffs = poly.all_coeffs()
        im_part = sp.nsimplify(coeffs[0]) if poly.degree() == 1 else 0
        re_part = sp.nsimplify(coeffs[-1])
    else:
        re_part, im_part = sp.nsimplify(result), sp.Integer(0)

    if im_part == 0:
        steps.append({"d": f"= {fmt(re_part)} + 0i", "s": f"That simplifies to {fmt(re_part)}."})
        steps.append({"d": f"∴  a = {fmt(re_part)}  and  b = 0",
                       "s": f"So in the form a plus b i, a is {fmt(re_part)} and b is 0."})
    else:
        steps.append({"d": f"= {ci_str(re_part + im_part * I, I)}",
                       "s": f"That simplifies to {fmt(re_part)} {'plus' if im_part>=0 else 'minus'} {fmt(abs(im_part))} i."})
        steps.append({"d": f"∴  a = {fmt(re_part)}  and  b = {fmt(im_part)}",
                       "s": f"So a is {fmt(re_part)} and b is {fmt(im_part)}."})

    if wants_modulus:
        mod = sp.simplify(sp.sqrt(re_part ** 2 + im_part ** 2))
        mod_str = fmt(mod) if mod.is_Rational else pretty(mod)
        steps.append({"d": f"|z| = √(a² + b²) = √(({fmt(re_part)})² + ({fmt(im_part)})²) = {mod_str}",
                       "s": f"The modulus is the square root of a squared plus b squared, which is {mod_str}."})
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


def is_vector_problem(problem: str) -> bool:
    """A vector (i, j, k) problem — not a complex number that merely contains an 'i'.
    Vectors use j or k alongside i; complex numbers use only i. The word "unit" alone is
    NOT enough ("i is the imaginary unit"), and neither is "magnitude" (complex modulus)
    unless a j or k is present."""
    low = problem.lower()
    has_jk = re.search(r"(?<![a-zA-Z])[jk](?![a-zA-Z])", problem)
    has_i = re.search(r"(?<![a-zA-Z])i(?![a-zA-Z])", problem)
    if has_jk and any(w in low for w in ("vector", "magnitude", "direction", "unit")):
        return True
    return bool("vector" in low and (has_i or has_jk))


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


EVAL_POINT_RE = re.compile(r"(?:at|when|for|if)\s*\(?\s*([a-zA-Z])\s*=\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
EVAL_SECONDS_RE = re.compile(r"(?:at|after|when)\s+(?:time\s+)?(-?\d+(?:\.\d+)?)\s*(?:s\b|sec)", re.IGNORECASE)
FUNC_WORDS = {"sin", "cos", "tan", "sec", "csc", "cot", "sqrt", "log", "ln", "exp", "pi", "abs"}


def swap_var(text: str, old: str, new: str) -> str:
    """Replace a single-letter variable only where it stands alone as a variable —
    never inside a longer name like sqrt, tan, or exp (a plain str.replace would
    turn sqrt(t) into sqrx(x))."""
    return re.sub(rf"(?<![A-Za-z]){re.escape(old)}(?![A-Za-z])", new, text)


def grab_expression(text: str, start: int) -> str:
    """Read a math expression starting at `start`, stopping where the math ends and the
    sentence carries on: at a sentence-ending period, a comma, an unmatched ')', or the
    first ordinary word (meters, where, with, find ...). This is what stops
    '... + 9t. Find its acceleration ...' from swallowing the next sentence."""
    depth, i, n = 0, start, len(text)
    while i < n:
        ch = text[i]
        if ch in ",;?!$":
            break
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                break
            depth -= 1
        elif ch == "." and not (i + 1 < n and text[i + 1].isdigit()):
            break
        elif ch.isalpha():
            j = i
            while j < n and text[j].isalpha():
                j += 1
            if j - i >= 2 and text[i:j].lower() not in FUNC_WORDS:
                break
            i = j
            continue
        i += 1
    return text[start:i].strip()


def find_motion_definition(text: str):
    """Locate 'x(t) = ...' or plain 'x = ...' and return (function name, variable, expression)."""
    m = re.search(r"(?<![A-Za-z])([A-Za-z])\(([A-Za-z])\)\s*=\s*", text)
    if m:
        expr = grab_expression(text, m.end())
        return (m.group(1), m.group(2), expr) if expr else None

    em = EVAL_POINT_RE.search(text)
    eval_var = em.group(1) if em else None
    for m in re.finditer(r"(?<![A-Za-z])([A-Za-z])\s*=\s*", text):
        name = m.group(1)
        if eval_var and name == eval_var:
            continue                      # that's "t = 2" (the point), not the position function
        expr = grab_expression(text, m.end())
        letters = {w for w in re.findall(r"[A-Za-z]+", expr) if w.lower() not in FUNC_WORDS}
        if not letters:
            continue                      # "t = 4": no variable in it, so not a function
        var = eval_var or (next(iter(letters)) if len(letters) == 1 else None)
        if var:
            return name, var, expr
    return None


def solve_motion(problem: str):
    """Word-problem handler for 'position function -> velocity / acceleration / speed'
    questions. The variable is often 't' (or another letter), never hardcoded 'x', so this
    swaps the problem's own variable for our internal x, computes, then swaps back for
    display. Returns None if this doesn't look like a motion problem, so the caller can
    fall through to something else."""
    text = problem.replace(BACKSLASH + "(", "(").replace(BACKSLASH + ")", ")").replace("$", "")
    low = text.lower()
    if not any(w in low for w in ("velocity", "acceleration", "speed")):
        return None
    found = find_motion_definition(text)
    if not found:
        return None
    func_name, var_name, expr_text = found

    is_accel = "acceleration" in low
    is_speed = "speed" in low and "velocity" not in low and not is_accel
    order = 2 if is_accel else 1
    quantity = "acceleration" if is_accel else ("speed" if is_speed else "velocity")
    has_units = "meter" in low and "second" in low
    unit = ("m/s" if order == 1 else "m/s²") if has_units else ""
    unit_suffix = f" {unit}" if unit else ""
    unit_words = (" meters per second" if unit == "m/s" else " meters per second squared") if unit else ""

    expr_x = parse_side(swap_var(expr_text, var_name, "x"), degrees=False)
    if expr_x.free_symbols - {X}:
        return None                       # something in there isn't the variable — not ours to guess
    vel_x = sp.expand(sp.diff(expr_x, X, 1))
    deriv_x = sp.expand(sp.diff(expr_x, X, order))

    def to_display(e):
        return swap_var(pretty(e), "x", var_name)

    steps = [{"d": f"Given: {func_name}({var_name}) = {expr_text.strip()}",
               "s": f"We're given the position function, {func_name} of {var_name}, equals {expr_text.strip()}."}]

    if order == 1:
        label = "Velocity" if not is_speed else "Velocity"
        steps.append({"d": f"{label} = d{func_name}/d{var_name} = {to_display(vel_x)}",
                       "s": f"Velocity is the derivative of position with respect to time, which is {to_display(vel_x)}."})
    else:
        steps.append({"d": f"Velocity = d{func_name}/d{var_name} = {to_display(vel_x)}",
                       "s": "First find velocity, the derivative of position."})
        steps.append({"d": f"Acceleration = d²{func_name}/d{var_name}² = {to_display(deriv_x)}",
                       "s": f"Acceleration is the derivative of velocity, which is {to_display(deriv_x)}."})

    em = EVAL_POINT_RE.search(text)
    sm = EVAL_SECONDS_RE.search(text)
    point_raw = em.group(2) if em else (sm.group(1) if sm else None)
    if point_raw is not None:
        point = sp.nsimplify(point_raw)
        value = sp.nsimplify(deriv_x.subs(X, point))
        substituted = swap_var(pretty(deriv_x), "x", f"({fmt(point)})")
        substituted = re.sub(r"\(\((-?[\d.]+)\)\)", r"(\1)", substituted)   # '√((4))' -> '√(4)'
        steps.append({"d": f"At {var_name} = {fmt(point)}:  {substituted}",
                       "s": f"Now substitute {var_name} equals {fmt(point)} into the "
                            f"{'velocity' if is_speed else quantity} expression."})
        if is_speed:
            steps.append({"d": f"Velocity = {fmt(value)}{unit_suffix}", "s": f"That gives a velocity of {fmt(value)}."})
            speed_val = abs(value)
            steps.append({"d": f"Speed = |velocity| = |{fmt(value)}| = {fmt(speed_val)}{unit_suffix}",
                           "s": f"Speed is the size of the velocity, ignoring direction, which is {fmt(speed_val)}{unit_words}."})
        else:
            steps.append({"d": f"= {fmt(value)}{unit_suffix}",
                           "s": f"So the {quantity} at {var_name} equals {fmt(point)} is {fmt(value)}{unit_words}."})
    return steps


DERIV_RE = re.compile(r"\(?\s*d\^?(\d)?\s*y\s*\)?\s*/\s*\(?\s*dx\^?(\d)?\s*\)?", re.IGNORECASE)
DERIV_NAMES = {1: "dy/dx", 2: "d²y/dx²", 3: "d³y/dx³", 4: "d⁴y/dx⁴"}


def try_recover_lost_exponent(text: str, end_of_capture: int, param: str):
    """'eat' (exponent notation lost in copy/paste) recovered as 'e^(at)' — but ONLY when
    the capture stopped right after a DANGLING operator (+ - * /). That's a narrow, strong
    signal something was cut off mid-expression, which is what tells 'eat' (in 't + eat')
    apart from 'event' appearing as an ordinary word elsewhere — real words essentially
    never immediately follow a bare trailing operator like that."""
    stripped = text[:end_of_capture].rstrip()
    if not stripped or stripped[-1] not in "+-*/":
        return None
    m = re.match(r"\s*([a-zA-Z]+)", text[end_of_capture:])
    if not m:
        return None
    word = m.group(1)
    if len(word) >= 2 and word[0].lower() == "e" and word[-1].lower() == param.lower():
        return word, end_of_capture + m.end()
    return None


def grab_expression_with_recovery(text: str, start: int, param: str):
    expr = grab_expression(text, start)
    end = start + len(text[start:start + len(expr)]) if expr else start
    # grab_expression doesn't return an end index, so recompute it by re-scanning —
    # cheap, and keeps that function's signature simple for its other (motion) caller.
    end = start
    depth = 0
    while end < len(text):
        ch = text[end]
        if ch in ",;?!$":
            break
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                break
            depth -= 1
        elif ch == "." and not (end + 1 < len(text) and text[end + 1].isdigit()):
            break
        elif ch.isalpha():
            j = end
            while j < len(text) and text[j].isalpha():
                j += 1
            word = text[end:j].lower()
            if word == "and" or (j - end >= 2 and word not in FUNC_WORDS):
                break
            end = j
            continue
        end += 1

    recovered = try_recover_lost_exponent(text, end, param)
    if not recovered:
        return expr, False
    word, new_end = recovered
    fixed_word = f"e^({word[1:]})"
    rest, changed_again = grab_expression_with_recovery(text, new_end, param)
    full = (expr + " " + fixed_word + (" " + rest if rest else "")).strip()
    return full, True


PARAM_DEF_RE = re.compile(r"([a-zA-Z])\s*(?:\([a-zA-Z]\))?\s*=\s*", re.IGNORECASE)

# Each entry: detection phrase -> (equations to build from X(t), Y(t), a solve target, and
# a describer for the final answer). Building this as a small table, rather than one-off
# code per problem, is what lets it grow to cover more conditions later without a rewrite.
def solve_parametric_curve(problem: str):
    text = problem.replace(BACKSLASH + "(", "(").replace(BACKSLASH + ")", "").replace("$", "")
    low = text.lower()
    if "parametric" not in low:
        return None

    mx = re.search(r"x\s*(?:\([a-zA-Z]\))?\s*=\s*", text, re.IGNORECASE)
    my = re.search(r"y\s*(?:\([a-zA-Z]\))?\s*=\s*", text, re.IGNORECASE)
    if not (mx and my):
        return None
    param = "t"  # the overwhelmingly standard convention; problems that use another
    # letter for the parameter but still call it "t" in prose are rare enough to treat
    # as a future refinement rather than guess wrong here.

    x_text, x_fixed = grab_expression_with_recovery(text, mx.end(), param)
    y_text, y_fixed = grab_expression_with_recovery(text, my.end(), param)
    if not x_text or not y_text:
        return None

    T = sp.Symbol(param, real=True)
    extra_letters = sorted(set(re.findall(r"[a-zA-Z]", x_text + y_text)) - {param, "e"})
    positive_params = {L for L in extra_letters if re.search(rf"\b{L}\s*>\s*0", text, re.IGNORECASE)}
    # 'e' MUST map to Euler's number, not become a generic symbol — otherwise
    # d/dt(e^(at)) uses the general power rule for an unknown base and picks up a
    # spurious log(e) term that never collapses to 1, corrupting the whole solve.
    local_dict = {param: T, "e": sp.E}
    for L in extra_letters:
        local_dict[L] = sp.Symbol(L, positive=(L in positive_params), real=True)

    try:
        X_t = parse_expr(swap_var(x_text, param, param).replace("^", "**"), local_dict=local_dict, transformations=TRANSFORMS)
        Y_t = parse_expr(swap_var(y_text, param, param).replace("^", "**"), local_dict=local_dict, transformations=TRANSFORMS)
    except Exception:
        return None

    dXdt, dYdt = sp.diff(X_t, T), sp.diff(Y_t, T)

    steps = [{"d": f"Given: x = {x_text}{',  ' + 'note: recovered a lost exponent (e·· → e^(··))' if (x_fixed or y_fixed) else ''}   y = {y_text}",
               "s": f"We're given the parametric curve, x equals {x_text}, and y equals {y_text}."}]
    if x_fixed or y_fixed:
        steps.append({"d": "(The pasted text appeared to be missing an exponent — read 'e··' as e^(··). "
                            "Please double-check this matches the original question.)",
                       "s": "I noticed what looked like a missing exponent and filled it in — please double check that's right."})

    conditions, unknowns, describe = None, [T] + [local_dict[L] for L in extra_letters], None

    if re.search(r"touch(?:es)?\s+(?:the\s+)?x[\s-]*axis|tangent\s+to\s+(?:the\s+)?x[\s-]*axis", low):
        conditions = [sp.Eq(Y_t, 0), sp.Eq(dYdt, 0)]
        steps.append({"d": "Touching the x-axis means y = 0 AND dy/dt = 0 at the same point "
                            "(y = 0 alone would only mean crossing it, not touching it)",
                       "s": "For the curve to touch the x-axis rather than just cross it, we need y equal to zero "
                            "and the slope, dy by dt, equal to zero, at the same point."})
        describe = ("x", X_t)
    elif re.search(r"touch(?:es)?\s+(?:the\s+)?y[\s-]*axis|tangent\s+to\s+(?:the\s+)?y[\s-]*axis", low):
        conditions = [sp.Eq(X_t, 0), sp.Eq(dXdt, 0)]
        steps.append({"d": "Touching the y-axis means x = 0 AND dx/dt = 0 at the same point",
                       "s": "For the curve to touch the y-axis, we need x equal to zero and dx by dt equal to zero, "
                            "at the same point."})
        describe = ("y", Y_t)
    elif "horizontal tangent" in low:
        conditions = [sp.Eq(dYdt, 0)]
        steps.append({"d": "A horizontal tangent means dy/dt = 0", "s": "A horizontal tangent means the slope, dy by dt, is zero."})
        describe = ("point", (X_t, Y_t))
    elif "vertical tangent" in low:
        conditions = [sp.Eq(dXdt, 0)]
        steps.append({"d": "A vertical tangent means dx/dt = 0", "s": "A vertical tangent means dx by dt is zero."})
        describe = ("point", (X_t, Y_t))
    else:
        pm = re.search(r"pass(?:es)?\s+through\s*\(?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)?", low)
        if pm:
            p, q = sp.nsimplify(pm.group(1)), sp.nsimplify(pm.group(2))
            conditions = [sp.Eq(X_t, p), sp.Eq(Y_t, q)]
            steps.append({"d": f"Passing through ({fmt(p)}, {fmt(q)}) means x = {fmt(p)} and y = {fmt(q)}",
                           "s": f"Passing through that point means x equals {fmt(p)} and y equals {fmt(q)}, at the same t."})
            describe = ("confirm", None)

    if conditions is None:
        raise HTTPException(400, "I found a parametric curve, but not a condition I currently recognize "
                                  "(I understand: touches the x-axis/y-axis, horizontal/vertical tangent, "
                                  "or passes through a point). Let me know exactly what's being asked.")

    try:
        solutions = sp.solve(conditions, unknowns, dict=True)
    except Exception:
        solutions = []
    solutions = [s for s in solutions if all(v.is_real is not False for v in s.values())]
    if not solutions:
        steps.append({"d": "No closed-form solution found for this system",
                       "s": "I wasn't able to find a closed-form solution to this system — this particular "
                            "curve may need a numerical or more specialized approach."})
        return steps

    sol = solutions[0]
    for L in extra_letters:
        sym = local_dict[L]
        if sym in sol:
            steps.append({"d": f"Solving the system gives {L} = {fmt(sol[sym])}",
                           "s": f"Solving that system, {L} equals {fmt(sol[sym])}."})
    if T in sol:
        steps.append({"d": f"…and t = {fmt(sol[T])}", "s": f"And the parameter t equals {fmt(sol[T])} at that point."})

    kind, target = describe
    if kind == "confirm":
        steps.append({"d": "So a value of t exists where the curve passes through that point — confirmed above.",
                       "s": "So a value of t does exist where the curve passes through that point."})
    elif kind == "point":
        xv, yv = sp.nsimplify(target[0].subs(sol)), sp.nsimplify(target[1].subs(sol))
        steps.append({"d": f"∴ Point = ({fmt(xv)}, {fmt(yv)})", "s": f"So the point is {fmt(xv)}, {fmt(yv)}."})
    else:
        val = sp.nsimplify(target.subs(sol))
        steps.append({"d": f"∴ {kind} = {fmt(val)}", "s": f"So {kind} equals {fmt(val)}."})
    return steps


def solve_ode_order_degree(problem: str):
    """Classifies (doesn't solve) a differential equation by order and degree —
    a conceptual identification question, not a computation."""
    m = re.search(r"differential equation\s*(.*)", problem, re.IGNORECASE)
    eq_text = (m.group(1) if m else problem).strip().rstrip("?").strip()
    eq_display = normalize(eq_text).replace("**", "^")  # readable form for the whiteboard

    order_syms = {}

    def repl(mm):
        order = int(mm.group(1) or mm.group(2) or 1)
        order_syms.setdefault(order, sp.Symbol(f"D{order}"))
        return f" {order_syms[order]} "

    subbed = DERIV_RE.sub(repl, normalize(eq_text)).replace("^", "**")
    if not order_syms:
        raise HTTPException(400, "Couldn't find any dy/dx-style derivative in that — "
                                  "try something like 'd^2y/dx^2 + 3(dy/dx)^2 + y = 0'.")

    lhs_text, rhs_text = subbed.split("=") if "=" in subbed else (subbed, "0")
    local_dict = {str(s): s for s in order_syms.values()}
    local_dict["y"] = sp.Symbol("y")
    expr = (parse_expr(lhs_text, local_dict=local_dict, transformations=TRANSFORMS) -
            parse_expr(rhs_text, local_dict=local_dict, transformations=TRANSFORMS))

    order = max(order_syms.keys())
    highest_sym = order_syms[order]
    highest_name = DERIV_NAMES.get(order, f"d^{order}y/dx^{order}")

    steps = [{"d": f"Given: {eq_display}", "s": "Let's find the order and degree of this differential equation."}]
    steps.append({"d": f"Highest-order derivative present: {highest_name}",
                   "s": f"The highest derivative in the equation is order {order}."})
    steps.append({"d": f"∴ Order = {order}", "s": f"So the order of the differential equation is {order}."})

    try:
        degree = sp.Poly(expr, highest_sym).degree()
        steps.append({"d": f"The equation is a polynomial in {highest_name} — it appears to the power {degree}",
                       "s": f"Since the equation is a polynomial in the highest derivative, the degree is the "
                            f"power it's raised to, which is {degree}."})
        steps.append({"d": f"∴ Degree = {degree}", "s": f"So the degree is {degree}."})
    except sp.PolynomialError:
        steps.append({"d": f"The equation isn't a polynomial in {highest_name} (a radical or fraction is involved) "
                            f"— clear that first before reading off the degree",
                       "s": "Since the highest derivative appears under a radical or fraction, the degree "
                            "isn't directly readable until that's cleared first."})
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
        if "parametric" in low and (param_steps := solve_parametric_curve(problem)):
            steps = param_steps
            topic = "Parametric Curves"
        elif any(w in low for w in ("velocity", "acceleration", "speed")) and (motion_steps := solve_motion(problem)):
            steps = motion_steps
            topic = "Applied Calculus (Motion)"
        elif ("order" in low and "degree" in low) or DERIV_RE.search(problem):
            steps = solve_ode_order_degree(problem)
            topic = "Differential Equations (Order & Degree)"
        elif is_vector_problem(problem):
            # Checked BEFORE complex numbers: "i - 2j" contains a standalone i, and would
            # otherwise be grabbed as a complex-number problem.
            steps = solve_vector(problem)
            topic = "Vectors"
        elif re.search(r"(?<![a-zA-Z])i(?![a-zA-Z])", problem) and re.search(r"\d", problem) and \
                ("/" in problem or "+" in problem or "-" in problem) and "sin" not in low and "cos" not in low:
            steps = solve_complex(problem)
            topic = "Complex Numbers"
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
        elif re.search(r"(?i)integrate|integral|∫", problem):
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
