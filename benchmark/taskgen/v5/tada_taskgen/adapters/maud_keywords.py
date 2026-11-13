"""MAUD label-free keyword rules reused from the accepted r7 baseline."""
import re

DEDICATED_QUESTIONS = frozenset({"DP132", "DP113", "DP110", "DP114", "DP086", "DP054", "DP069", "DP089", "DP091"})
Q = "[\"\u201c\u201d]"

def definition(text, term_rx, length=2500):
    rx = re.compile(Q + r"\s*" + term_rx + r"\s*" + Q + r"\s*(?:shall\s+)?(?:mean|means|has the meaning)", re.I)
    m = rx.search(text)
    if not m:
        return ""
    return text[m.start():m.start() + length]


def conduct_window(text, length=25000):
    for m in re.finditer(r"Conduct\s+of\s+(?:the\s+)?(?:Business|Businesses)", text, re.I):
        nxt = text[m.end():m.end() + 2500]
        if "ordinary course" not in nxt.lower():
            continue
        if len(re.findall(r"Section", text[m.end():m.end() + 300])) >= 3:
            continue  # table of contents
        return text[m.start():m.start() + length], True
    m = re.search(r"ordinary course of business", text[20000:], re.I)
    if m:
        s = 20000 + m.start()
        return text[max(0, s - 1500):s + length], False
    return "", False


def first_sentence_with(window, rx):
    for s in re.split(r"(?<=[.;])\s+", window):
        if re.search(rx, s, re.I):
            return s
    return ""


def code(ctx, dr, dp):
    return coded(ctx, dr, dp)[0]


def coded(ctx, dr, dp):
    key = (dr, dp)
    if key not in ctx._code:
        ctx._code[key] = _code(ctx, dr, dp)
    return ctx._code[key]


def _code(ctx, dr, dp):
    """-> (answer, sure).  sure = False when the clause was not located or a fallback default is used."""
    t = ctx.text(dr)
    if dp == "DP132":
        if re.search(r"Exchange\s+Ratio", t):
            if re.search(r"(Cash|Stock)\s+Election", t):
                return ctx.pick(dp, exact="Mixed Cash/Stock: Election"), True
            if re.search(r"Cash\s+Consideration|in\s+cash,?\s+without\s+interest", t, re.I):
                return ctx.pick(dp, exact="Mixed Cash/Stock"), True
            return ctx.pick(dp, exact="All Stock"), True
        return ctx.pick(dp, exact="All Cash"), bool(re.search(r"in\s+cash", t, re.I))
    if dp in ("DP113", "DP110", "DP114", "DP086"):
        w, found = conduct_window(t)
        if dp == "DP113":
            return ctx.pick(dp, exact="Yes" if re.search(r"COVID|pandemic", w[:5000], re.I) else "No"), found
        if dp == "DP110":
            return ctx.pick(dp, exact="Yes" if re.search(r"COVID|pandemic", w[5000:], re.I) else "No"), found
        if dp == "DP114":
            s = first_sentence_with(w[:4000], r"ordinary course")
            m = re.search(r"(commercially reasonable|reasonable best)\s+efforts[^.;]{0,120}ordinary course", s, re.I)
            if m:
                if m.group(1).lower().startswith("reasonable best"):
                    return ctx.pick(dp, exact="Reasonable best efforts"), found
                return ctx.pick(dp, exact="Commercially reasonable efforts"), found
            return ctx.pick(dp, exact="Flat covenant (no efforts standard)", contains=["flat"]), found and bool(s)
        s = first_sentence_with(w[:4000], r"ordinary course")
        od = definition(t, r"[Oo]rdinary\s+[Cc]ourse(?:\s+of\s+[Bb]usiness)?", 800)
        yes = "past practice" in s.lower() or "past practice" in od.lower()
        return ctx.pick(dp, exact="Yes" if yes else "No"), found and bool(s)
    ie = definition(t, r"(?:Company\s+)?Intervening\s+Event")
    if dp == "DP054":
        return ctx.pick(dp, exact="Yes" if ie else "No"), bool(ie)
    if dp == "DP069":
        low = ie.lower()
        if "consequence" in low:
            if re.search(r"consequences[^.;]{0,80}foreseeab", low):
                return ctx.pick(dp, exact="Known, but consequences unknown or not reasonably foreseeable, at signing"), True
            return ctx.pick(dp, exact="Known, but consequences unknown, at signing"), True
        if "foreseeab" in low:
            return ctx.pick(dp, exact="Not known and not reasonably foreseeable at signing"), True
        return ctx.pick(dp, exact="Not known at signing"), bool(ie) and "not known" in low
    if dp == "DP089":
        if re.search(r"(?:aris|occur)\w*[^.]{0,120}(?:after|following|subsequent to)\s+the\s+"
                     r"(?:date\s+(?:of\s+this\s+Agreement|hereof)|execution)", ie, re.I) \
                and "not known" not in ie.lower():
            return ctx.pick(dp, exact="Must occur or arise after signing"), True
        return ctx.pick(dp, exact="May occur or arise prior to signing"), bool(ie)
    if dp == "DP091":
        k = definition(t, r"(?:Company\s+)?Knowledge(?:\s+of\s+the\s+Company)?", 1500) or \
            definition(t, r"[^\"\u201c\u201d]{0,40}Knowledge[^\"\u201c\u201d]{0,20}", 1500)
        if re.search(r"inquiry|investigation|should\s+have\s+known|reasonably\s+(?:be\s+)?expected\s+to\s+know"
                     r"|by\s+virtue\s+of", k, re.I):
            return ctx.pick(dp, exact="Constructive knowledge"), True
        return ctx.pick(dp, exact="Actual knowledge"), bool(k) and "actual" in k.lower()
    return ctx.pick(dp), False


class _Context:
    def __init__(self, text, questions):
        self.value = text
        self.opts = {q: d['options'] for q, d in questions.items()}
        self._code = {}
    def text(self, record):
        return self.value
    def pick(self, dp, exact=None, contains=None):
        options = self.opts[dp]
        if exact in options:
            return exact
        if contains:
            for option in options:
                if all(c.lower() in option.lower() for c in contains):
                    return option
        return options[0]

def predict(question, text, questions):
    return code(_Context(text, questions), 'record', question)
