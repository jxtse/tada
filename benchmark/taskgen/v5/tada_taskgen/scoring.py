#!/usr/bin/env python3
"""Task-independent v5 scorer. Bounded XLSX reader retained from accepted MAUD r7.

Original labels: decode published per-question encoding, then exact equality.
Aggregation averages recall across gold-present classes, then across questions.
Scope uses precision/recall independently of content. Derived credit is at most 5%,
computed only from submitted values with every underlying label correct.
"""
import json
import math
import os
import posixpath
import re
from pathlib import Path
import sys
import zipfile
import xml.parsers.expat
from collections import defaultdict

LIMITS = {
    "file_bytes": 150 * 1024 * 1024,
    "zip_entries": 10000,
    "package_part_bytes": 8 * 1024 * 1024,
    "sheets_in_workbook": 1000,
    "sheet_part_bytes": 128 * 1024 * 1024,
    "cell_elements_per_sheet": 1000000,
    "cells_per_sheet": 200000,
    "merge_ranges_per_sheet": 2000,
    "merge_area_cells": 100000,
    "sst_part_bytes": 1024 * 1024 * 1024,
    "shared_strings_scanned": 20000000,
    "text_chars": 2000,
}
HEADER_SCAN_ROWS = 50
MAX_INT = 999999
FORMULA_NO_CACHE = "\x00<<formula-without-cached-value>>"
INVALID = "\x00<<invalid-cell>>"
INT_RX = re.compile(r"(\d{1,6})(?:\.0{1,6})?")
REF_RX = re.compile(r"^\$?([A-Za-z]{1,3})\$?(\d{1,7})$")


class LimitExceeded(Exception):
    pass


class _Stop(Exception):
    pass


# ------------------------------------------------------------------ normalisation
def safe_str(v):
    try:
        s = str(v)
    except Exception:
        return INVALID
    if len(s) > LIMITS["text_chars"]:
        return INVALID
    return s


def hdr(s):
    if s is None:
        return ""
    s = safe_str(s)
    if s == INVALID:
        return ""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def norm_text(v):
    try:
        if v is None:
            return ""
        if isinstance(v, str) and v in (FORMULA_NO_CACHE, INVALID):
            return INVALID
        if isinstance(v, bool):
            return str(v).casefold()
        if isinstance(v, float):
            if not math.isfinite(v):
                return INVALID
            if v.is_integer() and abs(v) <= MAX_INT:
                v = int(v)
        s = safe_str(v)
        if s == INVALID:
            return INVALID
        for a, b in (("\u201c", '"'), ("\u201d", '"'), ("\u2018", "'"), ("\u2019", "'"), ("\u00a0", " "),
                     ("\u2013", "-"), ("\u2014", "-")):
            s = s.replace(a, b)
        s = re.sub(r"\s+", " ", s).strip()
        return s.casefold()
    except Exception:
        return INVALID


def norm_id(v):
    s = norm_text(v)
    if s == INVALID:
        return INVALID
    s = s.upper().replace(" ", "")
    if "/" in s:
        s = s.rsplit("/", 1)[1]
    if s.endswith(".TXT"):
        s = s[:-4]
    return s


def norm_int(v):
    """Non-negative integer with at most 6 digits, else None.  Never goes through float() for strings."""
    try:
        if v is None or isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v if 0 <= v <= MAX_INT else None
        if isinstance(v, float):
            if not math.isfinite(v) or not v.is_integer() or v < 0 or v > MAX_INT:
                return None
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            if len(s) > 13:
                return None
            m = INT_RX.fullmatch(s)
            return int(m.group(1)) if m else None
        return None
    except Exception:
        return None


def nonempty(v):
    if v is None:
        return False
    try:
        if isinstance(v, str):
            return v.strip() != ""
        return True
    except Exception:
        return True


def clamp01(x):
    return max(0.0, min(1.0, x))


# ------------------------------------------------------------------ bounded text accumulator (F04)
class BoundedText:
    """Collects character data for one cell / shared string.  Stores pieces only while the running
    length stays <= limit; at the first overflow it drops what it holds and ignores everything else.
    At most `limit` non-empty pieces are ever held, and nothing is concatenated before value()."""
    __slots__ = ("limit", "parts", "length", "over", "seen")

    def __init__(self, limit):
        self.limit = limit
        self.parts = []
        self.length = 0
        self.over = False
        self.seen = False

    def add(self, data):
        self.seen = True
        if self.over or not data:
            return
        if self.length + len(data) > self.limit:
            self.over = True
            self.parts = []
            return
        self.parts.append(data)
        self.length += len(data)

    def value(self):
        if self.over:
            return INVALID
        return "".join(self.parts)


# ------------------------------------------------------------------ bounded XLSX reader
def _local(name):
    return name.rsplit("}", 1)[-1]


def _attr(attrs, local, namespaced=None):
    for k, v in attrs.items():
        if _local(k) == local:
            if namespaced is None or (("}" in k) == namespaced):
                return v
    return None


def _parser():
    p = xml.parsers.expat.ParserCreate(namespace_separator="}")
    p.buffer_text = True

    def no_dtd(*_a):
        raise LimitExceeded("DTD / entity declarations are not allowed")

    p.StartDoctypeDeclHandler = no_dtd
    p.EntityDeclHandler = no_dtd
    return p


def _stream(zf, info, cap, start, end=None, chars=None):
    p = _parser()
    p.StartElementHandler = start
    if end is not None:
        p.EndElementHandler = end
    if chars is not None:
        p.CharacterDataHandler = chars
    total = 0
    try:
        with zf.open(info) as fh:
            while True:
                chunk = fh.read(1 << 16)
                if not chunk:
                    break
                total += len(chunk)
                if total > cap:
                    raise LimitExceeded("part %s exceeds %d uncompressed bytes" % (info.filename, cap))
                p.Parse(chunk, False)
        p.Parse(b"", True)
    except _Stop:
        pass
    return total


def _read_small(zf, info):
    cap = LIMITS["package_part_bytes"]
    with zf.open(info) as fh:
        data = fh.read(cap + 1)
    if len(data) > cap:
        raise LimitExceeded("package part %s exceeds %d bytes" % (info.filename, cap))
    return data


def _parse_small(data, start):
    p = _parser()
    p.StartElementHandler = start
    p.Parse(data, True)


def _resolve(base, target):
    if target.startswith("/"):
        return posixpath.normpath(target.lstrip("/"))
    return posixpath.normpath(posixpath.join(base, target))


def _rels(zf, info):
    out = []

    def start(name, attrs):
        if _local(name) == "Relationship":
            if (_attr(attrs, "TargetMode") or "").lower() == "external":
                return
            out.append((_attr(attrs, "Id") or "", _attr(attrs, "Type") or "", _attr(attrs, "Target") or ""))

    _parse_small(_read_small(zf, info), start)
    return out


def open_package(path):
    size = os.path.getsize(path)
    if size > LIMITS["file_bytes"]:
        raise LimitExceeded("file is %d bytes (limit %d)" % (size, LIMITS["file_bytes"]))
    zf = zipfile.ZipFile(path)
    infos = zf.infolist()
    if len(infos) > LIMITS["zip_entries"]:
        raise LimitExceeded("%d zip entries" % len(infos))
    by_lower = {}
    for i in infos:
        by_lower.setdefault(i.filename.lstrip("/").lower(), i)

    def get(name):
        return by_lower.get(name.lstrip("/").lower()) if name else None

    wb_part = "xl/workbook.xml"
    rr = get("_rels/.rels")
    if rr is not None:
        for _rid, typ, tgt in _rels(zf, rr):
            if typ.endswith("/officeDocument"):
                wb_part = _resolve("", tgt)
                break
    wbi = get(wb_part)
    if wbi is None:
        raise ValueError("workbook part %s not found" % wb_part)
    sheets = []

    def wstart(name, attrs):
        if _local(name) == "sheet":
            if len(sheets) >= LIMITS["sheets_in_workbook"]:
                raise LimitExceeded("too many sheets")
            sheets.append((_attr(attrs, "name") or "", _attr(attrs, "id", namespaced=True) or ""))

    _parse_small(_read_small(zf, wbi), wstart)
    base = posixpath.dirname(wb_part)
    rels_name = posixpath.join(base, "_rels", posixpath.basename(wb_part) + ".rels")
    targets, sst = {}, None
    ri = get(rels_name)
    if ri is not None:
        for rid, typ, tgt in _rels(zf, ri):
            targets[rid] = _resolve(base, tgt)
            if typ.endswith("/sharedStrings"):
                sst = _resolve(base, tgt)
    if sst is None and get(posixpath.join(base, "sharedStrings.xml")) is not None:
        sst = posixpath.join(base, "sharedStrings.xml")
    return zf, get, [(n, targets.get(r)) for n, r in sheets], sst


def _col_index(letters):
    n = 0
    for ch in letters.upper():
        n = n * 26 + (ord(ch) - 64)
    return n


def _range_area(ref):
    try:
        parts = ref.split(":")
        if len(parts) == 1:
            parts = parts * 2
        a, b = REF_RX.match(parts[0].strip()), REF_RX.match(parts[1].strip())
        if not a or not b:
            return 0
        c1, r1 = _col_index(a.group(1)), int(a.group(2))
        c2, r2 = _col_index(b.group(1)), int(b.group(2))
        return (abs(c2 - c1) + 1) * (abs(r2 - r1) + 1)
    except Exception:
        return 0


def read_sheet(zf, info):
    """Stream one canonical worksheet part.  -> (cells {(r, c): value or ('SST', idx)}, sst indices, meta)."""
    cells = {}
    need = set()
    st = {"row": 0, "col": 0, "cur": None, "in_v": False, "in_is_t": False, "rph": 0,
          "n_c": 0, "merges": 0, "max_merge": 0, "dimension": None, "over_cells": 0}
    tmax = LIMITS["text_chars"]

    def start(name, attrs):
        tag = _local(name)
        if tag == "row":
            r = _attr(attrs, "r")
            if r is not None and r.isdigit() and 0 < int(r) <= 1048576:
                st["row"] = int(r)
            else:
                st["row"] += 1
            st["col"] = 0
        elif tag == "c":
            st["n_c"] += 1
            if st["n_c"] > LIMITS["cell_elements_per_sheet"]:
                raise LimitExceeded("more than %d cell elements" % LIMITS["cell_elements_per_sheet"])
            ref = _attr(attrs, "r")
            m = REF_RX.match(ref) if ref else None
            if m and _col_index(m.group(1)) <= 16384 and 0 < int(m.group(2)) <= 1048576:
                c, r = _col_index(m.group(1)), int(m.group(2))
                st["row"] = r
            else:
                c, r = st["col"] + 1, max(st["row"], 1)
            st["col"] = c
            st["cur"] = {"r": r, "c": c, "t": _attr(attrs, "t") or "n", "f": False,
                         "v": BoundedText(tmax), "is": BoundedText(tmax)}
            st["in_v"] = st["in_is_t"] = False
        elif tag == "v" and st["cur"] is not None:
            st["in_v"] = True
            st["cur"]["v"].seen = True
        elif tag == "f" and st["cur"] is not None:
            st["cur"]["f"] = True
        elif tag == "rPh":
            st["rph"] += 1
        elif tag == "t" and st["cur"] is not None and st["rph"] == 0:
            st["in_is_t"] = True
            st["cur"]["is"].seen = True
        elif tag == "mergeCell":
            st["merges"] += 1
            if st["merges"] > LIMITS["merge_ranges_per_sheet"]:
                raise LimitExceeded("more than %d merged ranges" % LIMITS["merge_ranges_per_sheet"])
            area = _range_area(_attr(attrs, "ref") or "")
            st["max_merge"] = max(st["max_merge"], area)
            if area > LIMITS["merge_area_cells"]:
                raise LimitExceeded("merged range %s covers %d cells" % (_attr(attrs, "ref"), area))
        elif tag == "dimension":
            st["dimension"] = (_attr(attrs, "ref") or "")[:40]

    def chars(data):
        cur = st["cur"]
        if cur is None:
            return
        if st["in_v"]:
            cur["v"].add(data)          # bounded as it arrives (F04)
        elif st["in_is_t"]:
            cur["is"].add(data)         # all rich-text runs of the cell share one bound

    def end(name):
        tag = _local(name)
        cur = st["cur"]
        if tag == "v":
            st["in_v"] = False
        elif tag == "t":
            st["in_is_t"] = False
        elif tag == "rPh":
            st["rph"] = max(0, st["rph"] - 1)
        elif tag == "c" and cur is not None:
            st["cur"] = None
            st["in_v"] = st["in_is_t"] = False
            if cur["v"].over or cur["is"].over:
                st["over_cells"] += 1
            val = _cell_value(cur)
            if val is None:
                return
            if isinstance(val, tuple):
                need.add(val[1])
            elif not nonempty(val):
                return
            cells[(cur["r"], cur["c"])] = val
            if len(cells) > LIMITS["cells_per_sheet"]:
                raise LimitExceeded("more than %d non-empty cells" % LIMITS["cells_per_sheet"])

    nbytes = _stream(zf, info, LIMITS["sheet_part_bytes"], start, end, chars)
    meta = {"bytes": nbytes, "cell_elements": st["n_c"], "cells": len(cells), "merged_ranges": st["merges"],
            "largest_merge_area": st["max_merge"], "dimension": st["dimension"],
            "cells_over_text_limit": st["over_cells"]}
    return cells, need, meta


def _cell_value(cur):
    t, f = cur["t"], cur["f"]
    vb, ib = cur["v"], cur["is"]
    if vb.over or ib.over:
        return INVALID
    v = vb.value() if vb.seen else None
    if t == "inlineStr":
        if ib.seen:
            return ib.value()
        return v if v is not None else (FORMULA_NO_CACHE if f else None)
    if v is None or v == "":
        return FORMULA_NO_CACHE if f else None
    if t == "s":
        s = v.strip()
        if s.isdigit() and len(s) <= 9:
            return ("SST", int(s))
        return INVALID
    if t in ("str", "d"):
        return v
    if t == "b":
        return v.strip() == "1"
    if t == "e":
        return INVALID
    s = v.strip()
    try:
        if re.fullmatch(r"-?\d{1,15}", s):
            return int(s)
        x = float(s)
        return x if math.isfinite(x) else INVALID
    except Exception:
        return INVALID


def read_sst(zf, info, need, out, status):
    """Stream sharedStrings into `out` (filled in place, so strings parsed before any failure are kept).
    Keeps only indices in `need`; stops after the largest one.  `status` is updated as the scan goes."""
    if not need:
        return
    maxi = max(need)
    tmax = LIMITS["text_chars"]
    st = {"i": -1, "in_si": False, "in_t": False, "rph": 0, "buf": None}

    def start(name, attrs):
        tag = _local(name)
        if tag == "si":
            st["i"] += 1
            if st["i"] >= LIMITS["shared_strings_scanned"]:
                raise LimitExceeded("more than %d shared strings scanned" % LIMITS["shared_strings_scanned"])
            st["in_si"] = True
            st["buf"] = BoundedText(tmax) if st["i"] in need else None
        elif tag == "rPh":
            st["rph"] += 1
        elif tag == "t" and st["in_si"] and st["rph"] == 0:
            st["in_t"] = True

    def chars(data):
        if st["in_t"] and st["buf"] is not None:
            st["buf"].add(data)         # bounded as it arrives (F04)

    def end(name):
        tag = _local(name)
        if tag == "t":
            st["in_t"] = False
        elif tag == "rPh":
            st["rph"] = max(0, st["rph"] - 1)
        elif tag == "si":
            st["in_si"] = False
            st["in_t"] = False
            if st["buf"] is not None:
                out[st["i"]] = st["buf"].value()
                st["buf"] = None
            status["parsed_through"] = st["i"]
            if st["i"] >= maxi:
                status["stopped_after_max_needed"] = True
                raise _Stop()

    _stream(zf, info, LIMITS["sst_part_bytes"], start, end, chars)
    if not status.get("stopped_after_max_needed"):
        status["reached_end_of_part"] = True
        status["strings_in_part"] = st["i"] + 1


def select_sheet(titles, canonical):
    if canonical in titles:
        return canonical, "exact"
    al = [t for t in titles if hdr(t) == hdr(canonical)]
    if len(al) == 1:
        return al[0], "alias"
    if len(al) > 1:
        return None, "ambiguous alias: %r" % al
    return None, "missing"


def load_book(path, canonical_sheets):
    """-> ({canonical: rows or None}, {canonical: status}, details).  rows = [(row, {col: value})]."""
    zf, get, sheets, sst_name = open_package(path)
    titles = [n for n, _ in sheets]
    part_of = {}
    for n, p in sheets:
        part_of.setdefault(n, p)
    det = {"sheet_selection": {}, "pre_check": {}, "problems": {}}
    raw, status = {}, {}
    for canon in canonical_sheets:
        t, how = select_sheet(titles, canon)
        det["sheet_selection"][canon] = {"title": t, "how": how}
        if t is None:
            status[canon] = "missing"
            continue
        info = get(part_of.get(t))
        if info is None:
            status[canon] = "abnormal"
            det["problems"][canon] = "worksheet part not found"
            continue
        try:
            cells, need, meta = read_sheet(zf, info)
            raw[canon] = (cells, need)
            det["pre_check"][canon] = meta
            status[canon] = "ok"
        except Exception as e:  # limits, malformed XML, bad zip member, memory
            status[canon] = "abnormal"
            det["problems"][canon] = "%s: %s" % (type(e).__name__, str(e)[:300])

    # ---- shared strings, tracked per sheet (F03)
    need_all = set()
    for _cells, nd in raw.values():
        need_all |= nd
    sst = {}
    scan = {"part": sst_name, "needed": len(need_all), "max_needed": (max(need_all) if need_all else None),
            "parsed_through": -1, "error": None}
    if need_all:
        si = get(sst_name) if sst_name else None
        if si is None:
            scan["error"] = "shared strings part not found"
        else:
            try:
                read_sst(zf, si, need_all, sst, scan)
            except Exception as e:  # strings already in `sst` are kept
                scan["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
    per_sheet = {}
    for canon, (_cells, nd) in raw.items():
        unresolved = sorted(i for i in nd if i not in sst)
        per_sheet[canon] = {"needed": len(nd), "resolved": len(nd) - len(unresolved),
                            "unresolved": len(unresolved), "unresolved_first": unresolved[:10]}
        if unresolved:
            det["problems"]["sharedStrings:" + canon] = (
                "%d shared-string indices unresolved (%s); those cells are invalid values"
                % (len(unresolved), scan["error"] or "index beyond the strings in the part"))
    det["shared_strings"] = {"scan": scan, "per_sheet": per_sheet}

    book = {}
    for canon, (cells, _nd) in raw.items():
        rows = defaultdict(dict)
        for (r, c), v in cells.items():
            if isinstance(v, tuple):
                v = sst.get(v[1], INVALID)
            if nonempty(v):
                rows[r][c] = v
        book[canon] = sorted(rows.items())
    try:
        zf.close()
    except Exception:
        pass
    return book, status, det


# ------------------------------------------------------------------ records
def parse_sheet(rows, cols, keys):
    """-> (records, extra rows above the header, header_found)."""
    if not rows:
        return [], 0, False
    want = {hdr(c): c for c in cols}
    keyh = [hdr(k) for k in keys]
    hpos = None
    for i, (_r, vals) in enumerate(rows[:HEADER_SCAN_ROWS]):
        hs = set(hdr(v) for v in vals.values())
        if all(k in hs for k in keyh):
            hpos = i
            break
    if hpos is None:
        return [], len(rows), False
    colmap = {}
    for c in sorted(rows[hpos][1]):
        h = hdr(rows[hpos][1][c])
        if h in want and want[h] not in colmap:
            colmap[want[h]] = c
    recs = []
    for _r, vals in rows[hpos + 1:]:
        recs.append({c: vals.get(colmap[c]) if c in colmap else None for c in cols})
    return recs, hpos, True


def group(recs, keys, expected):
    occ = defaultdict(list)
    bad = 0
    for r in recs:
        k = tuple(norm_id(r.get(c)) for c in keys)
        if any(x in ("", INVALID) for x in k) or k not in expected:
            bad += 1
            continue
        occ[k].append(r)
    dups = sum(len(v) - 1 for v in occ.values())
    return occ, bad + dups


def precision(matched, extras):
    return 1.0 if matched + extras == 0 else matched / float(matched + extras)


def _sheet(parsed, name):
    p = parsed.get(name)
    if p is None:
        return "missing", [], 0
    return p.get("status", "ok"), p.get("records", []), p.get("extra", 0)


def _key(values):
    return tuple(norm_id(v) for v in values)


def decoded(value, encoding):
    # Encoding maps are emitted from the approved source adapter; no fuzzy matches.
    for answer, visible in encoding:
        if type(value) is type(visible) and value == visible:
            return answer
    return None



LABEL_AGGREGATION = 'per_question_chance_corrected_balanced_accuracy'


def balanced_label_scores(cells, contributions):
    """Chance-correct each question's present-class recall using its full codebook.

    Every gold cell stays in its original class denominator, including missing
    or invalid submissions. Per-cell contributions include sheet precision
    before chance correction. k is never reduced to the gold-present classes.
    """
    grouped = defaultdict(lambda: defaultdict(list))
    item_questions = defaultdict(list)
    codebooks = {}
    for cell in cells:
        question = cell['question']
        if cell.get('kind') == 'benchmark_item':
            item_questions[question].append(contributions[cell['id']])
            continue
        encoding = tuple(tuple(pair) for pair in cell['encoding'])
        if (len(encoding) < 2 or any(len(pair) != 2 for pair in encoding)
                or len({pair[0] for pair in encoding}) != len(encoding)
                or len({pair[1] for pair in encoding}) != len(encoding)):
            raise ValueError('chance correction requires a bijective codebook with at least two options')
        if question in codebooks and codebooks[question] != encoding:
            raise ValueError('inconsistent codebook for question: ' + question)
        if cell['answer'] not in {pair[0] for pair in encoding}:
            raise ValueError('gold answer absent from codebook: ' + question)
        codebooks[question] = encoding
        grouped[question][cell['answer']].append(contributions[cell['id']])
    questions = {}
    for question, classes in grouped.items():
        recalls = {answer: sum(values) / len(values) for answer, values in classes.items()}
        balanced = sum(recalls.values()) / len(recalls)
        option_count = len(codebooks[question])
        chance = 1.0 / option_count
        questions[question] = {
            'present_classes': sorted(classes),
            'class_counts': {answer: len(values) for answer, values in classes.items()},
            'class_recalls': recalls,
            'balanced_accuracy': balanced,
            'codebook_option_count': option_count,
            'chance_floor': chance,
            'score': max(0.0, (balanced - chance) / (1.0 - chance)),
        }
    questions.update({q: {'score': sum(values) / len(values), 'items': len(values),
                          'verification': 'original_deterministic_checker'} for q, values in item_questions.items()})
    aggregate = sum(q['score'] for q in questions.values()) / len(questions) if questions else 0.0
    return aggregate, questions


def score_records(parsed, gold):
    if gold.get('label_aggregation', LABEL_AGGREGATION) != LABEL_AGGREGATION:
        raise ValueError('unsupported label aggregation')
    if 'sources' in gold:
        identities = []
        for cell in gold['cells']:
            original = cell.get('origin', {})
            source = original.get('source')
            if source not in gold['sources'] or cell.get('source') != source:
                raise ValueError('invalid original source identity')
            if gold.get('qualified_labels') and any(
                    cell[key] != source + '::' + original.get(key, '') for key in ('record', 'question')):
                raise ValueError('source-qualified label identity differs from original')
            if sum(type(answer) is type(cell['answer']) and answer == cell['answer']
                   for answer, _ in cell['encoding']) != 1:
                raise ValueError('original answer type/value differs from codebook')
            identities.append((source, original.get('record'), original.get('question')))
        if len(set(identities)) != len(identities):
            raise ValueError('duplicate original source label')
    lookup, sheet_details, penalties = {}, {}, {}
    for name, spec in gold['sheets'].items():
        st, recs, above = _sheet(parsed, name)
        expected = {_key([r[c] for c in spec['key_columns']]) for r in spec['rows'].values()}
        occurrences, extra = group(recs, spec['key_columns'], expected)
        extra += above
        penalties[name] = precision(len(occurrences), extra) if st == 'ok' else 0.0
        lookup[name] = occurrences
        sheet_details[name] = {'status': st, 'extra_rows': extra, 'matched_keys': len(occurrences),
                               'precision': penalties[name]}

    def submitted(cell):
        return [r.get(cell['column']) for r in lookup.get(cell['sheet'], {}).get(_key(cell['key']), [])]

    correct, decoded_values = {}, {}
    label_values, item_cells = {}, defaultdict(list)
    item_values = defaultdict(lambda: defaultdict(list))
    for cell in gold['cells']:
        vals = submitted(cell)
        if cell.get('kind') == 'benchmark_item':
            try:
                from .benchmark_checkers import score_answer
            except ImportError:
                from benchmark_checkers import score_answer
            dec = []
            for value in vals:
                try:
                    dec.append(json.loads(value) if isinstance(value, str) else value)
                except (ValueError, TypeError):
                    dec.append(value)
            item_scores = [float(score_answer(v, cell['answer'], cell['checker'], root=gold.get('_task_root'))) for v in dec]
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in item_scores):
                raise ValueError('invalid official checker result')
            accuracy = min(item_scores, default=0.0)
            ok = bool(dec) and accuracy == 1.0
        else:
            dec = [decoded(v, cell['encoding']) for v in vals]
            ok = bool(dec) and all(v == cell['answer'] for v in dec)
            accuracy = float(ok)
        correct[cell['id']] = ok
        decoded_values[cell['id']] = dec[0] if dec else None
        value = accuracy * penalties[cell['sheet']]
        label_values[cell['id']] = value
        item_cells[cell['work_item']].append(cell)
    facts = []
    fact_correct = {}
    for fact in gold.get('facts', []):
        vals = submitted(fact)
        ok = bool(vals) and all(type(v) is type(fact['value']) and v == fact['value'] for v in vals)
        fact_correct[fact['id']] = ok
        decoded_values[fact['id']] = vals[0] if vals else None
        value = float(ok) * penalties[fact['sheet']]
        if fact.get('counted_in') == 'scope':
            continue  # scope status is scored once, by the scope F1 component
        facts.append(value)
        item_values[fact['work_item']]['facts'].append(value)
    scope_values, scopes = [], {}
    for scope in gold.get('scopes', []):
        predicted = set()
        for key, rows in lookup.get(scope['sheet'], {}).items():
            if all(r.get(scope['column']) == scope.get('included_value', 'Yes') for r in rows):
                predicted.add(key)
        expected = {_key(v) for v in scope['expected_keys']}
        tp = len(predicted & expected)
        p = tp / len(predicted) if predicted else 0.0
        r = tp / len(expected) if expected else float(not predicted)
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if not expected and not predicted:
            f1 = 1.0
        value = f1 * penalties[scope['sheet']]
        scope_values.append(value)
        scopes[scope['work_item']] = {'precision': p, 'recall': r, 'f1': value}
    consistency = []
    for check in gold.get('consistency', []):
        dependencies = check['dependencies']
        vals = submitted(check)
        allowed = bool(dependencies) and all(correct.get(c, fact_correct.get(c, False)) for c in dependencies)
        own = [decoded_values.get(c) for c in dependencies]
        if check['operation'] == 'count_option':
            target = sum(v == check['option'] for v in own)
        elif check['operation'] == 'count_answered':
            target = sum(v is not None for v in own)
        elif check['operation'] == 'count_changed':
            target = sum(a != b for a, b in zip(own[::2], own[1::2]))
        else:
            allowed, target = False, None
        ok = allowed and bool(vals) and all(norm_int(v) == target for v in vals)
        value = float(ok) * penalties[check['sheet']]
        consistency.append(value)
        item_values[check.get('work_item', 'summary')]['consistency'].append(value)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    labels, question_details = balanced_label_scores(gold['cells'], label_values)
    parts = {'labels': labels, 'facts': mean(facts), 'scope': mean(scope_values),
             'consistency': mean(consistency)}
    weights = gold['weights']
    if weights.get('consistency', 0) > .05 or abs(sum(weights.values()) - 1) > 1e-9:
        raise ValueError('invalid scorer weights')
    item_details = {}
    for item in sorted(set(item_cells) | set(item_values)):
        components = {component: mean(values) for component, values in item_values[item].items()}
        if item_cells[item]:
            components['labels'], per_question = balanced_label_scores(item_cells[item], label_values)
        else:
            per_question = {}
        total_weight = sum(weights.get(component, 0) for component in components)
        value = (next(iter(components.values())) if len(components) == 1 else
                 sum(weights.get(component, 0) * value for component, value in components.items()) / total_weight
                 if total_weight else 0.0)
        item_details[item] = {'score': value, 'components': components, 'per_question': per_question}
    # A component with no scored items (e.g. no document facts) is dropped and the others renormalised.
    present = {'labels': bool(gold['cells']), 'facts': bool(facts), 'scope': bool(scope_values),
               'consistency': bool(consistency)}
    active = sum(weights.get(k, 0) for k, on in present.items() if on)
    reward = clamp01(sum(weights.get(k, 0) * v for k, v in parts.items() if present[k]) / active) if active else 0.0
    return reward, {'reward': reward, 'components': parts, 'weights': weights,
                    'sheets': sheet_details, 'scope': scopes, 'cell_correct': correct,
                    'label_aggregation': LABEL_AGGREGATION, 'per_question': question_details,
                    'per_work_item': {k: v['score'] for k, v in item_details.items()},
                    'per_work_item_details': item_details}


def score(path, gold):
    try:
        book, status, reader_details = load_book(path, gold['sheets'])
        parsed = {}
        for name, spec in gold['sheets'].items():
            st = status.get(name, 'missing')
            rows, extras, found = parse_sheet(book.get(name, []), spec['columns'], spec['key_columns'])
            parsed[name] = {'status': st if found else 'abnormal', 'records': rows, 'extra': extras}
        reward, details = score_records(parsed, gold)
        details.update(reader_details)
        details['limits'] = dict(LIMITS)
        return reward, details
    except Exception as exc:
        return 0.0, {'reward': 0.0, 'error': f'{type(exc).__name__}: {str(exc)[:300]}', 'limits': LIMITS}


def main():
    out = sys.argv[3] if len(sys.argv) > 3 else '.'
    os.makedirs(out, exist_ok=True)
    try:
        with open(sys.argv[2], encoding='utf-8') as fh:
            gold = json.load(fh)
        gold['_task_root'] = str(Path(sys.argv[2]).resolve().parent.parent)
        reward, details = score(sys.argv[1], gold)
    except BaseException as exc:
        reward, details = 0.0, {'error': f'{type(exc).__name__}: {str(exc)[:300]}'}
    if not math.isfinite(reward):
        reward = 0.0
    with open(os.path.join(out, 'reward.txt'), 'w') as fh:
        fh.write(f'{reward:.6f}\n')
    with open(os.path.join(out, 'details.json'), 'w') as fh:
        json.dump(details, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
