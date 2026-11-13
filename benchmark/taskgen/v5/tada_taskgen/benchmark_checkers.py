"""Offline benchmark scoring; no model judges or source imports.

Methods mirror local upstream files cited by each adapter's checker metadata.
SQL executes only SELECT/WITH using SQLite read-only, authorizer and VM budget.
"""
import json
import math
import re
import sqlite3
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sql_rows(query, database):
    if not isinstance(query, str) or not re.match(r'^\s*(?:SELECT|WITH)\b', query, re.I):
        raise ValueError('Only a read-only SQL query is accepted')
    con = sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
    con.set_authorizer(lambda action, *args: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)
    remaining = [10000]
    def progress():
        remaining[0] -= 1
        return remaining[0] <= 0
    con.set_progress_handler(progress, 1000)
    try:
        result = con.execute(query).fetchmany(100001)
        if len(result) > 100000:
            raise ValueError('SQL result exceeds bounded verifier row limit')
        return [list(row) for row in result]
    finally:
        con.close()


def score_answer(predicted, expected, spec, root=None):
    """Return the benchmark's deterministic score in [0,1], failing closed."""
    try:
        kind = spec['kind']
        if isinstance(predicted, str):
            try:
                predicted = json.loads(predicted)
            except (ValueError, TypeError):
                pass
        if kind in ('oolong_synth','oolong_real','sqe_derivation','sqe_set_f1'):
            try:from .benchmark_oolong_score import score_oolong
            except ImportError:from benchmark_oolong_score import score_oolong
            return score_oolong(predicted,expected,spec)
        if kind == 'sembench_movie':
            try:from .benchmark_sembench_score import score_answer as sembench_score
            except ImportError:from benchmark_sembench_score import score_answer as sembench_score
            return sembench_score(predicted,expected,spec,root)
        if kind.startswith('loft_') and kind!='loft_sql':
            return loft_score(predicted,expected,spec)
        if kind == 'spider2_table':
            return spider2_table(predicted,expected,spec)
        if kind == 'lro_impute':
            if len(predicted)!=len(expected) or not expected:return 0.0
            return round(sum(str(p)==str(e) for p,e in zip(predicted,expected))/len(expected),4)
        if kind == 'lro_exact_table':
            def table(x):
                if not isinstance(x,list):return [[str(x)]]
                return [[str(v) for v in row] if isinstance(row,list) else [str(row)] for row in x]
            return float(table(predicted)==table(expected))
        if kind in ('lro_order','lro_nmi'):
            return lro_score(predicted,expected,kind)
        if kind == 'semtpch':
            return semtpch_score(predicted,expected,spec)
        if kind == 'dabstep':
            return float(dabstep_score(predicted, expected))
        if kind == 'blade_mcq':
            if isinstance(predicted, dict): return float(canonical(predicted)==canonical(expected))
            if isinstance(predicted, str) and predicted in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ': predicted=ord(predicted)-65
            return float(type(predicted)==int and predicted==spec['correct_index'])
        if kind == 'palimpzest_fields':
            return sum(float(predicted.get(k)==expected[k]) for k in ('sender','subject'))/2
        if kind == 'infiagent':
            p = dict(predicted) if isinstance(predicted,(dict,list)) else dict(re.findall(r"@(\w+)\[(.*?)\]",str(predicted)))
            def equal(a,b):
                if a==b:return True
                try:return abs(float(a)-float(b))<1e-6
                except (TypeError,ValueError):return False
            return float(all(equal(p.get(k),v) for k,v in dict(expected).items()))
        if kind == 'tablebench':
            return tablebench_score(predicted,expected,spec['metric'])
        if kind == 'krama_f1':
            return krama_f1(predicted,expected)
        if kind == 'krama_rae':
            def numeric(v):
                if isinstance(v,str) and v.endswith('%'):return float(v.strip('%'))/100
                return float(v)
            p,e=numeric(predicted),numeric(expected)
            return 1/(1+abs(p-e)/abs(e)) if e else 0.0
        if kind == 'exact':
            return float(canonical(predicted) == canonical(expected))
        if kind == 'normalized_exact':
            return float(str(predicted).strip().lower() == str(expected).strip().lower())
        if kind == 'lro_set_f1':
            # Official forward interface constructs DataFrames, casts cells
            # to str, then matches unique complete rows.
            def rows(table):
                values=[row if isinstance(row,list) else [row] for row in table]
                width=max(map(len,values),default=0)
                return {tuple(str(v) for v in row)+('nan',)*(width-len(row)) for row in values}
            p,e=rows(predicted),rows(expected)
            return round(2*len(p&e)/(len(p)+len(e)),4) if p and e else 0.0
        if kind == 'set_f1':
            a, b = set(map(str, predicted)), set(map(str, expected))
            return round(2 * len(a & b) / (len(a) + len(b)), 4) if a and b else 0.0
        if kind == 'set_exact':
            return float({canonical(v) for v in predicted} == {canonical(v) for v in expected})
        if kind == 'loft_sql':
            normalize = lambda rows: [set(str(f'{float(v):.2f}' if _numeric(v) else v).strip().lower()
                                        for v in row if str(v).strip()) for row in rows]
            p, e = normalize(predicted), normalize(expected)
            return float(len(p) == len(e) and all(row in e for row in p))
        if kind == 'sqlite_result_set':
            base = Path(root or '.')
            path = (base / spec['database']).resolve()
            if not path.is_relative_to(base.resolve()):
                return 0.0
            return float({tuple(v) for v in sql_rows(predicted, path)} ==
                         {tuple(v) for v in sql_rows(expected, path)})
        if kind == 'krama_success':
            def number(value):
                if isinstance(value,(int,float)):return value
                try:
                    return float(str(value).strip().rstrip('%'))
                except ValueError:
                    return value
            p, e = number(predicted), number(expected)
            if isinstance(p, (int, float)) and isinstance(e, (int, float)):
                if not math.isfinite(p) or not math.isfinite(e):
                    return 0.0
                if p == e: return 1.0
                if isinstance(p,int) and isinstance(e,int):return 0.0
                if e == 0: return float(abs(p) < 1e-6)
                return float(abs(p-e)/abs(e) < 1e-6 or abs(p-e) < 1e-6 or abs(p-e) < max(abs(p),abs(e))*1e-4)
            return float(str(predicted).strip().lower() == str(expected).strip().lower())
        raise ValueError('Unsupported deterministic benchmark checker')
    except (ValueError, TypeError, KeyError, sqlite3.Error, OverflowError, AttributeError, IndexError):
        return 0.0


def _numeric(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def dabstep_score(left, right):
    """DABstep question_scorer, including its original numeric/list conventions."""
    from difflib import SequenceMatcher
    if left is None or right is None:return False
    a,b=str(left).strip().lower(),str(right).strip().lower()
    def number(v):
        m=re.search(r'(\d*\.\d+|\d+\.?\d*)%?',v.replace(',','').replace('$',''))
        return float(m.group(1)) if m else None
    def numeric(x,y):
        if x==y:return True
        if x<1 and y<1:return math.isclose(x,y,rel_tol=1e-4,abs_tol=1e-4)
        places=min(len(str(v).split('.')[-1]) if '.' in str(v) else 0 for v in (x,y))
        return round(x,places)==round(y,places) or math.isclose(x,y,rel_tol=1e-4,abs_tol=1e-4)
    comma_numeric=r'^\$?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+[.,]\d+)$'
    x,y=number(a),number(b)
    if re.match(comma_numeric,a) or re.match(comma_numeric,b):return x is not None and y is not None and numeric(x,y)
    if any(c in a+b for c in ',;'):
        split=lambda s:sorted(v.strip() for v in re.split('[,;]',re.sub(r'^\[|\]$','',s)) if v.strip())
        aa,bb=split(a),split(b)
        return len(aa)==len(bb) and all(dabstep_score(x,y) for x,y in zip(aa,bb))
    if x is not None and y is not None:return numeric(x,y)
    if re.sub(r'[^\w]','',a)==re.sub(r'[^\w]','',b):return True
    aa,bb=re.findall(r'\b\w+\b',a),re.findall(r'\b\w+\b',b)
    if (len(aa)==1 or len(bb)==1) and aa and bb:return set(aa)<=set(bb) or set(bb)<=set(aa)
    return SequenceMatcher(None,a,b).ratio()>0.95


def tablebench_normalize(value):
    import string
    value=''.join(c for c in str(value).lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',value).split())


def tablebench_score(predicted,expected,metric):
    from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
    p,e=tablebench_normalize(predicted),tablebench_normalize(expected)
    if metric=='rouge_l':
        a,b=re.findall('[a-z0-9]+',p),re.findall('[a-z0-9]+',e)
        previous=[0]*(len(b)+1)
        for x in a:
            current=[0]
            for j,y in enumerate(b):current.append(previous[j]+1 if x==y else max(previous[j+1],current[-1]))
            previous=current
        return 2*previous[-1]/(len(a)+len(b)) if a and b else 0.0
    pa,ea=p.split(','),e.split(',');score=0
    isnum=lambda v:bool(re.fullmatch(r'-?\d+(\.\d+)?%?',v.strip()))
    precision=min((len(v.split('.')[-1]) if '.' in v else 0 for v in ea if isnum(v) and not v.endswith('%')),default=0)
    for left,right in zip(pa,ea):
        if not isnum(right):score+=float(left==right);continue
        try:
            x,y=Decimal(left),Decimal(right)
            if metric=='tolerance_10':score+=float(x==y if y==0 else abs(x-y)/abs(y)<=Decimal('0.1'))
            else:
                q=Decimal('1.'+'0'*precision)
                score+=float(x.quantize(q,rounding=ROUND_HALF_UP)==y.quantize(q,rounding=ROUND_HALF_UP))
        except (ValueError,InvalidOperation):pass
    return score/len(ea)


def krama_f1(predicted, expected):
    if isinstance(predicted,str) and isinstance(expected,str):return float(predicted.strip().lower()==expected.strip().lower())
    if isinstance(expected,str):expected=json.loads(expected)
    if isinstance(predicted,str):
        try:predicted=json.loads(predicted)
        except ValueError:predicted=[v.strip() for v in predicted.split(',') if v.strip()]
    if not expected:return float(not predicted)
    used=set();matches=0
    for target in expected:
        for j,pred in enumerate(predicted):
            if isinstance(target,str):ok=target.strip().lower()==str(pred).strip().lower()
            elif isinstance(target,(int,float)):
                try:
                    number=float(str(pred).rstrip('%'))/(100 if str(pred).endswith('%') else 1)
                    ok=abs(number-target)/max(abs(target),1e-12)<1e-6
                except (ValueError,TypeError):ok=False
            else:ok=pred==target
            if ok:used.add(j);matches+=1;break
    precision=len(used)/len(predicted) if predicted else 0
    recall=matches/len(expected)
    return 2*precision*recall/(precision+recall) if precision+recall else 0


def spider2_table(predicted,expected,spec):
    """Spider2 compare_pandas_table: column matching, 1e-2 absolute tolerance."""
    tables=expected['tables']
    if isinstance(predicted,dict): predicted=predicted['tables'][0]
    def columns(rows):
        if not rows:return []
        result=[]
        for col in zip(*rows):
            # pandas read_csv infers a numeric column when all nonmissing values parse.
            numeric=all(v in (None,'','NaN','nan') or _numeric(v) for v in col)
            result.append([0 if v in (None,'','NaN','nan') else float(v) if numeric else v for v in col])
        return result
    def matches(a,b):
        if spec.get('ignore_order'):
            key=lambda x:(x is None,str(x),isinstance(x,(int,float)))
            a,b=sorted(a,key=key),sorted(b,key=key)
        if len(a)!=len(b):return False
        for x,y in zip(a,b):
            if isinstance(x,(int,float)) and isinstance(y,(int,float)):
                if not math.isclose(float(x),float(y),abs_tol=1e-2):return False
            elif x!=y:return False
        return True
    pcols=columns(predicted)
    conditions=spec.get('condition_cols') or []
    for i,table in enumerate(tables):
        ecols=columns(table)
        cols=conditions[i] if conditions and isinstance(conditions[0],list) else conditions
        if cols:ecols=[ecols[c] for c in cols]
        if all(any(matches(e,p) for p in pcols) for e in ecols):return 1.0
    return 0.0


def loft_score(predicted,expected,spec):
    import unicodedata
    kind=spec['kind']
    if kind=='loft_sql_multi':
        if len(predicted)!=len(expected):return 0.0
        return sum(score_answer(p,e,{'kind':'loft_sql'}) for p,e in zip(predicted,expected))/len(expected)
    normalize=lambda v:tablebench_normalize(unicodedata.normalize('NFD',str(v)))
    if kind=='loft_retrieval':
        if spec.get('multi_turn'):
            if len(predicted)!=len(expected):return 0.0
            return sum(loft_score(p,[e],{'kind':kind}) for p,e in zip(predicted,expected))/len(expected)
        gold=[str(x[0] if isinstance(x,list) else x).strip() for x in expected]
        pred=predicted if isinstance(predicted,list) else [predicted]
        pred=[str(x[0] if isinstance(x,list) else x).strip() for x in pred[:5]]
        return len(set(pred)&set(gold))/len(gold) if gold else 0.0
    def single(pred,gold):
        if not isinstance(pred,list):pred=[pred]
        if not isinstance(gold,list):gold=[gold]
        if not pred:return 0.0
        if kind=='loft_set_em':return float(set(map(normalize,pred))==set(map(normalize,gold)))
        return float(normalize(pred[0]) in set(map(normalize,gold)))
    if spec.get('multi_turn'):
        if len(predicted)!=len(expected):return 0.0
        return sum(single(p,e) for p,e in zip(predicted,expected))/len(expected)
    return single(predicted,expected)


def lro_score(predicted,expected,kind):
    from collections import Counter
    if kind=='lro_order':
        a={canonical(v):i for i,v in enumerate(expected)};b={canonical(v):i for i,v in enumerate(predicted)}
        common=list(a.keys()&b.keys())
        if len(common)<2:return 0.0
        concordant=discordant=0
        for i,x in enumerate(common):
            for y in common[i+1:]:
                direction=(a[x]-a[y])*(b[x]-b[y])
                concordant+=direction>0;discordant+=direction<0
        total=concordant+discordant
        return round(((concordant-discordant)/total+1)/2,4) if total else .5
    # sklearn normalized_mutual_info_score average_method='arithmetic'.
    if len(predicted)!=len(expected):return 0.0
    n=len(expected)
    if not n:return 1.0
    ca,cb=Counter(map(canonical,expected)),Counter(map(canonical,predicted))
    if len(ca)==len(cb)==1:return 1.0
    joint=Counter(zip(map(canonical,expected),map(canonical,predicted)))
    mi=sum(count/n*math.log(n*count/(ca[a]*cb[b])) for (a,b),count in joint.items())
    ha=-sum(c/n*math.log(c/n) for c in ca.values());hb=-sum(c/n*math.log(c/n) for c in cb.values())
    return round(max(0,2*mi/(ha+hb)),4) if ha+hb else 1.0


def semtpch_score(predicted,expected,spec):
    """Original SemTPCH row-set/B-cubed F1 (tools/score.py), no judge."""
    from collections import Counter
    s=spec['scoring'];limit=int(s.get('limit',50));gold=expected[:limit];pred=predicted[:limit]
    def value(g,p,tol):
        try:
            gn,pn=float(g),float(p)
            if math.isnan(gn) or math.isnan(pn):return False
            return abs(gn-pn)/max(abs(gn),abs(pn),1e-12)<=tol
        except (TypeError,ValueError):return str(g or '').strip().lower()==str(p or '').strip().lower()
    if s.get('type')=='cluster_overlap':
        key=s['key_cols'][0];gc=s['gold_cluster_col'];pc=s['pred_cluster_col']
        a={str(r.get(key) or '').strip():str(r.get(gc) or '').strip() for r in gold}
        # Original oracle shortcut renames the gold cluster column for gold-as-prediction.
        b={str(r.get(key) or '').strip():str(r.get(pc,r.get(gc)) or '').strip() for r in pred}
        keys=a.keys()&b.keys()
        if not keys:return 0.0
        ga=Counter(a[k] for k in keys);pb=Counter(b[k] for k in keys);joint=Counter((a[k],b[k]) for k in keys)
        precision=sum(joint[a[k],b[k]]/pb[b[k]] for k in keys)/len(keys)
        recall=sum(joint[a[k],b[k]]/ga[a[k]] for k in keys)/len(keys)
    else:
        matched=set();count=0
        for g in gold:
            for i,p in enumerate(pred):
                if i in matched:continue
                if not all(str(g.get(k) or '').strip()==str(p.get(k) or '').strip() for k in s['key_cols']):continue
                if not all(value(g.get(v['name'],''),p.get(v['name'],''),float(v.get('tol_rel',0))) for v in s['value_cols']):continue
                count+=1;matched.add(i);break
        precision=count/len(pred) if pred else float(not gold)
        recall=count/len(gold) if gold else 1.0
    return round(2*precision*recall/(precision+recall),4) if precision+recall else 0.0
