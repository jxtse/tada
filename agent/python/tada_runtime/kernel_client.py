"""Kernel-side helpers. Credentials and provider dispatch stay in the supervisor."""
import json
import math
import socket
import time
import uuid
from urllib.request import Request, urlopen
import pandas as pd
from .rpc_contracts import KernelRequest, KernelResult, RpcFailure
from .state import snapshot, restore


class Job:
    """Handle for an operator submitted with tag_async()/categorize_async().

    The job runs in the supervisor in short time slices, so it never holds the operator slot for long
    and never blocks other cells. Poll status(), wait with result(timeout=...), or cancel().
    """
    def __init__(self, bridge, view):
        self._bridge, self.id, self.op = bridge, view['job_id'], view['op']

    def status(self) -> dict:
        return self._bridge.call(op='job.status', job_id=self.id)

    def done(self) -> bool:
        return self.status()['status'] in ('done', 'failed', 'cancelled')

    def cancel(self) -> dict:
        return self._bridge.call(op='job.cancel', job_id=self.id)

    def result(self, timeout: float | None = None, poll: float = 5.0):
        """Wait up to `timeout` seconds (None: until finished). Raises TimeoutError with progress when still running."""
        end = None if timeout is None else time.monotonic() + timeout
        while True:
            view = self._bridge.call(op='job.result', job_id=self.id)
            if view['status'] == 'done':
                return restore(view['value'])
            if view['status'] in ('failed', 'cancelled'):
                error = view.get('error') or {}
                raise RuntimeError(f"job {self.id} {view['status']}: {error.get('kind', '')}: {error.get('message', '')}")
            if end is not None and time.monotonic() >= end:
                raise TimeoutError(f'job {self.id} still {view["status"]}; progress={view.get("progress")}. '
                                   'Call result() again later; finished batches are kept.')
            time.sleep(poll if end is None else max(0.05, min(poll, end - time.monotonic())))

    def __repr__(self):
        return f'Job({self.id!r}, op={self.op!r})'


class KernelBridge:
    def __init__(self, url, token, rpc_timeout=1800.0):
        self.url, self.token, self.rpc_timeout = url, token, rpc_timeout

    def call(self, **kwargs):
        body = KernelRequest.model_validate(kwargs).model_dump_json().encode()
        request = Request(self.url, data=body, headers={'Authorization':f'Bearer {self.token}', 'Content-Type':'application/json'})
        from urllib.error import HTTPError, URLError
        try:
            with urlopen(request, timeout=self.rpc_timeout) as response:
                return KernelResult.model_validate_json(response.read()).value
        except HTTPError as error:
            result = RpcFailure.model_validate_json(error.read())
            raise RuntimeError(f'{result.error.kind}: {result.error.message}') from None
        except (socket.timeout, TimeoutError, URLError) as error:
            if isinstance(error, URLError) and not isinstance(error.reason, (socket.timeout, TimeoutError)):
                raise
            # Stop the server-side work this client no longer waits for, instead of leaving it to spend budget.
            if kwargs.get('call_id'):
                try:
                    self._cancel(kwargs['call_id'])
                except Exception:
                    pass
            raise TimeoutError(f'operator RPC waited {self.rpc_timeout:.0f}s; the server-side call was cancelled. '
                               'Finished batches are cached: repeat the call, or use tag_async().') from None

    def _cancel(self, call_id):
        body = KernelRequest(op='call.cancel', call_id=call_id).model_dump_json().encode()
        request = Request(self.url, data=body, headers={'Authorization':f'Bearer {self.token}', 'Content-Type':'application/json'})
        with urlopen(request, timeout=30) as response:
            response.read()

    @staticmethod
    def _tag_args(texts, query, name, fresh, cardinality, labels, multi, long_rows, row_context, images, mode, dry_run):
        from tada_operators.engine import normalize_labels
        return dict(snapshot=snapshot(texts).model_dump(mode='json'), query=query, name=name, fresh=fresh,
                    cardinality=cardinality, labels=normalize_labels(labels) if labels is not None else None, multi=multi,
                    long_rows=long_rows,
                    row_context=snapshot(row_context).model_dump(mode='json') if row_context is not None else None,
                    images=snapshot(images.map(lambda v: str(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None))
                        .model_dump(mode='json') if images is not None else None,
                    mode=mode, dry_run=dry_run)

    def tag(self, texts, query, *, name=None, fresh=False, cardinality=None, labels=None, multi=False, long_rows=None,
            row_context=None, images=None, mode='auto', dry_run=False):
        args = self._tag_args(texts, query, name, fresh, cardinality, labels, multi, long_rows, row_context, images, mode, dry_run)
        return restore(self.call(op='tag', call_id=uuid.uuid4().hex, **args))

    def tag_async(self, texts, query, *, name=None, fresh=False, cardinality=None, labels=None, multi=False, long_rows=None,
                  row_context=None, images=None, mode='auto'):
        args = self._tag_args(texts, query, name, fresh, cardinality, labels, multi, long_rows, row_context, images, mode, False)
        return Job(self, self.call(op='job.submit', job_op='tag', **args))

    def categorize(self, texts, query, *, name=None, fresh=False, cardinality=None):
        return restore(self.call(op='categorize', call_id=uuid.uuid4().hex, snapshot=snapshot(texts).model_dump(mode='json'),
                                 query=query, name=name, fresh=fresh, cardinality=cardinality))

    def categorize_async(self, texts, query, *, name=None, fresh=False, cardinality=None):
        return Job(self, self.call(op='job.submit', job_op='categorize', snapshot=snapshot(texts).model_dump(mode='json'),
                                   query=query, name=name, fresh=fresh, cardinality=cardinality))

    def jobs(self):
        return self.call(op='job.list')

    def budget(self):
        return self.call(op='budget')

    def list(self):
        return self.call(op='state.list')

    def get(self, key):
        return restore(self.call(op='state.get', key=key))

    def put(self, alias, value):
        return self.call(op='state.put', name=alias, snapshot=snapshot(value).model_dump(mode='json'))

    def inventory(self, namespace):
        rows = []
        for name, value in sorted(namespace.items()):
            if name.startswith('_') or name in {'In','Out','get_ipython','exit','quit'}:
                continue
            shape = f' shape={value.shape}' if isinstance(value,(pd.DataFrame,pd.Series)) else ''
            rows.append(f'{name}: {type(value).__name__}{shape}')
        self.call(op='inventory', namespace=rows)


def check_deliverable(path, rules=None, *, _defaults=None):
    """Check a workbook (or CSV) against the output contract before submitting it.

    Reports cells longer than max_cell_chars, non-finite numbers (NaN/inf), empty answer cells and
    answer cells that look like sentences rather than a canonical short value. Returns
    {'ok': bool, 'issues': [...]}; nothing is modified.
    """
    from pathlib import Path
    settings = {'max_cell_chars': None, 'short_answer_columns': ['Answer', 'Value'], 'short_answer_max_words': 40}
    settings.update(_defaults or {})
    settings.update(rules or {})
    max_chars = settings.get('max_cell_chars')
    short_columns = set(settings.get('short_answer_columns') or [])
    max_words = int(settings.get('short_answer_max_words') or 40)
    path = Path(path)
    if path.suffix.lower() == '.csv':
        sheets = {path.stem: pd.read_csv(path, dtype=object, keep_default_na=False)}
    else:
        sheets = pd.read_excel(path, sheet_name=None, dtype=object, keep_default_na=False)
    issues = []
    for sheet, frame in sheets.items():
        for column in frame.columns:
            for row_number, value in enumerate(frame[column].tolist(), start=2):
                where = f'{sheet}!{column} row {row_number}'
                if isinstance(value, float) and not math.isfinite(value):
                    issues.append(f'{where}: non-finite number {value!r}')
                    continue
                text = '' if value is None else str(value)
                if text.strip().lower() in ('nan', 'inf', '-inf', 'infinity', '-infinity'):
                    issues.append(f'{where}: non-finite value {text!r}')
                if max_chars and len(text) > max_chars:
                    issues.append(f'{where}: {len(text)} characters, above the {max_chars}-character cell limit')
                if column in short_columns:
                    if text.strip() == '':
                        issues.append(f'{where}: empty answer')
                    elif len(text.split()) > max_words and not text.lstrip().startswith(('[', '{')):
                        issues.append(f'{where}: {len(text.split())} words; give the canonical short answer '
                                      '(no units or explanation unless the question asks for them)')
    return {'ok': not issues, 'issues': issues[:200], 'issue_count': len(issues)}


def install(namespace, url, token, aliases, rpc_timeout=1800.0, rules_json='{}'):
    import numpy as np
    import json as _json
    bridge = KernelBridge(url, token, rpc_timeout)
    from pathlib import Path
    rules = json.loads(rules_json or '{}')
    namespace['outputs_dir'] = Path.cwd()
    namespace.update(pd=pd, np=np, json=_json, tag=bridge.tag, categorize=bridge.categorize, state=bridge,
                     tag_async=bridge.tag_async, categorize_async=bridge.categorize_async, jobs=bridge.jobs, budget=bridge.budget)
    from tada_operators.engine import validate_category_references, labels_frame
    namespace['validate_category_references'] = validate_category_references
    namespace['labels_frame'] = labels_frame
    namespace['check_deliverable'] = lambda path, overrides=None: check_deliverable(path, overrides, _defaults=rules)
    namespace['preview'] = lambda table, n=5: table.head(n)
    namespace['_tada_inventory'] = lambda: bridge.inventory(namespace)
    for alias in aliases:
        if alias.isidentifier() and not alias.startswith('_') and alias not in namespace:
            namespace[alias] = bridge.get(alias)
