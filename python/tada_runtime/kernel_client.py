"""Kernel-side helpers. Credentials and provider dispatch stay in the supervisor."""
from urllib.request import Request, urlopen
import pandas as pd
from .rpc_contracts import KernelRequest, KernelResult, RpcFailure
from .state import snapshot, restore

class KernelBridge:
    def __init__(self, url, token):
        self.url, self.token = url, token

    def call(self, **kwargs):
        body = KernelRequest.model_validate(kwargs).model_dump_json().encode()
        request = Request(self.url, data=body, headers={'Authorization':f'Bearer {self.token}', 'Content-Type':'application/json'})
        from urllib.error import HTTPError
        try:
            with urlopen(request, timeout=1800) as response:
                return KernelResult.model_validate_json(response.read()).value
        except HTTPError as error:
            result = RpcFailure.model_validate_json(error.read())
            raise RuntimeError(f'{result.error.kind}: {result.error.message}') from None

    def tag(self, texts, query, *, name=None, fresh=False, cardinality=None):
        return restore(self.call(op='tag', snapshot=snapshot(texts).model_dump(mode='json'), query=query, name=name, fresh=fresh, cardinality=cardinality))

    def categorize(self, texts, query, *, name=None, fresh=False, cardinality=None):
        return restore(self.call(op='categorize', snapshot=snapshot(texts).model_dump(mode='json'), query=query, name=name, fresh=fresh, cardinality=cardinality))

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


def install(namespace, url, token, aliases):
    import numpy as np
    import json
    bridge = KernelBridge(url, token)
    from pathlib import Path
    namespace['outputs_dir'] = Path.cwd()
    namespace.update(pd=pd, np=np, json=json, tag=bridge.tag, categorize=bridge.categorize, state=bridge)
    from tada_operators.engine import validate_category_references
    namespace['validate_category_references'] = validate_category_references
    namespace['preview'] = lambda table, n=5: table.head(n)
    namespace['_tada_inventory'] = lambda: bridge.inventory(namespace)
    for alias in aliases:
        if alias.isidentifier() and not alias.startswith('_') and alias not in namespace:
            namespace[alias] = bridge.get(alias)
