"""All agreed batch-1 IDs, with explicit unavailable adapters and paired cases.

CUADX remains a legacy read interface, not an extra batch-1 dataset. SCDB and CAP
form one inseparable label/text pair and therefore one combination-space member.
"""
import json
from ..batch1 import IDS, LEGACY, INVENTORY


TEXT = {'asap-chinese-restaurant-absa': 'ASAPAdapter', 'biodex': 'BioDEXAdapter',
        'uda-bench-vldb': 'UDAAdapter', 'ntsb_aviation_avall': 'NTSBAdapter'}


def make_adapter(source, **kwargs):
    if source == 'NTSB':
        source = 'ntsb_aviation_avall'
    if source in TEXT:
        from . import batch1_text
        return getattr(batch1_text, TEXT[source])(**kwargs)
    if source in IDS:
        from .benchmark import BenchmarkItemAdapter
        return BenchmarkItemAdapter(source, **kwargs)
    raise ValueError('Unsupported source: ' + source)


def inventory():
    from . import get_adapter
    from ..batch1 import public_metadata
    result = []
    for identifier in IDS:
        source = LEGACY.get(identifier, identifier)
        adapter = get_adapter(source)
        meta = public_metadata(adapter)
        status = getattr(adapter, 'status', 'available')
        count = meta['scorable_unit_count']
        if status == 'available' and (count is None or identifier in LEGACY):
            pool = adapter.label_pool()
            count = len({record for record, _ in pool})
            meta['scorable_cell_count'] = len(pool)
        result.append(dict(meta, dataset_id=identifier, source=source, status=status,
                           reason=getattr(adapter, 'reason', ''), scorable_unit_count=count or 0))
    return result


def available():
    # Availability is explicit source/checker support, not a model judgement.
    if INVENTORY.exists():
        return tuple(sorted({row['source'] for row in json.loads(INVENTORY.read_text())['datasets']
                             if row['status'] == 'available'}))
    from . import get_adapter
    ids = set(LEGACY.values())
    for identifier in IDS:
        if identifier not in LEGACY and getattr(get_adapter(identifier), 'status', 'available') == 'available':
            ids.add(identifier)
    return tuple(sorted(ids))
