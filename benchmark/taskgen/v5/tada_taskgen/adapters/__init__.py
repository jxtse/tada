"""Source adapter registry. Sources are opened only when queried."""
from .base import SourceAdapter
from .maud import MAUDAdapter
from .cuad import CUADAdapter
from .cuadx import CUADXAdapter
from .nda import NDAAdapter
from .prec import PRECAdapter
from .biomedical import BAdapter


_ADAPTERS = {'MAUD': MAUDAdapter, 'CUAD': CUADAdapter, 'CUADX': CUADXAdapter,
             'NDA': NDAAdapter, 'PREC': PRECAdapter, 'B': BAdapter}
AVAILABLE_SOURCES = tuple(sorted(_ADAPTERS))
REFUSED_SOURCES = {'NTSB_FINDINGS': (
    'sources/ntsb_sources.py marks the coded findings, their descriptions and identity as private. '
    'The original C/F expert labels attach to those finding units, but there is no public, '
    'label-independent candidate finding inventory. Rendering the finding units leaks private '
    'labels; replacing them with event/category aggregates or absent-finding negatives changes '
    'the original labels and is not bijective. Public factual narratives alone do not repair this contract.')}


def canonical_source(source):
    if not isinstance(source, str):
        raise ValueError('Source must be a dataset ID string')
    if source.upper() == 'NTSB':
        return 'ntsb_aviation_avall'
    from ..batch1 import IDS, LEGACY
    if source.lower() in IDS:
        return LEGACY.get(source.lower(), source.lower())
    source = source.upper()
    return 'CUAD' if source == 'C2' else source


def available_sources():
    """Canonical dataset IDs whose public and original-label contracts are supported."""
    from .batch1_registry import available
    return available()


def get_adapter(source, **kwargs):
    source = canonical_source(source)
    if source in REFUSED_SOURCES:
        raise ValueError(f'Unsupported source: {source}. {REFUSED_SOURCES[source]}')
    cls = _ADAPTERS.get(source)
    if cls is None:
        from .batch1_registry import make_adapter
        return make_adapter(source, **kwargs)
    return cls(**kwargs)


__all__ = ['SourceAdapter', 'MAUDAdapter', 'CUADAdapter', 'CUADXAdapter', 'NDAAdapter',
           'PRECAdapter', 'BAdapter', 'get_adapter', 'available_sources', 'canonical_source',
           'AVAILABLE_SOURCES', 'REFUSED_SOURCES']
