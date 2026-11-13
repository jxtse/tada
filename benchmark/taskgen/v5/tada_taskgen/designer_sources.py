"""Label-free, fixed dataset-combination inputs for the scenario designer."""
import importlib.util
from pathlib import Path
import sys

from .adapters import get_adapter


def canonical_sources(values):
    if isinstance(values, str):
        values = values.split('+')
    return sorted({get_adapter(value).source for value in values})


def spec_sources(spec):
    return canonical_sources(spec.get('sources') or [spec['source']])


def assigned_sources(intent, fallback=None):
    return canonical_sources(intent.get('sources') or intent.get('combination') or
                             intent.get('source') or fallback)


def catalogues(sources):
    # The small v5 test/runtime environment deliberately reuses the provisioned
    # source reader dependencies. No install or global environment mutation.
    if importlib.util.find_spec('pandas') is None:
        from local_config import path as local_path
        runtime = local_path('datasets_python').parent.parent / 'lib' / (
            f'python{sys.version_info.major}.{sys.version_info.minor}') / 'site-packages'
        if runtime.is_dir() and str(runtime) not in sys.path:
            sys.path.append(str(runtime))
    result = {}
    for source in canonical_sources(sources):
        adapter = get_adapter(source)
        questions = {}
        from .adapters.base import disabled_questions
        disabled = disabled_questions(adapter.quarantine, adapter.source)
        for identifier, metadata in adapter.questions().items():
            if identifier in disabled:
                continue
            questions[identifier] = {key: metadata[key] for key in
                ('question', 'options', 'coding_conventions', 'provision') if key in metadata}
            if getattr(adapter, 'kind', 'record_label') != 'benchmark_item':
                questions[identifier]['encoding'] = [[option, adapter.encode(identifier, option)]
                                                      for option in metadata['options']]
        from .batch1 import public_metadata
        result[source] = {'source': source, 'questions': questions,
                          'label_caveats': list(getattr(adapter, 'limitations', ())),
                          'catalogue': public_metadata(adapter)}
    return result


def validate_assignment(spec, intent):
    if any(intent.get(key) for key in ('sources', 'combination')):
        expected = assigned_sources(intent)
        if spec_sources(spec) != expected:
            raise ValueError('Use every assigned dataset and no other dataset: ' + '+'.join(expected))
