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
                          'catalogue': public_metadata(adapter),
                          'input_bytes_per_record': input_size_summary(adapter)}
    return result


def validate_assignment(spec, intent):
    if any(intent.get(key) for key in ('sources', 'combination')):
        expected = assigned_sources(intent)
        if spec_sources(spec) != expected:
            raise ValueError('Use every assigned dataset and no other dataset: ' + '+'.join(expected))


INPUT_SIZES = Path(__file__).resolve().parents[1] / 'input_sizes.json'


def input_size_summary(adapter):
    """Precomputed aggregate public-input sizes (``python -m tada_taskgen.designer_sources --sizes``).
    Reading a static file keeps catalogue generation free of record or label access."""
    import json
    if not INPUT_SIZES.is_file():
        return {}
    return json.loads(INPUT_SIZES.read_text()).get('sources', {}).get(adapter.source, {})


def compute_input_sizes(sources):
    """Distribution statistics of one record's public bytes (text plus copied assets), per source."""
    import statistics
    import tomllib
    from .compiler import REPO, asset_bytes
    cap = tomllib.loads((REPO / 'v5/config.toml').read_text()).get('compiler', {}).get('workspace_max_bytes')
    out = {}
    for source in canonical_sources(sources):
        adapter = get_adapter(source)
        sizes = []
        for record in adapter.records().values():
            size = record.get('text_bytes')
            if size is None:
                size = len(str(record.get('text', '')).encode())
            try:
                size += sum(asset_bytes(a) for a in record.get('assets', []))
            except OSError:
                continue
            sizes.append(size)
        if sizes:
            sizes.sort()
            out[adapter.source] = {'records': len(sizes), 'median': int(statistics.median(sizes)),
                                   'p90': sizes[min(len(sizes) - 1, int(0.9 * len(sizes)))],
                                   'max': sizes[-1], 'workspace_max_bytes': cap}
    return out


if __name__ == '__main__':
    import json
    import sys
    if sys.argv[1:] == ['--sizes']:
        from .adapters import available_sources
        INPUT_SIZES.write_text(json.dumps({'sources': compute_input_sizes(available_sources())}, indent=1, sort_keys=True) + '\n')
        print(INPUT_SIZES)
