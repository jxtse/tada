"""Run the TADA task designer with pi.

Tools given to the designer: web_search (configured search pool) and write/read/list confined to the
run's task/ directory. No shell, no source-data access, no memory.

Usage:
  run_designer.py B --dry-run          # assemble prompt + audit only, no model call
  run_designer.py B [--model ollama/claude-opus-5-5] [--thinking high] [--tag note]

Output: engineering/pi/runs/<family>-<utc>/{prompt.md, pi.jsonl, pi.err, task/, trace.json, receipt.json}
Nothing is built or executed here; controller.py builds and checks later.
"""
from __future__ import annotations

import argparse, datetime, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path
from contextlib import nullcontext

HERE = Path(__file__).resolve().parent
V4 = HERE.parents[1]
EXT = HERE / 'tada_designer_tools.ts'
TOOLS = ['list_task_files', 'read_task_file', 'web_search', 'write_task_file']
REQUIRED = ['task.toml', 'instruction.md', 'design.md', 'build/build.py', 'build/sensitivity.py',
            'build/wrong_answers.py', 'build/keyword_baseline.py',
            'environment/Dockerfile',
            'tests/Dockerfile', 'tests/test.sh', 'tests/scoring.py', 'solution/solve.sh']
V5_REQUIRED = ['dependency_graph.json', 'scenario_spec.json', 'instruction.md', 'design.md']
V5_TOOLS = ['check_task_files', 'list_task_files', 'read_task_file', 'write_task_file']
# Label-free structural example for the designer prompt (validated by tests against the compiler).
V5_EXAMPLE = {
    'version': 5, 'id': 'example-scenario', 'source': 'MAUD', 'category': 'sequential_dependency',
    'questions': ['DP041', 'DP068', 'DP110', 'DP114', 'DP086'], 'sample_size': 16,
    'work_items': [
        {'id': 'scope', 'order': 1, 'title': 'Confirm the current review population', 'depends_on': []},
        {'id': 'coding', 'order': 2, 'title': 'Code the assigned terms', 'depends_on': ['scope'], 'scope_from': 'in_scope'},
        {'id': 'summary', 'order': 3, 'title': 'Summarise the coded terms', 'depends_on': ['coding']}],
    'documents': [{'id': 'cutoff_email', 'path': 'correspondence/cutoff.txt', 'kind': 'email',
                   'facts': {'meeting_date': '2024-10-02'}}],
    'dependency_graph': {
        'nodes': [
            {'id': 'in_scope', 'type': 'construction_fact', 'provision': 'in-document', 'scoring': 'prerequisite',
             'carrier': 'scope_revision', 'work_item': 'scope', 'template': 'record_scope'},
            {'id': 'term_codes', 'type': 'original_label', 'provision': 'in-document', 'scoring': 'label',
             'carrier': 'agreements/{record}.txt', 'work_item': 'coding', 'template': 'record_question'},
            {'id': 'term_counts', 'type': 'derived_quantity', 'provision': 'solver-derived',
             'scoring': 'consistency-only', 'work_item': 'summary', 'template': 'question_count'},
            {'id': 'meeting_date', 'type': 'construction_fact', 'provision': 'in-document', 'scoring': 'input',
             'carrier': 'cutoff_email', 'fact_key': 'meeting_date', 'work_item': 'summary', 'template': 'document_fact'}],
        'edges': [{'from': 'in_scope', 'to': 'term_codes'}, {'from': 'term_codes', 'to': 'term_counts'},
                  {'from': 'meeting_date', 'to': 'term_counts'}]},
    'parameters': {'scope_exclude_every': 4},
    'sheets': {'scope': 'Review_Scope', 'coding': 'Terms_Coding', 'summary': 'Terms_Summary'},
    'thresholds': {'include_all': 0.9, 'ignore_revisions': 0.9},
}
# Strings that would indicate per-record labels leaked into the prompt. (Paths such as
# /tests/gold.json are legitimate: the designer must tell the scorer where gold lives.)
FORBIDDEN_IN_PROMPT = ['"direction": -1', '"direction": 0', '"direction": 1', 'evidence_spans": [',
                       '"Label Code":', '"label": -1', '"label": 0', '"label": 1']


def two_source_example():
    """Label-free MAUD/CUAD shape; each dataset retains its original question IDs."""
    import copy
    example = copy.deepcopy(V5_EXAMPLE)
    example.pop('source')
    example['id'] = 'example-two-source'
    example['sources'] = ['CUAD', 'MAUD']
    example['questions'] = {'MAUD': example['questions'],
        'CUAD': ['change_of_control', 'exclusivity', 'termination_for_convenience']}
    nodes, edges = [], []
    for source in example['sources']:
        prefix = source.lower() + '_'
        for node in example['dependency_graph']['nodes'][:3]:
            new = dict(node, id=prefix + node['id'], source=source)
            nodes.append(new)
        edges += [{'from': prefix + 'in_scope', 'to': prefix + 'term_codes'},
                  {'from': prefix + 'term_codes', 'to': prefix + 'term_counts'}]
    example['documents'] = []
    example['work_items'][1].pop('scope_from', None)
    example['dependency_graph'] = {'nodes': nodes, 'edges': edges}
    return example


def assemble(family: str, v5: bool = False, intent: str | None = None) -> str:
    if v5:
        tpl = (HERE / 'designer_v5.md').read_text()
        sys.path.insert(0, str(HERE / 'v5'))
        from tada_taskgen.designer_sources import assigned_sources, catalogues
        reviewed_intent = json.loads(Path(intent).read_text()) if intent else {}
        sources = assigned_sources(reviewed_intent, family)
        inputs = {'combination': '+'.join(sources), 'sources': sources,
                  'source_catalogues': catalogues(sources)}
        if intent:
            inputs['intent'] = reviewed_intent
        schemas = {p.name: json.loads(p.read_text()) for p in sorted((HERE / 'v5/schemas').glob('*.json'))}
        return tpl.replace('{{INPUTS}}', json.dumps(inputs, ensure_ascii=False, indent=2)).replace(
            '{{SCHEMAS}}', json.dumps(schemas, ensure_ascii=False, indent=2)).replace(
            '{{EXAMPLE}}', json.dumps(V5_EXAMPLE, ensure_ascii=False, indent=1)).replace(
            '{{MULTISOURCE_EXAMPLE}}', json.dumps(two_source_example(), ensure_ascii=False, indent=1))
    legacy = [HERE / 'designer_pi.md', HERE / 'lessons_pi.md', HERE / 'packs' / family / 'pack.json',
              HERE / 'directions' / f'{family}.md']
    if not all(item.is_file() for item in legacy):
        raise ValueError('Legacy v4 designer inputs are not included; use --v5 with a v5 intent.')
    tpl = (HERE / 'designer_pi.md').read_text()
    lessons = (HERE / 'lessons_pi.md').read_text()
    pack = json.loads((HERE / 'packs' / family / 'pack.json').read_text())
    parts = {
        '{{LESSONS}}': lessons,
        '{{HARBOR_TEMPLATES}}': (HERE / 'harbor_templates.md').read_text(),
        '{{DIRECTION}}': (HERE / 'directions' / f'{family}.md').read_text(),
        '{{INPUTS}}': '```json\n' + json.dumps(pack, ensure_ascii=False, indent=1) + '\n```',
    }
    for k, v in parts.items():
        assert k in tpl, k
        tpl = tpl.replace(k, v)
    assert '{{' not in tpl, 'unfilled placeholder'
    return tpl


def audit_prompt(prompt: str) -> list[str]:
    return [s for s in FORBIDDEN_IN_PROMPT if s in prompt]


def parse_trace(jsonl: Path) -> dict:
    tools, usage_total, final, errors = [], {}, '', []
    stops = []
    for line in jsonl.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = d.get('type')
        if t == 'turn_end':
            stops.append((d.get('message') or {}).get('stopReason'))
        if t == 'tool_execution_start':
            args = d.get('args') or {}
            if d.get('toolName') == 'write_task_file':
                args = {'path': args.get('path'), 'bytes': len((args.get('content') or '').encode())}
            tools.append({'tool': d.get('toolName'), 'args': args})
        if t == 'tool_execution_end' and d.get('isError'):
            errors.append({'tool': d.get('toolName'), 'result': str(d.get('result'))[:300]})
        if t == 'message_end' and d.get('message', {}).get('role') == 'assistant':
            m = d['message']
            txt = ''.join(c.get('text', '') for c in m.get('content', []) if c.get('type') == 'text')
            if txt.strip():
                final = txt
            for k, v in (m.get('usage') or {}).items():
                if isinstance(v, (int, float)):
                    usage_total[k] = usage_total.get(k, 0) + v
    return {'tool_calls': tools, 'tool_errors': errors, 'usage_total': usage_total, 'final_text': final, 'turn_stop_reasons': stops}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('family')
    ap.add_argument('--model', default='ollama/claude-opus-5-5')
    ap.add_argument('--thinking', default='high')
    ap.add_argument('--timeout', type=int, default=5400)
    ap.add_argument('--tag', default='')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--max-continues', type=int, default=3)
    ap.add_argument('--revise', help='previous run dir whose task/ is copied in and revised')
    ap.add_argument('--brief', help='controller-written revision brief (markdown; no record-level answers)')
    ap.add_argument('--v5', action='store_true', help='Write graph, scenario, instruction and design only')
    ap.add_argument('--intent', help='Label-free reviewed v5 intent JSON')
    ap.add_argument('--out', type=Path, help='Fresh run directory (v5 must be under v5_runs)')
    a = ap.parse_args()
    if a.v5 and a.family == 'CUAD':
        a.family = 'C2'  # The CUAD adapter's existing public catalogue is packs/C2.

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    required = V5_REQUIRED if a.v5 else REQUIRED
    tools = V5_TOOLS if a.v5 else TOOLS
    run = (a.out or HERE / ('v5_runs' if a.v5 else 'runs') / f'{a.family}-{stamp}').resolve()
    if a.v5 and not run.is_relative_to((HERE / 'v5_runs').resolve()):
        ap.error('v5 output must be a fresh directory under v5_runs')
    if run.exists():
        raise FileExistsError('preserve existing designer output; choose a fresh --out')
    (run / 'task').mkdir(parents=True)
    prompt = assemble(a.family, a.v5, a.intent)
    if a.revise:
        import shutil
        prev = Path(a.revise).resolve()
        assert json.loads((prev / 'receipt.json').read_text())['family'] == a.family, 'family mismatch'
        shutil.rmtree(run / 'task')
        shutil.copytree(prev / 'task', run / 'task')
        if a.v5 and any(p.is_file() and str(p.relative_to(run / 'task')) not in required
                        for p in (run / 'task').rglob('*')):
            raise ValueError('v5 revisions require four-file v5 designer output')
        brief = Path(a.brief).read_text()
        prompt += ('\n\n## Revision request\n\nYour previous package for this family is already in the task directory '
                   '(list and read it first). An independent reviewer and the controller examined it. Revise it in place: '
                   'keep what works, fix every listed defect, and rewrite design.md so it describes the revised design. '
                   'Any scoring change must be justified by a listed defect; do not change weights to move a measured '
                   'baseline. The measured results below are aggregates; you still cannot run code.\n\n' + brief)
        (run / 'revision_of.txt').write_text(str(prev) + '\n')
    elif a.v5 and a.brief:
        prompt += ('\n\n## Revision request\n\nThe previous attempt did not produce a reusable '
                   'four-file package. Address this consolidated feedback in the new files:\n\n'
                   + Path(a.brief).read_text())
    leaks = audit_prompt(prompt)
    (run / 'prompt.md').write_text(prompt)
    prompt_paths = (['designer_v5.md', 'v5/tada_taskgen/designer_sources.py', 'tada_designer_tools.ts'] if a.v5 else
                    ['designer_pi.md', 'harbor_templates.md', 'lessons_pi.md', f'directions/{a.family}.md',
                     f'packs/{a.family}/pack.json', 'tada_designer_tools.ts'])
    prompt_files = {p: hashlib.sha256((HERE / p).read_bytes()).hexdigest()[:12] for p in prompt_paths}
    receipt = {'family': a.family, 'tag': a.tag, 'model': a.model, 'thinking': a.thinking, 'prompt_chars': len(prompt),
               'revision_of': a.revise, 'brief_sha256': hashlib.sha256(Path(a.brief).read_bytes()).hexdigest() if a.brief else None,
               'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'prompt_files': prompt_files,
               'prompt_leak_hits': leaks, 'tools': tools, 'required_files': required, 'pipeline_version': 5 if a.v5 else 4, 'started_utc': stamp}
    if leaks:
        receipt['status'] = 'refused_prompt_leak'
        (run / 'receipt.json').write_text(json.dumps(receipt, indent=1))
        sys.exit(f'prompt leak check failed: {leaks}')
    if a.dry_run:
        receipt['status'] = 'dry_run'
        (run / 'receipt.json').write_text(json.dumps(receipt, indent=1))
        print(json.dumps(receipt, indent=1)); print(run)
        return

    cmd_base = ['pi', '-p', '--mode', 'json', '--session-dir', str(run / 'session'), '--no-context-files', '--no-skills',
                '--no-prompt-templates', '--no-extensions', '--no-mcp', '--no-builtin-tools',
                '-e', str(EXT), '--tools', ','.join(tools), '--model', a.model, '--thinking', a.thinking]
    env = {k: v for k, v in os.environ.items() if not k.startswith(('HERMES_', 'PYTHONPATH'))}
    # Isolated pi config: only the designer model, with a 128k output cap (global ~/.pi untouched).
    env.update({'TADA_TOOL_AUDIT': '1', 'TADA_TASK_ROOT': str(run / 'task'),
                'PI_CODING_AGENT_DIR': str(HERE / 'pi_agent_dir')})
    if a.v5:
        env['TADA_DESIGNER_V5'] = '1'
    t0 = time.monotonic()
    code = None
    nudges = 0
    for attempt in range(1 + a.max_continues):
        if attempt == 0:
            cmd = cmd_base + [f'@{run / "prompt.md"}',
                              'Revise the package now. Read the existing files first, then write every changed file with write_task_file.'
                              if a.revise else 'Design the task now. Write every required file with write_task_file.']
        else:
            missing_now = [p for p in required if not (run / 'task' / p).exists()]
            cmd = cmd_base + ['--continue',
                              'Your previous turn ended before the package was complete (likely the output limit). '
                              'Do not restart your reasoning. Write the remaining files now with write_task_file, '
                              f'shortest path first. Still missing: {", ".join(missing_now)}.']
            nudges += 1
        with open(run / 'pi.jsonl', 'a') as out, open(run / 'pi.err', 'a') as err:
            try:
                sys.path.insert(0, str(HERE / 'v5'))
                from tada_taskgen.concurrency import model_slot, model_family
                inherited = (os.environ.get('TADA_MODEL_SLOT_HELD') == '1'
                             and os.environ.get('TADA_MODEL_SLOT_FAMILY') == model_family(a.model))
                with (nullcontext() if inherited else model_slot(a.model)):
                    proc = subprocess.run(cmd, stdout=out, stderr=err, cwd=run / 'task', env=env, timeout=a.timeout)
                code = proc.returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        if code != 0 or all((run / 'task' / p).exists() for p in required):
            break
    trace = parse_trace(run / 'pi.jsonl')
    (run / 'final.txt').write_text(trace['final_text'])
    active = re.search(r'TADA_ACTIVE_TOOLS before=(\S+) after=(\S+)', (run / 'pi.err').read_text())
    written = sorted(str(p.relative_to(run / 'task')) for p in (run / 'task').rglob('*') if p.is_file())
    missing = [p for p in required if p not in written]
    unexpected_files = [p for p in written if p not in required] if a.v5 else []
    v5_validation = None
    if a.v5 and not missing and not unexpected_files:
        try:
            sys.path.insert(0, str(HERE / 'v5'))
            from tada_taskgen.schema import load_designer_output
            from tada_taskgen.designer_sources import assigned_sources, validate_assignment
            intended = json.loads(Path(a.intent).read_text()) if a.intent else {}
            validate_assignment(load_designer_output(run / 'task'),
                                {'sources': assigned_sources(intended, a.family)})
            v5_validation = {'pass': True}
        except (ValueError, KeyError, TypeError) as exc:
            v5_validation = {'pass': False, 'error': str(exc)}
    counts = {}
    for c in trace['tool_calls']:
        counts[c['tool']] = counts.get(c['tool'], 0) + 1
    receipt.update({'exit_code': code, 'seconds': round(time.monotonic() - t0, 1), 'continue_nudges': nudges,
                    'active_tools_after': json.loads(active.group(2)) if active else None,
                    'tool_call_counts': counts, 'tool_errors': trace['tool_errors'],
                    'unexpected_tools': sorted(set(counts) - set(tools)), 'unexpected_files': unexpected_files, 'v5_validation': v5_validation,
                    'files_written': written, 'missing_required': missing, 'usage_total': trace['usage_total'],
                    'turns_hit_length_limit': sum(1 for s in trace['turn_stop_reasons'] if s == 'length'),
                    'status': 'design_written' if not missing and not unexpected_files and code == 0 and (not a.v5 or v5_validation.get('pass')) else 'incomplete'})
    (run / 'trace.json').write_text(json.dumps(trace, ensure_ascii=False, indent=1))
    (run / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=1))
    print(json.dumps({k: receipt[k] for k in ['status', 'exit_code', 'seconds', 'active_tools_after', 'tool_call_counts',
                                              'unexpected_tools', 'missing_required']}, ensure_ascii=False))
    print(run)


if __name__ == '__main__':
    main()
