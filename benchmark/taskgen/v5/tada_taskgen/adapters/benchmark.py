"""Original benchmark items from the unified author reader, with explicit allowlists.

Visible: original questions, declared input tables/documents, and answer format.
Hidden: answers, reference SQL/code, generated-label markers, rationales, gold
files and evaluator metadata. Each answer is reversible canonical JSON; the
private original answer is retained unchanged. Deterministic checker sources
are enumerated in CHECKER_SOURCES. Repair reader is used only for datasets whose
frozen core lacks input-asset bindings; omitted public dev/workload files use
explicit descriptors through the same source_adapters.iter_units API.
"""
from __future__ import annotations
import ast
from local_config import public_text
from collections import Counter
import random
from .base import exclusions
from functools import cached_property
import hashlib
import json
from pathlib import Path
import zipfile
import shutil
import tempfile
from .base import REPO
from .base import SourceAdapter, QUARANTINE
from .benchmark_reader import DATA, UNIFIED, plan, units, extended_units
from ..benchmark_checkers import canonical, score_answer, sql_rows

BENCHMARK_IDS = ('oolong-synth','oolong-real','semantic-query-engines-experiments','tag-bench','bird','loft','palimpzest','longds-bench','kramabench','discoverybench','dsbench','lrobench','spider2','semtpch','dabstep','infiagent-dabench','tablebench','blade')
RAW = DATA/'raw/analytics_benchmarks'
CHECKER_SOURCES = {
 'tag-bench':'tag-bench/repo/tag/analyze.py; tag/text2sql.py: literal_eval-compatible Answer and prediction == answer',
 'bird':'longds-bench/repo/datamind/eval/sql/eval_bird.py: execute_sql result-set equality',
 'loft':'loft/repo/evaluation/{sql,rag,icl,retrieval,utils}.py: SQL execution accuracy, normalized EM, set EM and recall@5; conversations averaged by turn',
 'blade':'blade/repo/blade_bench/baselines/run_mcq.py: answer_index == options.index(correct_answer)',
 'palimpzest':'palimpzest/repo/demos/enron-demo.py: EnronValidator.map_score_fn',
 'kramabench':'kramabench/repo/benchmark/metrics.py: Success/F1; benchmark/task_fixtures/answer_type_fixtures.json',
 'lrobench':'lrobench/repo/forward_eval_interface.py: score; src/metrics/classic.py: F1, NMI, Kendall tau, list exact accuracy',
 'dabstep':'longds-bench/repo/datacope/reason_task/eval/val/dabstep_benchmark/evaluation/scorer.py: question_scorer',
 'tablebench':'https://github.com/TableBench/TableBench/blob/main/metrics/qa_metrics.py; metrics/custom_em_metric.py; eval_tablebench_script.py: Mix_Metric',
 'infiagent-dabench':'https://github.com/InfiAgent/InfiAgent/blob/main/examples/DA-Agent/eval_closed_form.py: is_equal, evaluate_responses, evaluate_accuracy_by_question',
 'dsbench':'https://github.com/LiqiangJing/DSBench: data-analysis answer verification',
 'spider2':'spider2/repo/spider2-lite/evaluation_suite/evaluate.py',
 'sembench':'sembench/repo/sembench/evaluation.py',
 'semtpch':'semtpch/repo/tools/score.py: score_one, row_match, value_match, _cluster_overlap_score',
}
UNAVAILABLE = {
 'longds-bench':('requires_llm_judge','The canonical longds/runners/src/judge.py and judge_prompt.py verify all turn answers with an LLM, including semantic choice of requested facts; no standalone deterministic verifier is supplied.'),
 'dsbench':('requires_llm_judge','Official data_analysis/compute_answer.py:evaluate_prediction sends every question, including MCQ/numeric, to GPT-4o. Data-modeling uses hidden competition targets, absent from the local public task contract. No deterministic answer verifier remains.'),
 'discoverybench':('requires_llm_judge','Original hypothesis/relationship verification requires semantic LLM matching; structured variable extraction also uses the judge (repo/discoverybench/evaluation). No original deterministic item-answer metric remains.'),
}


def asset(path, public=None, member=None):
    path=Path(path)
    if member:
        with zipfile.ZipFile(path) as z: size=z.getinfo(member).file_size
    else: size=path.stat().st_size
    result={'source_path':str(path),'path':public or (Path(member).name if member else path.name),'bytes':size}
    if member: result['member']=member
    return result


def linked_assets(unit, source):
    result=[]
    for a in unit.get('input_assets',[]):
        if a.get('status')!='available': continue
        # The repaired contracts use these exact reviewed public-input roles.
        role=a.get('source_role')
        allowed={'released local SQLite input','task_external_knowledge','task_visible_input','task_visible_media'}
        # TAG/LOFT precede role tagging; their explicit per-item input_assets
        # binding is the source contract (never a directory or archive copy).
        if role not in allowed and not (role is None and source in {'tag-bench','loft'}):continue
        result.append(asset(a['path'],Path(a.get('member') or a['path']).name,a.get('member')))
    return result


def item(uid,text,answer,kind,assets=(),**checker):
    return {'id':str(uid),'text':text,'answer':answer,'checker':{'kind':kind,**checker},'assets':list(assets)}


class LazyRecord(dict):
    def __getitem__(self,key):
        if key=='text' and self.get('_loader'):
            return self['_loader']()
        return super().__getitem__(key)


_ITEM_CACHE={}


class BenchmarkItemAdapter(SourceAdapter):
    """One original question/answer per cell; visible inputs are explicit allowlists.

    Canonical JSON encode/decode is bijective for the original answer, including
    BLADE's complete original correct-option object. Its official answer-index
    checker additionally accepts the solver's index, without exposing rationale
    or is_llm_generated. Judge-only components remain unscored metadata.
    """
    kind='benchmark_item'
    def __init__(self,source,quarantine=QUARANTINE,permutation_seed=None,workspace_max_bytes=2*1024**3,**kwargs):
        super().__init__(quarantine,permutation_seed)
        self.source=source.lower(); self.workspace_max_bytes=workspace_max_bytes
        self.dropped=Counter();self._status,self._reason=UNAVAILABLE.get(self.source,('available',''))
        self.visible_prefix=self.source
    def reference(self):
        return type(self)(self.source,quarantine=self.quarantine,permutation_seed='',workspace_max_bytes=self.workspace_max_bytes)
    @property
    def status(self):
        self._items
        return self._status
    @property
    def reason(self):
        self._items
        return self._reason
    @cached_property
    def _items(self):
        result={}
        if self._status!='available': return result
        cachekey=(self.source,self.workspace_max_bytes)
        if cachekey in _ITEM_CACHE:
            values,status,reason,dropped=_ITEM_CACHE[cachekey];self._status=status;self._reason=reason;self.dropped.update(dropped);return values
        for value in self._load():
            if value['answer'] is None or value['answer']=='': self.dropped['missing_gold']+=1;continue
            if len(self.encode('answer',value['answer']))>2000: self.dropped['answer_exceeds_2000_cell_chars']+=1;continue
            if sum(a['bytes'] for a in value['assets'])+value.get('text_bytes',len(value.get('text','').encode()))>self.workspace_max_bytes:
                self.dropped['workspace_size']+=1;continue
            if value['id'] in result: self.dropped['duplicate_original_item']+=1;continue
            if value['checker']['kind']!='sqlite_result_set' and score_answer(self.encode('answer',value['answer']),value['answer'],value['checker'])!=1.0:
                self.dropped['original_checker_gold_not_one']+=1;continue
            if value['checker']['kind']!='sqlite_result_set' and score_answer('THIS_IS_AN_INCORRECT_RESPONSE',value['answer'],value['checker'])>=1.0:
                self.dropped['original_checker_vacuous_answer']+=1;continue
            result[value['id']]=value
        encodings={}
        for value in result.values():
            encoded=self.encode('answer',value['answer']);original=canonical(value['answer'])
            if encoded in encodings and encodings[encoded]!=original:raise ValueError('Original-answer encoding is not bijective')
            encodings[encoded]=original
        if not result:
            self._status='unavailable';self._reason=self._reason or 'No original answered items with complete local inputs and a supported original deterministic checker.'
        _ITEM_CACHE[cachekey]=(result,self._status,self._reason,dict(self.dropped))
        return result
    def questions(self):
        return {'answer':{'question':'Answer the original benchmark item. Use the item-specific answer format; JSON values are accepted.','coding_conventions':['Preserve the original answer; enter JSON or the item-specific native answer syntax.'],'options':[], 'kind':self.kind}}
    @cached_property
    def _records(self):
        return {rid:LazyRecord(text=v.get('text',''),assets=v['assets'],text_bytes=v.get('text_bytes',len(v.get('text','').encode())),text_sha256=v.get('text_sha256'),_loader=v.get('text_loader')) for rid,v in self._items.items()}
    def records(self): return self._records
    def iter_record_texts(self,ids):
        if self.source.startswith('oolong-'):
            from .benchmark_oolong import selected_texts
            yield from selected_texts(self.source,ids)
        else:
            for rid in ids:yield rid,self.records()[rid]['text']
    @cached_property
    def _pool(self):
        pairs,records=exclusions(self.quarantine,self.source)
        result={(rid,'answer'):(v['answer'],[]) for rid,v in self._items.items() if rid not in records and (rid,'answer') not in pairs}
        if self.permutation_seed and len(result)>1:
            keys=sorted(result);values=[result[k] for k in keys];random.Random(self.permutation_seed+':'+self.source).shuffle(values)
            result=dict(zip(keys,values))
        return result
    def label_pool(self): return self._pool
    def encode(self,question,answer):
        if question!='answer':raise KeyError(question)
        encoded=canonical(answer)
        if self.source=='dabstep' and len(encoded)>2000 and isinstance(answer,str):
            encoded=canonical(answer.replace(', ',','))
        return encoded
    def decode(self,question,visible):
        if question!='answer':raise KeyError(question)
        if self.source=='dabstep':
            for value in self._items.values():
                if self.encode(question,value['answer'])==visible:return value['answer']
        try:return json.loads(visible)
        except (TypeError,ValueError):raise ValueError('Original benchmark answer uses canonical JSON')
    def benchmark_checker(self,record,question='answer'):
        if question!='answer':raise KeyError(question)
        return dict(self._items[record]['checker'])
    checker=benchmark_checker
    def scoring_rule(self,question):return {'name':'original_benchmark_checker','decode':'canonical_json','scope':'one original benchmark item answer'}
    def keyword_predict(self,question,text):return ''
    @property
    def limitations(self):
        result=[self.reason] if self.reason else []
        extra=self._helper_metadata().get('limitations',[])
        result.extend([extra] if isinstance(extra,str) and extra else extra if isinstance(extra,list) else [])
        notes=self._license_metadata()['licensing_notes']
        if notes:result.append(notes)
        if self.source=='longds-bench':result.append('Kaggle inputs are for local use; original redistribution restrictions apply.')
        if self.source=='dsbench':result.append('Original DSBench data are restricted to non-commercial educational and research use.')
        return result
    def _helper_metadata(self):
        if self.source in {'oolong-synth','oolong-real','semantic-query-engines-experiments'}:
            from .benchmark_oolong import metadata
            return metadata(self.source)
        if self.source=='sembench':
            from .benchmark_sembench import metadata
            return metadata(self.source)
        return {}
    def _license_metadata(self):
        profile_path=UNIFIED/'output/cards'/self.source/'profile.json'
        if not profile_path.is_file():
            from ..batch1 import INVENTORY
            row=next((r for r in json.loads(INVENTORY.read_text())['datasets'] if r['dataset_id']==self.source),{})
            return {'licensing_notes':public_text(row.get('licensing_notes','')), 'license_evidence':''}
        profile=json.loads(profile_path.read_text())
        return {'licensing_notes':public_text(profile.get('license','Not specified in original card; inspect upstream terms.')),
                'license_evidence':public_text(profile.get('license_evidence',''))}
    @property
    def metadata(self):
        return {'dataset_id':self.source,'kind':self.kind,'status':self.status,'scoring':self.status,
                'reason':self.reason,'scorable_units':len(self._items),'scorable_count':len(self._items),
                'dropped_counts':dict(self.dropped),'checker':sorted({x['checker']['kind'] for x in self._items.values()}),
                'checker_source':self._checker_source(),
                'visible_fields':self._helper_metadata().get('visible_fields',['original question','declared input assets','answer format']),
                'hidden_fields':self._helper_metadata().get('hidden_fields',['answer','reference SQL/code','gold files','rationale','is_llm_generated']),
                'summary':'Original benchmark questions with local inputs and original deterministic verification.',
                'reader':'source_adapters.iter_units; versioned source_adapters.repair_reader.iter_units for repaired asset contracts; explicit local unified descriptors for omitted dev/workload/gold files',
                **self._license_metadata(),
                'label_card':'${unified_root}/output/cards/'+self.source+'/label_card.md'}
    def _checker_source(self):
        if self.source in {'oolong-synth','oolong-real','semantic-query-engines-experiments'}:
            from .benchmark_oolong import metadata
            return metadata(self.source).get('checker_source',metadata(self.source))
        if self.source=='sembench':
            from .benchmark_sembench import metadata
            return metadata(self.source).get('checker_source','sembench/repo original evaluators')
        return CHECKER_SOURCES.get(self.source,self.reason)
    def _load(self):
        if self.source in {'oolong-synth','oolong-real','semantic-query-engines-experiments'}:
            from .benchmark_oolong import iter_items
            if self.source=='semantic-query-engines-experiments':self.dropped['unlabelled_aggregation_queries']+=6
            yield from iter_items(self.source);return
        loader=getattr(self,'_load_'+self.source.replace('-','_'),None)
        if loader is None:
            self._reason='No verified deterministic original-answer binding has been implemented for this source.';return
        yield from loader()

    def _load_tag_bench(self):
        for u in units(self.source,'34e2949f740c30dd',repair=True):
            r=u['raw_row'];a=r.get('Answer')
            if not a or r.get('Query type','').lower()=='aggregation':self.dropped['missing_aggregation_gold']+=1;continue
            try:a=ast.literal_eval(a)
            except (ValueError,SyntaxError):pass
            assets=linked_assets(u,self.source)
            if not assets:self.dropped['missing_database']+=1;continue
            yield item(u['unit_id'],r['Query']+'\nReturn the answer value (JSON scalar or list), using the supplied database.',a,'exact',assets)

    def _load_bird(self):
        databases=UNIFIED/'continuations/20261104-data-repair/implementation/assets/tag-bench'
        archive=RAW/'bird/dev.zip';nested='dev_20240627/dev_databases.zip'
        groups={}
        for d in plan(self.source):
            if d.get('role') not in ('query_spec','records'):continue
            for u in units(self.source,d['table_id']):groups.setdefault(u['raw_row']['db_id'],[]).append(u)
        temporary_root=REPO/'v5_runs/repair_r7/tmp';temporary_root.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive) as outer, outer.open(nested) as stream, zipfile.ZipFile(stream) as inner:
            for db,values in groups.items():
                existing=databases/(db+'.sqlite');member=f'dev_databases/{db}/{db}.sqlite'
                if member not in inner.namelist():self.dropped['missing_database']+=len(values);continue
                size=inner.getinfo(member).file_size
                if size>self.workspace_max_bytes:self.dropped['workspace_size']+=len(values);continue
                with tempfile.TemporaryDirectory(dir=temporary_root,prefix='bird-verify-') as temporary:
                    path=existing
                    if not path.is_file():
                        path=Path(temporary)/'database.sqlite'
                        with inner.open(member) as src,path.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
                    for u in values:
                        r=u['raw_row']
                        try:sql_rows(r['SQL'],path)
                        except Exception:self.dropped['gold_sql_not_executable']+=1;continue
                        a=asset(existing,'database.sqlite') if existing.is_file() else {'source_path':str(archive),'nested_member':nested,'member':member,'path':'database.sqlite','bytes':size}
                        yield item(u['unit_id'],r['question']+'\nEvidence: '+r.get('evidence','')+'\nReturn one read-only SQLite SELECT query.',r['SQL'],'sqlite_result_set',[a],database='database.sqlite')

    def _load_loft(self):
        for d in plan(self.source,repair=True):
            if d.get('role')!='records':continue
            for u in units(self.source,d['table_id'],repair=True):
                r=u['raw_row'];assets=linked_assets(u,self.source)
                if not assets:self.dropped['missing_inputs']+=1;continue
                path=d['relative_path'];multi=isinstance(r.get('query_text'),list)
                if 'sql/' in path or 'sql_' in path:kind='loft_sql_multi' if multi else 'loft_sql'
                elif 'retrieval/' in path or 'retrieval_' in path:kind='loft_retrieval'
                elif 'rag/nq/' in path or 'rag_nq/' in path or 'icl' in path:kind='loft_em'
                else:kind='loft_set_em'
                answer=r.get('answers',r.get('target'))
                question=r.get('query_text',r.get('input',''))
                yield item(u['unit_id'],json.dumps(question,ensure_ascii=False)+'\nReturn the original answer as JSON (one result per turn for a conversation; passage IDs for retrieval; result rows for SQL).',answer,kind,assets,multi_turn=multi)

    def _load_blade(self):
        for d in plan(self.source):
            if not d['relative_path'].endswith('/mcq_dataset.json'):continue
            parent=Path(d['path']).parent
            info_desc=next(x for x in plan(self.source) if x['path']==str(parent/'info.json'))
            info=next(units(self.source,info_desc['table_id']))['raw_row']
            context=json.dumps({k:info[k] for k in ('research_questions','data_desc','data_description') if k in info},ensure_ascii=False)
            assets=[asset(parent/'data.csv')] if (parent/'data.csv').is_file() else []
            for u in units(self.source,d['table_id']):
                r=u['raw_row'];mcqs=list(r.get('mcqs_cvar',[]))
                for vals in r.get('mcqs_transform',{}).values():mcqs.extend(vals)
                for i,m in enumerate(mcqs):
                    options=[o.get('cvar_str',o.get('code','')) for o in m['options']]
                    verb='least' if m['mc_type']=='select_neg' else 'most'
                    text=context+'\nChoose the '+verb+' justifiable option for this research question. Return its zero-based index.\n'+json.dumps(options,ensure_ascii=False)
                    yield item(u['unit_id']+':'+str(i),text,m['correct_answer'],'blade_mcq',assets,correct_index=m['options'].index(m['correct_answer']))

    def _load_dabstep(self):
        base=RAW/'dabstep/hf_adyen_DABstep/data'
        assets=[asset(p) for p in sorted((base/'context').iterdir()) if p.is_file()]
        for u in extended_units(self.source,[base/'tasks/dev.jsonl'],text_fields=['question','guidelines'],label_fields=['answer']):
            r=u['raw_row'];yield item(u['unit_id'],r['question']+'\n'+r['guidelines'],r['answer'],'dabstep',assets)

    def _load_palimpzest(self):
        labels={u['raw_row']['__native_key']:u['raw_row']['value'] for u in units(self.source,'c81a12779c2582b9')}
        for u in units(self.source,'2852e8ad80cdea45'):
            name=Path(u['source_locator']['path']).name;gold=labels.get(name)
            if not gold:self.dropped['unscored_filter_only_record']+=1;continue
            # Original map_score_fn verifies sender and subject on labelled positives.
            expected={k:gold[0][k] for k in ('sender','subject')}
            yield item(u['unit_id'],'Extract the sender email address and subject. Return a JSON object with sender and subject.\n'+u['raw_row']['text'],expected,'palimpzest_fields')

    def _load_infiagent_dabench(self):
        labels={u['raw_row']['id']:u['raw_row']['common_answers'] for u in units(self.source,'c6460363768b2537')}
        base=RAW/'infiagent-dabench/hf_infiagent_DABench/da-dev-tables'
        for u in units(self.source,'3598da22449ba421'):
            r=u['raw_row'];p=base/r['file_name']
            if not p.is_file() or r['id'] not in labels:self.dropped['missing_input_or_gold']+=1;continue
            yield item(u['unit_id'],'\n'.join(str(r[k]) for k in ('question','constraints','format')),labels[r['id']],'infiagent',[asset(p)])

    def _load_tablebench(self):
        for u in units(self.source,'c44174fde2b587ca'):
            r=u['raw_row']
            if r['qtype']=='Visualization':self.dropped['chart_artifact_not_an_answer_cell']+=1;continue
            if r['qtype'] in ('FactChecking','NumericalReasoning') or r['qsubtype']=='ImpactAnalysis':metric='em'
            elif r['qsubtype'] in ('CorrelationAnalysis','TrendForecasting','StatisticalAnalysis'):metric='tolerance_10'
            else:metric='rouge_l'
            yield item(u['unit_id'],r['question']+'\nTable: '+json.dumps(r['table'],ensure_ascii=False)+'\nReturn only the final answer.',r['answer'],'tablebench',metric=metric)

    def _load_lrobench(self):
        databases={u['raw_row']['name']:u['raw_row']['tables'] for u in units(self.source,'7b83e55dea94ef79')}
        archive=RAW/'lrobench/databases.zip'
        for d in plan(self.source):
            if d.get('role')!='query_spec':continue
            for u in units(self.source,d['table_id']):
                r=u['raw_row'];a=r.get('ground_truth')
                if isinstance(a,str):
                    try:a=ast.literal_eval(a)
                    except (SyntaxError,ValueError):pass
                if a is None:self.dropped['missing_gold']+=1;continue
                if a==[]:self.dropped['empty_gold_original_f1_zero']+=1;continue
                assets=[]
                for dbase in (r['db_used'] if isinstance(r['db_used'],list) else [r['db_used']]):
                    for table in databases.get(dbase,[]):
                        try:assets.append(asset(archive,table['path'],table['path']))
                        except KeyError:pass
                if not assets:self.dropped['missing_inputs']+=1;continue
                kind=('lro_nmi' if 'cluster_metadata' in d['relative_path'] else 'lro_order' if 'order_metadata' in d['relative_path'] else 'lro_impute' if 'impute_metadata' in d['relative_path'] else 'lro_exact_table' if 'multi_metadata' in d['relative_path'] else 'lro_set_f1')
                yield item(u['unit_id'],r['question']+'\nReturn a JSON array: cluster labels in source row order, a ranking for order questions, or result rows otherwise.',a,kind,assets)

    def _load_kramabench(self):
        base=RAW/'kramabench/repo'
        paths=[base/'workload'/(name+'.json') for name in ('archeology','astronomy','biomedical','environment','legal','wildfire')]
        for u in extended_units(self.source,paths,text_fields=['query'],label_fields=['answer','subtasks']):
            r=u['raw_row'];domain=Path(u['source_locator']['path']).stem
            types={'numeric_exact':'krama_success','string_exact':'krama_success','list_exact':'krama_f1','numeric_approximate':'krama_rae'}
            kind=types.get(r.get('answer_type'))
            if not kind:self.dropped['requires_llm_judge']+=1;continue
            assets=[];missing=False
            sources=r.get('data_sources',[])
            if isinstance(sources,str):sources=[sources]
            inputs=base/'data'/domain/'input'
            for name in sources:
                p=inputs/name
                matches=[p] if p.is_file() else list(inputs.rglob(Path(name).name))
                if len(matches)!=1 or not matches[0].is_file():missing=True;break
                assets.append(asset(matches[0],str(matches[0].relative_to(inputs))))
            if missing or not assets:self.dropped['missing_or_ambiguous_inputs']+=1;continue
            yield item(u['unit_id'],r['query']+'\nReturn the final answer as a JSON value.',r.get('answer'),kind,assets)

    def _load_spider2(self):
        base=RAW/'spider2/repo/spider2-lite/evaluation_suite/gold'
        standards={u['raw_row']['instance_id']:u['raw_row'] for u in extended_units(self.source,[base/'spider2lite_eval.jsonl'])}
        for u in units(self.source,'88ff0e753d82d11b',repair=True):
            r=u['raw_row'];uid=r['instance_id']
            if not uid.startswith('local'):self.dropped['remote_warehouse']+=1;continue
            assets=linked_assets(u,self.source)
            if not any(a['path'].endswith('.sqlite') for a in assets):self.dropped['missing_database']+=1;continue
            paths=sorted((base/'exec_result').glob(uid+'.csv')) or sorted((base/'exec_result').glob(uid+'_[a-z].csv'))
            if not paths:self.dropped['missing_gold']+=1;continue
            tables=[]
            for p in paths:
                rows=[{k:v for k,v in x['raw_row'].items() if not k.startswith('__')} for x in extended_units(self.source,[p],format='csv')]
                tables.append([list(row.values()) for row in rows])
            std=standards.get(uid,{})
            yield item(u['unit_id'],r['question']+'\nReturn the result table as a JSON array of rows, in the requested order.',{'tables':tables},'spider2_table',assets,condition_cols=std.get('condition_cols',[]),ignore_order=std.get('ignore_order',False))

    def _load_sembench(self):
        from .benchmark_sembench import iter_items
        yield from iter_items(self.source)

    def _load_semtpch(self):
        for d in plan(self.source,repair=True):
            if d.get('role')!='records':continue
            for u in units(self.source,d['table_id'],repair=True):
                r=u['raw_row'];evaluator=r['evaluator'];agent=r['agent_input']
                gold=RAW/'semtpch/repo'/evaluator['gold_answer_csv']
                if not gold.is_file():self.dropped['missing_gold']+=1;continue
                answer=[{k:v for k,v in x['raw_row'].items() if not k.startswith('__')} for x in extended_units(self.source,[gold],format='csv')]
                assets=[]
                for a in u['input_assets']:
                    if a.get('status')!='available' or a.get('source_role') not in {'task_visible_input','task_visible_media'}:continue
                    path=Path(a['path']);parts=path.parts
                    if 'build' not in parts:continue
                    public=str(Path(*parts[parts.index('build'):]))
                    if path.name=='full.csv':raise ValueError('Hidden SemTPCH table entered public asset binding')
                    assets.append(asset(path,public,a.get('member')))
                if not assets:self.dropped['missing_inputs']+=1;continue
                text=agent['nl_question']+'\nInput table: '+agent['view_csv']+'\nReturn result rows as a JSON array of objects. Media paths are relative to this input directory.'
                yield item(u['unit_id'],text,answer,'semtpch',assets,scoring=evaluator['scoring'])
