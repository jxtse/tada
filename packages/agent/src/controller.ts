import { auditedStream } from './stream.js';
import { providerBaseUrl } from './provider-url.js';
import { randomUUID } from 'node:crypto';
import { join } from 'node:path';
import { mkdir } from 'node:fs/promises';
import {
  createAgentSession, createExtensionRuntime, ModelRuntime, SessionManager, SettingsManager,
  type ResourceLoader, type ToolDefinition,
} from '@earendil-works/pi-coding-agent';
import {
  stream, streamSimple, createAssistantMessageEventStream,
  type Model, type Api,
} from '@earendil-works/pi-ai/compat';
import type { Provider, CredentialStore } from '@earendil-works/pi-ai';
import type { Supervisor } from './rpc.js';
import type { Config, InitializeResult } from './generated/contracts.js';


export function strippedLoader(systemPrompt:string):ResourceLoader {
  return {
    getExtensions:()=>({extensions:[],errors:[],runtime:createExtensionRuntime()}),
    getSkills:()=>({skills:[],diagnostics:[]}), getPrompts:()=>({prompts:[],diagnostics:[]}),
    getThemes:()=>({themes:[],diagnostics:[]}), getAgentsFiles:()=>({agentsFiles:[]}),
    getSystemPrompt:()=>systemPrompt, getSystemPromptSource:()=>undefined,
    getAppendSystemPrompt:()=>[],getAppendSystemPromptSources:()=>[],extendResources:()=>{},reload:async()=>{},
  };
}

export async function createController(rpc:Supervisor, initial:InitializeResult) {
  const config:Config = initial.config;
  const profile = config.models.agent;
  const api:Api = profile.protocol === 'openai-chat' ? 'openai-completions' : profile.protocol;
  const model:Model<Api> = {
    id:profile.model,name:profile.model,api,provider:'tada-explicit',
    baseUrl:providerBaseUrl(profile.base_url,profile.protocol),reasoning:!!profile.reasoning_effort,
    input:['text'],contextWindow:Number(process.env.TADA_AGENT_CONTEXT_WINDOW ?? 200000),maxTokens:profile.max_output_tokens ?? 4096,
    cost:{input:0,output:0,cacheRead:0,cacheWrite:0},
  };
  const credentials:CredentialStore = {read:async()=>undefined,list:async()=>[],modify:async(_,fn)=>fn(undefined),delete:async()=>{}};
  const agentDir=join(initial.session_dir,'pi-runtime');
  await mkdir(agentDir,{recursive:true,mode:0o700});
  const runtime=await ModelRuntime.create({credentials,modelsPath:null,modelsStorePath:join(agentDir,'models-cache.json'),allowModelNetwork:false,refreshOnCreate:false});
  const auth = async()=>({auth:{apiKey:initial.agent_api_key ?? undefined},source:'explicit TADA profile'});
  const provider:Provider = {
    id:model.provider,name:'Explicit TADA provider',getModels:()=>[model],
    auth:{apiKey:{name:'Configured environment or local placeholder',resolve:auth,check:async()=>({type:'api_key',source:'explicit TADA profile'})}},
    stream:(m,c,o)=>stream(m,c,o as Parameters<typeof stream>[2]),streamSimple:(m,c,o)=>streamSimple(m,c,o),
  };
  runtime.registerNativeProvider(provider);
  const manager=SessionManager.inMemory(process.cwd(),undefined,initial.restored_entries?.length ? initial.restored_entries as Parameters<typeof SessionManager.inMemory>[2] : undefined);
  if (!initial.restored_entries?.length) {
    for (const message of initial.restored_messages ?? []) manager.appendMessage(message as Parameters<typeof manager.appendMessage>[0]);
  }
  const tool:ToolDefinition = {
    name:'execute_python',label:'Execute Python',description:'Execute one Python cell in the persistent session namespace. Use tag and categorize for semantic operations.',
    parameters:initial.tool_schema as ToolDefinition['parameters'],executionMode:'sequential',
    execute:async(toolCallId,_sdkParams,signal)=>{
      const abort=()=>{ void rpc.call('cancel').catch(()=>{}); };
      signal?.addEventListener('abort',abort,{once:true});
      try {
        if(signal?.aborted) throw new Error('Execution cancelled');
        const observation=await rpc.call('execute_tool',{tool_call_id:toolCallId});
        return {content:[{type:'text',text:JSON.stringify(observation)}],details:observation,isError:observation.status!=='ok'};
      } finally { signal?.removeEventListener('abort',abort); }
    },
  };
  const prompt=`You are TADA, a trusted-local text analysis agent. Your only tool is execute_python.
Use the persistent Python namespace, retaining intermediate variables across cells and turns.
Available: pd, np, json; tag(texts: pd.Series, query: str, *, name=None, fresh=False, cardinality=None, labels=None, multi=False, long_rows=None) -> Series of strings;
categorize(texts: pd.Series, query: str, *, name=None, fresh=False, cardinality=None) -> category DataFrame indexed by cid with name and description.
For a requested category count, cardinality={"k": positive_integer, "mode": "exact" or "at_most"} explicitly constrains discovery, including joint tagging's shared domain.
When the allowed answers are known, pass labels= (a list of names, a list of {"name","description"} records, or a categorize() table). With multi=False each row gets exactly one label name;
with multi=True each row gets a JSON array string of every applicable label (possibly "[]"), so several yes/no clause or topic checks share one pass over the text. labels_frame(series, labels) expands it to a boolean table.
Long texts are handled inside tag: rows too long for one request are segmented and aggregated (labels+multi: union of segment labels; otherwise one extra reduce judgment per long row). Do not split long texts yourself.
long_rows={"aggregate": "collect"} returns a JSON array of non-empty segment results for free-text extraction; other long_rows fields are optional overrides.
Every execute_python result includes budget: requests_used, requests_remaining and seconds_remaining, shared by you and all operator calls. Plan semantic work against it.
A tag call costs at least one request per batch; a long document costs one request per batch of its segments plus one reduce request, so re-reading long documents is expensive.
Ask for every field you need from a document set in one tag call (a JSON-encoded string with several keys) and pass many rows per call instead of looping one field or one row at a time.
Write a structurally valid deliverable as soon as you have enough results, then refine it while budget remains; never let the budget run out before the required file exists.
If an operator call fails, read the reason in the error message; the runtime already retries contract violations and transient provider failures up to runtime.batch_retries times, so repeating the identical call rarely helps.
Large discovery runs its pilot, retention planner and one global reducer internally. Do not implement bootstrap in a cell or bypass a capacity refusal with hidden hierarchical pruning.
Inputs have immutable rid indices. Do not replace row identities with current positions. Prepare missing/nontext input explicitly.
Tag outputs are always strings, including structured JSON encoded inside strings. Use json.loads explicitly when needed.
Categorize discovers corpus categories; tag assigns individual rows. Include cid/name/description in assignment queries and validate references.
Reuse existing semantic columns for aggregation followups. New semantic dimensions need new operator calls.
state.list(), state.get(id_or_alias), state.put(alias,value) persist supported data. Committed states survive later cell exceptions.
Only committed state/input objects survive resume; ordinary functions and temporary variables do not. Check state and namespace before repeating work.
Use ordinary pandas to count/group/join. Store outputs under outputs_dir and report their paths. Never assert task success without checking results.
Do not run subprocess model clients or make direct inference/network calls. All model calls go through supervised tag/categorize.
If a cell errors, use its actual error and existing namespace to repair through the same loop. A budget failure is a hard stop for new semantic work.
Namespace: ${initial.namespace_summary}`;
  // Automatic threshold/overflow compaction keeps long tasks inside the context window.
  // Compaction requests go through the same audited stream (role='compaction') and the
  // kernel/state are untouched; the summary and before/after messages are logged.
  // Transient upstream failures (429/5xx/timeouts, Pi's classifier) are retried with backoff by Pi.
  // Each attempt is a separately reserved, audited and budget-counted request; failed attempts stay in the log.
  const settings=SettingsManager.inMemory({compaction:{enabled:true,reserveTokens:16384,keepRecentTokens:Number(process.env.TADA_AGENT_KEEP_RECENT_TOKENS ?? 20000)},retry:{enabled:true,maxRetries:4,baseDelayMs:5000,maxAgentDelayMs:60000},cacheWarming:'off'});
  const {session,modelFallbackMessage}=await createAgentSession({cwd:process.cwd(),agentDir,model,modelRuntime:runtime,resourceLoader:strippedLoader(prompt),tools:['execute_python'],customTools:[tool],settingsManager:settings,sessionManager:manager,thinkingLevel:(profile.reasoning_effort ?? 'off') as any});
  if(modelFallbackMessage) throw new Error(`Forbidden model fallback: ${modelFallbackMessage}`);
  session.agent.toolExecution='sequential';
  // Agent listeners are awaited; this also covers SDK rejections that never invoke execute().
  const unsubscribeToolResults=session.agent.subscribe(async event=>{
    if(event.type==='tool_execution_end') await rpc.call('tool_observation',{
      tool_call_id:event.toolCallId,name:event.toolName,is_error:event.isError,provider_raw:event.result,
    });
  });
  let role:'agent'|'compaction'='agent';
  let turnId:string|undefined;
  let manualCompaction=false;
  const emit=(type:string,payload:unknown,ids:unknown={})=>rpc.call('event',{type,payload,ids});
  // Pi awaits nothing here, so serialize audit RPCs for automatic compaction explicitly.
  let autoAudit:Promise<unknown>=Promise.resolve();
  const unsubscribeCompaction=session.subscribe(event=>{
    if(manualCompaction) return;
    if(event.type==='auto_retry_start'||event.type==='auto_retry_end'){
      autoAudit=autoAudit.then(()=>emit('agent/retry',{phase:event.type==='auto_retry_start'?'start':'end',attempt:event.attempt,
        ...(event.type==='auto_retry_start'?{max_attempts:event.maxAttempts,delay_ms:event.delayMs,error:event.errorMessage}:{success:event.success,error:event.finalError ?? null})},
        turnId?{turn_id:turnId}:{}));
      return;
    }
    if(event.type==='compaction_start'){
      role='compaction';
      autoAudit=autoAudit.then(()=>rpc.call('compaction',{phase:'start',messages:session.messages}));
    } else if(event.type==='compaction_end'){
      const ok=!!event.result && !event.aborted;
      autoAudit=autoAudit.then(()=>rpc.call('compaction',{phase:'end',messages:session.messages,
        summary:ok?`[auto:${event.reason}] ${event.result!.summary}`:`[auto:${event.reason}] ${event.errorMessage ?? (event.aborted?'aborted':'no result')}`,
        status:ok?'success':'failed'}));
      role='agent';
    }
  });
  session.agent.streamFunction=(requested,context,options)=>auditedStream({
    rpc,runtime,profile,apiKey:initial.agent_api_key,role,turnId,requested,context,options,emit,createStream:createAssistantMessageEventStream,
  });
  const checkpoint=()=>rpc.call('checkpoint',{messages:session.messages,pi_entries:[session.sessionManager.getHeader(),...session.sessionManager.getEntries()]});
  return {
    session,
    async prompt(text:string) {
      turnId=randomUUID();
      await emit('turn/start',{status:'running'},{turn_id:turnId});
      await emit('user/message',{text,origin:'human'},{turn_id:turnId});
      try {
        await session.prompt(text);
        await autoAudit;
        const last=session.messages.at(-1);
        if(last?.role==='assistant' && (last.stopReason==='error'||last.stopReason==='aborted')) throw new Error(last.errorMessage??last.stopReason);
        await emit('turn/end',{status:'completed'},{turn_id:turnId});
        await checkpoint();
        return last;
      } catch(error) {
        await autoAudit.catch(()=>{});
        await emit('turn/end',{status:'failed',reason:String(error)},{turn_id:turnId});
        await checkpoint();
        throw error;
      }
    },
    async compact() {
      manualCompaction=true;
      await rpc.call('compaction',{phase:'start',messages:session.messages});
      role='compaction';
      try {
        const result=await session.compact('Preserve all state IDs, aliases, input identities, output paths, user requirements, and completed semantic columns. The Python kernel remains alive.');
        await rpc.call('compaction',{phase:'end',messages:session.messages,summary:result.summary,status:'success'});
        await checkpoint();
        return result;
      } catch(error) {
        await rpc.call('compaction',{phase:'end',messages:session.messages,status:'failed',summary:String(error)});
        throw error;
      } finally { role='agent'; manualCompaction=false; }
    },
    async abort(){await rpc.call('cancel');await session.abort();},
    async close(){unsubscribeToolResults();unsubscribeCompaction();session.dispose();},
  };
}
