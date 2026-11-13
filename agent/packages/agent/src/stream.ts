import { randomUUID } from 'node:crypto';
import type { ModelRuntime } from '@earendil-works/pi-coding-agent';
import type { Api, Model, Context, SimpleStreamOptions, AssistantMessage, AssistantMessageEventStream } from '@earendil-works/pi-ai';
import type { ProviderConfig, ToolAdmission } from './generated/contracts.js';

type StreamInput = {
  rpc: {call<T = any>(method:string,params:unknown):Promise<T>};
  runtime: Pick<ModelRuntime,'streamSimple'>;
  profile: ProviderConfig;
  apiKey: string | null;
  role: 'agent'|'compaction';
  turnId?:string;
  requested: Model<Api>;
  context:Context;
  options?:SimpleStreamOptions;
  emit:(type:string,payload:unknown,ids?:unknown)=>Promise<unknown>;
  createStream:()=>AssistantMessageEventStream;
};

export function auditedStream({rpc,runtime,profile,apiKey,role,turnId,requested,context,options,emit,createStream}: StreamInput) {
    const output=createStream();
    void (async()=>{
      let requestId:string|undefined,reportedModel:string|undefined;
      let sequence=0;
      let terminal=false;
      const stepId=randomUUID();
      const rawArguments=new Map<number,string>();
      const admissions=new Map<string,ToolAdmission>();
      const admit=async(toolCall:Extract<AssistantMessage['content'][number],{type:'toolCall'}>,index:number)=>{
        if(admissions.has(toolCall.id)) return admissions.get(toolCall.id)!;
        const prepared=await rpc.call<ToolAdmission>('prepare_tool',{
          tool_call_id:toolCall.id,name:toolCall.name,
          arguments:rawArguments.get(index)??JSON.stringify(toolCall.arguments),
          sdk_arguments_error:toolCall.argumentsError??null,
          ids:{request_id:requestId,turn_id:turnId,step_id:stepId},
        });
        admissions.set(toolCall.id,prepared);
        return prepared;
      };
      const admittedTool=(toolCall:Extract<AssistantMessage['content'][number],{type:'toolCall'}>)=>{
        const admission=admissions.get(toolCall.id);
        if(admission?.status==='accepted') return toolCall;
        return {...toolCall,argumentsError:admission?.error
          ? `${admission.error.kind}: ${admission.error.message} ${admission.error.details?.join('; ')??''}`
          : 'Tool call lacks original-argument admission'};
      };
      try {
        await emit('step/start',{status:'running'},{step_id:stepId,turn_id:turnId});
        const source=runtime.streamSimple(requested,context,{
          ...options,apiKey:apiKey ?? undefined,
          maxTokens:profile.max_output_tokens,temperature:profile.temperature ?? undefined,
          reasoning:profile.reasoning_effort ?? undefined,maxRetries:0,timeoutMs:(profile.timeout_seconds ?? 120)*1000,transport:'sse',
          onPayload:async payload=>{
            // Some SDKs negotiate strict-tool support with a second request even when retries are off.
            // Preserve and charge the superseded attempt instead of overwriting its reservation.
            if(requestId) await rpc.call('model_settle',{request_id:requestId,status:'unknown',provider_raw:null,error:'SDK began another HTTP attempt before a terminal response; prior dispatch remains uncertain'});
            const reservation=await rpc.call('model_reserve',{role,provider_raw:payload,...(turnId?{turn_id:turnId}:{}),step_id:stepId});
            requestId=reservation.request_id;
            return payload;
          },
          fetch:async(input,init)=>{
            const response=await globalThis.fetch(input,init);
            if(!requestId) throw new Error('HTTP response without a durable reservation');
            const wireRequestId=requestId;
            await rpc.call('model_stream',{request_id:wireRequestId,channel:'provider_raw',text:'',provider_sequence:sequence++,availability:'opaque',provider_raw:{http_status:response.status,headers:Object.fromEntries(response.headers)}});
            if(!response.body) return response;
            const reader=response.body.getReader();
            const body=new ReadableStream<Uint8Array>({
              async pull(controller) {
                try {
                  const next=await reader.read();
                  if(next.done) {
                    await rpc.call('model_stream',{request_id:wireRequestId,channel:'provider_raw',text:'',provider_sequence:sequence++,availability:'opaque',provider_raw:{transport_end:true,complete:true}});
                    controller.close();
                  } else {
                    // Await the audit barrier before the SDK consumes a byte, with one-chunk backpressure.
                    await rpc.call('model_stream',{request_id:wireRequestId,channel:'provider_raw',text:'',provider_sequence:sequence++,availability:'opaque',provider_raw:{encoding:'base64',data:Buffer.from(next.value).toString('base64'),transport_chunk:true}});
                    controller.enqueue(next.value);
                  }
                } catch(error) {
                  await reader.cancel(error).catch(()=>{});
                  controller.error(error);
                }
              },
              async cancel(reason) {await reader.cancel(reason);},
            });
            return new Response(body,{status:response.status,statusText:response.statusText,headers:response.headers});
          },
          onProviderStreamEvent:async raw=>{
            // Raw vendor objects remain opaque until the Python normalization boundary.
            const value=raw as {model?:string;message?:{model?:string};response?:{model?:string}};
            reportedModel=value.model ?? value.message?.model ?? value.response?.model ?? reportedModel;
            if(!requestId) throw new Error('Provider stream before durable reservation');
            await rpc.call('model_stream',{request_id:requestId,channel:'provider_raw',text:'',provider_sequence:sequence++,availability:'opaque',provider_raw:raw});
          },
        });
        for await(const event of source) {
          if(event.type==='text_delta'||event.type==='thinking_delta'||event.type==='toolcall_delta') {
            if(!requestId) throw new Error('Normalized stream before reservation');
            const channel=event.type==='thinking_delta'?'reasoning':event.type==='toolcall_delta'?'tool_call':'text';
            if(event.type==='toolcall_delta') rawArguments.set(event.contentIndex,(rawArguments.get(event.contentIndex)??'')+event.delta);
            await rpc.call('model_stream',{request_id:requestId,channel,text:event.delta,provider_sequence:sequence++,availability:channel==='reasoning'?'exposed':'not_provided',block_index:event.contentIndex});
          }
          if(event.type==='toolcall_end') {
            await admit(event.toolCall,event.contentIndex);
          }
          if(event.type==='done'||event.type==='error') {
            terminal=true;
            const message=event.type==='done'?event.message:event.error;
            if(event.type==='done') {
              for(const [index,block] of message.content.entries()) {
                if(block.type==='toolCall') await admit(block,index);
              }
            }
            const status=event.type==='error'?(message.stopReason==='aborted'?'cancelled':'unknown'):(message.content.length?'success':'empty');
            if(requestId) await rpc.call('model_settle',{request_id:requestId,status,provider_raw:message,...(reportedModel?{reported_model:reportedModel}:{})});
            await emit('step/end',{status},{step_id:stepId,turn_id:turnId});
          }
          // Preserve the unmodified provider response above; only the Pi-facing projection gets
          // a rejection marker, which Pi checks before its coercing parameter validator.
          if(event.type==='done') output.push({...event,message:{...event.message,content:event.message.content.map(block=>block.type==='toolCall'?admittedTool(block):block)}});
          else if(event.type==='toolcall_end') output.push({...event,toolCall:admittedTool(event.toolCall)});
          else output.push(event);
        }
        if(!terminal) throw new Error('Provider stream ended without a terminal event');
        output.end();
      } catch(error) {
        const message=error instanceof Error?error.message:String(error);
        if(requestId) await rpc.call('model_settle',{request_id:requestId,status:'unknown',provider_raw:null,error:message}).catch(()=>{});
        await emit('step/end',{status:'failed',reason:message},{step_id:stepId,turn_id:turnId}).catch(()=>{});
        const failed:AssistantMessage={role:'assistant',content:[],api:requested.api,provider:requested.provider,model:requested.id,usage:{input:0,output:0,cacheRead:0,cacheWrite:0,totalTokens:0,cost:{input:0,output:0,cacheRead:0,cacheWrite:0,total:0}},stopReason:'error',errorMessage:message,timestamp:Date.now()};
        output.push({type:'error',reason:'error',error:failed}); output.end();
      }
    })();
    return output;
  }
