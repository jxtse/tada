import { test } from 'node:test';
import assert from 'node:assert/strict';
import { auditedStream } from '../src/stream.ts';

// A deterministic adapter fixture, not a substitute for the live Pi acceptance gate.
class FixtureStream {
  queue=[]; waiting=[]; closed=false;
  push(event){const next=this.waiting.shift();if(next)next({done:false,value:event});else this.queue.push(event);}
  end(){this.closed=true;for(const next of this.waiting.splice(0))next({done:true});}
  async *[Symbol.asyncIterator](){while(true){if(this.queue.length)yield this.queue.shift();else if(this.closed)return;else{const next=await new Promise(resolve=>this.waiting.push(resolve));if(next.done)return;yield next.value;}}}
}
const message={role:'assistant',content:[{type:'text',text:'done'}],stopReason:'stop',model:'configured-model',provider:'test',api:'anthropic-messages',timestamp:1};
const model={id:'configured-model',provider:'test',api:'anthropic-messages'};
async function fixture(events,{role='agent',rejectReserve=false,renegotiate=false,readTransport=false}={}){
  const calls=[];let dispatches=0;let reservations=0;
  const rpc={async call(method,params){calls.push({method,params});if(method==='model_reserve'){if(rejectReserve)throw Error('budget exhausted');return {request_id:'req-'+(++reservations)};}if(method==='prepare_tool')return {status:'rejected',tool_call_id:params.tool_call_id,error:{kind:'ValidationError',message:'Fixture rejected'}};return {};}};
  const runtime={streamSimple(_model,_context,options){return {[Symbol.asyncIterator]:async function*(){
    await options.onPayload({model:model.id,max_tokens:32});dispatches++;
    if(renegotiate){await options.onPayload({model:model.id,max_tokens:32,strict:false});dispatches++;}
    if(readTransport){const response=await options.fetch('https://unused.invalid/fixture');await response.arrayBuffer();}
    await options.onProviderStreamEvent({type:'message_start',message:{model:'reported-model'}});
    for(const event of events){if(event instanceof Error)throw event;yield event;}
  }};}};
  const stream=auditedStream({rpc,runtime,profile:{model:model.id,protocol:'anthropic-messages',max_output_tokens:32,timeout_seconds:10},apiKey:null,role,requested:model,context:{messages:[]},emit:async(type,payload,ids)=>{calls.push({method:'event',params:{type,payload,ids}});},createStream:()=>new FixtureStream()});
  const output=[];for await(const event of stream)output.push(event);
  return {calls,output,dispatches};
}
test('stream audit preserves raw arguments and separates reasoning, text, tool channels',async()=>{
  const result=await fixture([
    {type:'thinking_delta',delta:'visible summary',contentIndex:0},
    {type:'text_delta',delta:'visible text',contentIndex:1},
    {type:'toolcall_delta',delta:'{"code":',contentIndex:2},
    {type:'toolcall_end',contentIndex:2,toolCall:{id:'call-1',name:'execute_python',arguments:{}}},
    {type:'done',message},
  ]);
  assert.equal(result.dispatches,1);
  const channels=result.calls.filter(c=>c.method==='model_stream').map(c=>c.params.channel);
  assert.deepEqual(channels,['provider_raw','reasoning','text','tool_call']);
  const rawArgs=result.calls.find(c=>c.method==='prepare_tool');
  assert.equal(rawArgs.params.arguments,'{"code":');
  const settled=result.calls.find(c=>c.method==='model_settle');
  assert.equal(settled.params.reported_model,'reported-model');
  assert.equal(settled.params.status,'success');
});
test('budget refusal prevents dispatch',async()=>{
  const result=await fixture([{type:'done',message}],{rejectReserve:true});
  assert.equal(result.dispatches,0);
  assert.equal(result.output.at(-1).type,'error');
  assert.match(result.output.at(-1).error.errorMessage,/budget exhausted/);
});
test('compaction uses the same pre-dispatch reservation path',async()=>{
  const result=await fixture([{type:'done',message}],{role:'compaction'});
  assert.equal(result.calls.find(c=>c.method==='model_reserve').params.role,'compaction');
  assert.equal(result.dispatches,1);
});
test('interrupted partial stream is retained and conservatively settled unknown',async()=>{
  const result=await fixture([{type:'text_delta',delta:'partial',contentIndex:0},new Error('connection lost')]);
  assert.equal(result.calls.find(c=>c.method==='model_stream'&&c.params.channel==='text').params.text,'partial');
  assert.equal(result.calls.find(c=>c.method==='model_settle').params.status,'unknown');
  assert.equal(result.output.at(-1).type,'error');
});
test('a stream without terminal event cannot hang or masquerade as success',async()=>{
  const result=await fixture([{type:'text_delta',delta:'partial',contentIndex:0}]);
  assert.equal(result.calls.find(c=>c.method==='model_settle').params.status,'unknown');
  assert.match(result.output.at(-1).error.errorMessage,/without a terminal/);
});

test('SDK negotiation reserves every attempt and preserves the superseded request',async()=>{
  const result=await fixture([{type:'done',message}],{renegotiate:true});
  assert.equal(result.dispatches,2);
  assert.equal(result.calls.filter(c=>c.method==='model_reserve').length,2);
  const settlements=result.calls.filter(c=>c.method==='model_settle');
  assert.equal(settlements[0].params.request_id,'req-1');
  assert.equal(settlements[0].params.status,'unknown');
  assert.equal(settlements[1].params.request_id,'req-2');
  assert.equal(settlements[1].params.status,'success');
});

test('raw wire audit preserves comments, invalid UTF-8 and incomplete SSE fragments',async()=>{
  const original=globalThis.fetch;
  const pieces=[Buffer.from(': heartbeat\n\n'),Buffer.from([0xff,0xfe]),Buffer.from('data: {"unfinished":')];
  globalThis.fetch=async()=>new Response(new ReadableStream({start(controller){for(const piece of pieces)controller.enqueue(piece);controller.close();}}));
  try {
    const result=await fixture([{type:'done',message}],{readTransport:true});
    const chunks=result.calls.filter(c=>c.method==='model_stream'&&c.params.provider_raw?.transport_chunk).map(c=>Buffer.from(c.params.provider_raw.data,'base64'));
    assert.deepEqual(Buffer.concat(chunks),Buffer.concat(pieces));
    assert.ok(result.calls.some(c=>c.params.provider_raw?.transport_end));
    assert.equal(result.dispatches,1);
  } finally {globalThis.fetch=original;}
});
