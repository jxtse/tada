import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Supervisor } from '../dist/rpc.js';
import { createController } from '../dist/controller.js';

// The real pinned Pi AgentSession, Anthropic adapter and Python/Jupyter are exercised.
// Only the remote provider is replaced by this local deterministic HTTP fixture.
test('original tool arguments pass Python authority before Pi coercion or kernel dispatch',async t=>{
  const root=await mkdtemp(join(tmpdir(),'tada-pi-raw-'));
  console.log('Offline Pi tool fixture artifacts: '+root);
  let raw='';let nextTool=true;let callId='';let httpCalls=0;const observations=[];
  const server=createServer(async(req,res)=>{
    let requestBody='';for await(const part of req)requestBody+=part;
    const request=JSON.parse(requestBody);httpCalls++;
    assert.equal(request.model,'offline-tool-fixture');
    assert.deepEqual(request.tools.map(tool=>tool.name),['execute_python']);
    const emit=(event,data)=>res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
    res.writeHead(200,{'Content-Type':'text/event-stream'});
    emit('message_start',{type:'message_start',message:{id:'fixture-'+httpCalls,type:'message',role:'assistant',model:'offline-tool-fixture',content:[],stop_reason:null,stop_sequence:null,usage:{input_tokens:10,output_tokens:0}}});
    if(nextTool){
      nextTool=false;
      emit('content_block_start',{type:'content_block_start',index:0,content_block:{type:'tool_use',id:callId,name:'execute_python',input:{}}});
      emit('content_block_delta',{type:'content_block_delta',index:0,delta:{type:'input_json_delta',partial_json:raw}});
    }else{
      observations.push(request.messages.at(-1));
      emit('content_block_start',{type:'content_block_start',index:0,content_block:{type:'text',text:''}});
      emit('content_block_delta',{type:'content_block_delta',index:0,delta:{type:'text_delta',text:'Fixture inspected the tool result.'}});
    }
    emit('content_block_stop',{type:'content_block_stop',index:0});
    emit('message_delta',{type:'message_delta',delta:{stop_reason:nextTool?'end_turn':httpCalls%2===1?'tool_use':'end_turn',stop_sequence:null},usage:{output_tokens:10}});
    emit('message_stop',{type:'message_stop'});res.end();
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const config=join(root,'fixture.toml');
  await writeFile(config,`[models.agent]\nbase_url="http://127.0.0.1:${server.address().port}/v1"\nmodel="offline-tool-fixture"\nprotocol="anthropic-messages"\nauth_mode="optional"\nauth_placeholder="fixture-no-secret"\n[models.operators]\nbase_url="http://127.0.0.1:1/v1"\nmodel="offline-operator-unused"\nprotocol="openai-responses"\nauth_mode="optional"\n[budget]\nledger_path="${join(root,'offline-fixture-ledger.json')}"\nmax_requests=30\ndeadline_seconds=120.0\n`);
  const rpc=new Supervisor();let controller;
  try {
    const initial=await rpc.call('initialize',{config_path:config,session_dir:join(root,'session')});
    controller=await createController(rpc,initial);
    const before=async()=> (await readFile(join(root,'session/events.jsonl'),'utf8')).trim().split('\n').map(line=>JSON.parse(line));
    const cases=[['number','{"code":123}'],['boolean','{"code":true}'],['null','{"code":null}'],['array','{"code":["print(1)"]}'],['invalid_json','{"code":"print(1)"']];
    for(const [name,args] of cases) await t.test(name,async()=>{
      raw=args;nextTool=true;callId='invalid-'+name;
      const previous=await before();
      await controller.prompt('Fixture '+name+': inspect the returned tool observation and stop.');
      const events=(await before()).slice(previous.length);
      assert.equal(events.filter(e=>e.type==='cell/start').length,0,'invalid raw arguments must never dispatch a kernel cell');
      const calls=events.filter(e=>e.type==='tool/call'&&e.tool_call_id===callId);
      assert.ok(calls.some(e=>e.payload.arguments===args&&e.payload.error?.kind==='ValidationError'),'retain raw arguments and strict Pydantic ValidationError');
      const results=events.filter(e=>e.type==='tool/result'&&e.tool_call_id===callId);
      assert.equal(results.length,1,'persist exactly one actual SDK error observation');
      const observed=JSON.stringify(observations.at(-1));
      assert.match(observed,/ValidationError|Contract validation failed/);
      const artifact=JSON.parse(await readFile(join(root,'session',results[0].payload.observation_ref.path),'utf8'));
      assert.equal(artifact.is_error,true);
    });
    await t.test('valid string still reaches the persistent kernel',async()=>{
      raw='{"code":"print(6 * 7)"}';nextTool=true;callId='valid-string';
      await controller.prompt('Fixture valid string: execute and inspect the observation.');
      assert.match(JSON.stringify(observations.at(-1)),/42/);
    });
  }finally{await controller?.close();await rpc.close();await new Promise(resolve=>server.close(resolve));}
});
