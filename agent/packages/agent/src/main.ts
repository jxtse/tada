import type { InitializeResult } from './generated/contracts.js';
import { parseArgs } from 'node:util';
import { createInterface } from 'node:readline/promises';
import { Supervisor } from './rpc.js';
import { createController } from './controller.js';

const {values,positionals}=parseArgs({allowPositionals:true,options:{
  config:{type:'string'},set:{type:'string',multiple:true},input:{type:'string',multiple:true},task:{type:'string'},followup:{type:'string',multiple:true},
  'compact-between':{type:'boolean'},interactive:{type:'boolean'},'session-dir':{type:'string'},
}});
const mode=positionals[0]??'run';
if(!['run','resume'].includes(mode)) throw new Error('Expected run or resume');
if(!values.task && !values.interactive) throw new Error('Provide --task or --interactive');
if(mode==='resume' && !positionals[1]) throw new Error('resume requires a session identifier');
const rpc=new Supervisor();
let controller:Awaited<ReturnType<typeof createController>>|undefined;
let initial:InitializeResult|undefined;
let cancelled=false;
const questionAbort=new AbortController();
const printReceipt=async(value:unknown)=>{ const checked=await rpc.call('receipt',value); process.stdout.write(JSON.stringify(checked)+'\n'); };
const abort=()=>{cancelled=true;questionAbort.abort();void controller?.abort().catch(error=>process.stderr.write(String(error)+'\n'));};
process.on('SIGINT',abort);
process.on('SIGTERM',abort);
try {
  const inputs=(values.input??[]).map(value=>{
    const equals=value.indexOf('=');
    if(equals<1) throw new Error('--input requires alias=PATH');
    return {name:value.slice(0,equals),path:value.slice(equals+1)};
  });
  initial=await rpc.call<InitializeResult>('initialize',{
    ...(values.config?{config_path:values.config}:{}),
    ...(mode==='resume'?{session_dir:positionals[1],resume:true}:values['session-dir']?{session_dir:values['session-dir']}:{}),inputs,overrides:values.set??[],
  });
  controller=await createController(rpc,initial);
  process.stderr.write(`TADA session ${initial.session_id}\n`);
  for(const [index,task] of [values.task,...(values.followup??[])].filter((v):v is string=>v!==undefined).entries()) {
    if(cancelled) break;
    if(index>0 && values['compact-between']) await controller.compact();
    const message=await controller.prompt(task);
    await printReceipt({session_id:initial.session_id,status:'turn_completed',message});
  }
  if(values.interactive && !cancelled) {
    const terminal=createInterface({input:process.stdin,output:process.stderr});
    try {
      while(!cancelled) {
        let text:string;
        try {
          text=await terminal.question('tada> ',{signal:questionAbort.signal});
        } catch(error) {
          if(questionAbort.signal.aborted) break;
          throw error;
        }
        if(text==='/quit'||text==='/exit') break;
        if(text==='/compact') {await controller.compact();continue;}
        if(!text.trim()) continue;
        const message=await controller.prompt(text);
        await printReceipt({session_id:initial.session_id,status:'turn_completed',message});
      }
    } finally {terminal.close();}
  }
  const inventory=await rpc.call('inventory');
  await printReceipt({session_id:initial.session_id,status:cancelled?'cancelled':'completed',inventory});
} catch(error) {
  process.stderr.write(String(error)+'\n');
  await printReceipt({session_id:initial?.session_id??null,status:cancelled?'cancelled':'failed',error:String(error)}).catch(receiptError=>process.stderr.write('Receipt unavailable: '+String(receiptError)+'\n'));
  process.exitCode=1;
} finally {
  process.removeListener('SIGINT',abort);process.removeListener('SIGTERM',abort);
  await controller?.close();
  await rpc.close().catch(error=>{process.stderr.write(String(error)+'\n');process.exitCode=1;});
}
