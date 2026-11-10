import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

// No inference server is used: this exercises cancellation of an idle real CLI/kernel.
test('idle interactive SIGINT cancels the pending question and closes the writer',async()=>{
  const root=await mkdtemp(join(tmpdir(),'tada-idle-cancel-'));
  const config=join(root,'fixture.toml');
  await writeFile(config,`[models.agent]\nbase_url="http://127.0.0.1:1/v1"\nmodel="offline-idle-fixture"\nprotocol="anthropic-messages"\nauth_mode="optional"\nauth_placeholder="fixture-no-secret"\n[models.operators]\nbase_url="http://127.0.0.1:1/v1"\nmodel="unused-offline"\nprotocol="openai-responses"\nauth_mode="optional"\n[budget]\nledger_path="${join(root,'offline-fixture-ledger.json')}"\nmax_requests=1\ndeadline_seconds=60.0\n`);
  const python=process.env.TADA_PYTHON??'.venv/bin/python';
  const child=spawn(python,['-m','tada_runtime.cli','run','--config',config,'--interactive','--session-dir',join(root,'session')],{stdio:['pipe','pipe','pipe']});
  let stdout='',stderr='';let exited=false;
  child.stdout.on('data',data=>{stdout+=data.toString();});
  child.stderr.on('data',data=>{stderr+=data.toString();});
  const exit=new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',(code,signal)=>{exited=true;resolve({code,signal});});});
  const until=async(predicate,timeout)=>{const start=Date.now();while(!predicate()){if(Date.now()-start>timeout)return false;await new Promise(resolve=>setTimeout(resolve,20));}return true;};
  try {
    assert.ok(await until(()=>stderr.includes('tada> ')||exited,15000),'CLI must reach its idle prompt');
    assert.ok(!exited,stderr);
    child.kill('SIGINT');
    assert.ok(await until(()=>exited,5000),'SIGINT must resolve idle readline without an extra newline');
    await exit;
    const receipts=stdout.trim().split('\n').map(line=>JSON.parse(line));
    assert.equal(receipts.at(-1).status,'cancelled');
    assert.equal(receipts.at(-1).error,null,'idle cancellation is a clean lifecycle outcome');
    const events=(await readFile(join(root,'session/events.jsonl'),'utf8')).trim().split('\n').map(line=>JSON.parse(line));
    assert.equal(events.at(-1).type,'session/end');
    const ledger=JSON.parse(await readFile(join(root,'offline-fixture-ledger.json'),'utf8'));
    assert.equal(ledger.reservations.length,0);
    const lockCheck=spawn(python,['-c','import fcntl,sys; f=open(sys.argv[1],"a"); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)',join(root,'session/.writer.lock')]);
    const lockExit=await new Promise(resolve=>lockCheck.once('close',resolve));
    assert.equal(lockExit,0,'writer must release its OS lock');
    console.log('Offline idle cancellation artifacts: '+root);
  } finally {
    // Permit the pre-fix implementation to shut down safely after the failing assertion.
    if(!exited){child.stdin.write('\n');child.stdin.end();await exit;}
  }
});
