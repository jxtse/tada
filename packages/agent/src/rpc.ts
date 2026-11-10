import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { createInterface } from 'node:readline';
import { randomUUID } from 'node:crypto';

/** JSON decoding only: the Python endpoint validates every request and response. */
export class Supervisor {
  private child: ChildProcessWithoutNullStreams;
  private pending = new Map<string, {resolve:(value:any)=>void; reject:(error:Error)=>void}>();
  constructor(python = process.env.TADA_PYTHON ?? '.venv/bin/python') {
    this.child = spawn(python, ['-m', 'tada_runtime.supervisor'], {stdio:['pipe','pipe','pipe'], env:process.env});
    this.child.stderr.on('data', data => process.stderr.write(data));
    createInterface({input:this.child.stdout}).on('line', line => {
      try {
        const reply = JSON.parse(line);
        const waiter = this.pending.get(reply.id);
        if (!waiter) throw new Error('Unexpected supervisor response ID');
        this.pending.delete(reply.id);
        if (reply.error) waiter.reject(new Error(JSON.stringify(reply.error)));
        else waiter.resolve(reply.result);
      } catch (error) { this.fail(error instanceof Error ? error : new Error(String(error))); }
    });
    this.child.on('error', error => this.fail(error));
    this.child.on('exit', (code, signal) => this.fail(new Error(`Supervisor exited: ${code ?? signal}`)));
  }
  private fail(error:Error) { for (const p of this.pending.values()) p.reject(error); this.pending.clear(); }
  call<T = any>(method:string, params:unknown = {}):Promise<T> {
    const id = randomUUID();
    return new Promise((resolve,reject) => {
      this.pending.set(id,{resolve,reject});
      this.child.stdin.write(JSON.stringify({schema_version:'1.0',id,method,params})+'\n', error => {
        if (error) { this.pending.delete(id); reject(error); }
      });
    });
  }
  async close() { try { await this.call('close'); } finally { this.child.stdin.end(); } }
}
