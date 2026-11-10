"""One CLI/automation entry point. Log commands are entirely offline."""
import argparse
import importlib.metadata
from pathlib import Path
from typing import Annotated, Literal
from pydantic import Field
from .contracts import StrictModel
import subprocess
import signal
import threading
import sys
from .audit import canonical, validate_session, export_html, export_jsonl
from .config import load_config

class EndpointStatus(StrictModel):
    kind: Literal['http'] = 'http'
    status: int = Field(ge=100, le=599)
    requested_model: str
    protocol: Literal['anthropic-messages','openai-responses','openai-chat']

class EndpointFailure(StrictModel):
    kind: Literal['error'] = 'error'
    error: str

class DoctorReceipt(StrictModel):
    schema_version: Literal['1.0'] = '1.0'
    configuration: Literal['valid'] = 'valid'
    inference_requests: Literal[0] = 0
    dependencies: dict[str,str]
    endpoints: dict[str,Annotated[EndpointStatus | EndpointFailure,Field(discriminator='kind')]]

def run_agent(arguments:list[str]):
    """Shared product entry point for interactive use and evaluation drivers."""
    root=Path(__file__).resolve().parents[2]
    entry=root/'packages/agent/dist/main.js'
    if not entry.exists():raise RuntimeError('Pi controller not built; run npm ci && npm run build')
    import os
    env=dict(os.environ,TADA_PYTHON=sys.executable)
    # Isolate child signal delivery: Node translates cancellation into RPC before closing
    # the supervisor. A terminal process-group SIGINT must never kill the writer first.
    child=subprocess.Popen(['node',str(entry),*arguments],env=env,start_new_session=True)
    previous={}
    def forward(signum, _frame):
        if child.poll() is None:
            child.send_signal(signum)
    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT,signal.SIGTERM):
                previous[signum]=signal.signal(signum,forward)
        return child.wait()
    finally:
        for signum,handler in previous.items():
            signal.signal(signum,handler)

def session_path(value):
    path=Path(value)
    return path if path.exists() else Path('.tada/sessions')/value

def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ('run','resume'):
        try:return sys.exit(run_agent(argv))
        except Exception as error:print(f'TADA: {error}',file=sys.stderr);return sys.exit(1)
    parser=argparse.ArgumentParser(prog='tada')
    sub=parser.add_subparsers(dest='command',required=True)
    doctor=sub.add_parser('doctor');doctor.add_argument('--config',required=True);doctor.add_argument('--probe-models',action='store_true')
    logs=sub.add_parser('log').add_subparsers(dest='action',required=True)
    for name in ('validate','view','export'):
        item=logs.add_parser(name);item.add_argument('session');item.add_argument('--output');item.add_argument('--format',choices=['html','jsonl'],default='html')
    args=parser.parse_args(argv)
    try:
        if args.command=='doctor':
            config=load_config(args.config)
            dependencies={}
            endpoints={}
            for name in ('pydantic','pandas','numpy','jupyter-client','ipykernel'):
                try:dependencies[name]=importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:dependencies[name]='missing'
            import httpx
            for name,provider in (('agent',config.models.agent),('operators',config.models.operators)):
                try:
                    key=provider.api_key()
                    headers={'Authorization':f'Bearer {key}'} if key else {}
                    response=httpx.get(provider.base_url.rstrip('/')+'/models',headers=headers,timeout=5)
                    endpoints[name]=EndpointStatus(status=response.status_code,requested_model=provider.model,protocol=provider.protocol)
                except Exception as error:endpoints[name]=EndpointFailure(error=type(error).__name__)
            results=DoctorReceipt(dependencies=dependencies,endpoints=endpoints)
            print(results.model_dump_json())
            if args.probe_models:
                config.require_budget()
                return sys.exit(run_agent(['run','--config',args.config,'--task','Run a small route probe: use execute_python to create pd.Series(["A good service"], index=["probe1"]), call tag with query "Return positive or negative sentiment" and print the result. Then execute another Python cell verifying the saved Series still exists. Do not invoke any other semantic calls.']))
            return sys.exit(1 if 'missing' in results.dependencies.values() else 0)
        path=session_path(args.session)
        if args.action=='validate':print(validate_session(path).model_dump_json())
        else:
            output=Path(args.output) if args.output else path/'views'/'session.html'
            if args.format=='html':export_html(path,output)
            else:export_jsonl(path,output)
            print(output)
    except Exception as error:
        print(f'TADA: {error}',file=sys.stderr);return sys.exit(1)

if __name__=='__main__':main()
