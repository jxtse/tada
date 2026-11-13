"""The CLI wrapper owns terminal signals; supervisor must not receive them directly."""
import os,signal,subprocess,sys,time
from pathlib import Path

def test_terminal_interrupt_is_forwarded_to_controller_only(tmp_path):
    binpath=tmp_path/'bin';binpath.mkdir();ready=tmp_path/'ready';interrupted=tmp_path/'interrupted'
    fixture=binpath/'node'
    fixture.write_text('#!'+sys.executable+'\nimport os,signal,time\nfrom pathlib import Path\n'
      'def stop(*args):\n Path(os.environ["INTERRUPTED"]).write_text("handled")\n raise SystemExit(0)\n'
      'signal.signal(signal.SIGINT,stop)\nPath(os.environ["READY"]).write_text("ready")\nwhile True:time.sleep(.05)\n')
    fixture.chmod(0o700)
    env={**os.environ,'PATH':str(binpath)+os.pathsep+os.environ['PATH'],'READY':str(ready),'INTERRUPTED':str(interrupted)}
    child=subprocess.Popen([sys.executable,'-m','tada_runtime.cli','run','--task','offline signal fixture'],env=env,start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        deadline=time.monotonic()+10
        while not ready.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert ready.exists()
        os.killpg(child.pid,signal.SIGINT)
        out,err=child.communicate(timeout=10)
        assert child.returncode==0,err
        assert interrupted.read_text()=='handled'
        assert 'KeyboardInterrupt' not in err
    finally:
        if child.poll() is None:child.kill();child.wait()
