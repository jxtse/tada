"""Actual macOS replay boundary; skip only when nested Seatbelt is unavailable."""
import json
import subprocess
import sys

import pytest

from tada_taskgen.redteam import PYTHON, script_sandbox


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS Seatbelt replay')
def test_runtime_starts_but_private_content_and_network_remain_denied(tmp_path):
    workspace = tmp_path / 'replay'
    (workspace / 'public').mkdir(parents=True)
    (workspace / 'out').mkdir()
    private = tmp_path / 'private-fixture.txt'
    private.write_text('Not a public input')
    script = workspace / 'solve.py'
    script.write_text('''from pathlib import Path
from openpyxl import Workbook
import socket
try:
    Path(''' + repr(str(private)) + ''').read_text()
except PermissionError:
    pass
else:
    raise AssertionError('Private content became readable')
try:
    socket.create_connection(('127.0.0.1', 1), timeout=1)
except PermissionError:
    pass
else:
    raise AssertionError('Network was not denied')
book = Workbook()
book.save('out/review.xlsx')
''')
    profile = script_sandbox(workspace)
    result = subprocess.run(['/usr/bin/sandbox-exec', '-p', profile, str(PYTHON), '-I', str(script)],
                            cwd=workspace, text=True, capture_output=True, timeout=20,
                            env={'PATH': '/usr/bin:/bin', 'TMPDIR': str(workspace / 'out'),
                                 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1'})
    if 'sandbox_apply: Operation not permitted' in result.stderr:
        pytest.skip('Nested sandbox unavailable; run this test with local Seatbelt permission')
    assert result.returncode == 0, json.dumps({'exit': result.returncode, 'stderr': result.stderr})
    assert (workspace / 'out/review.xlsx').is_file()
