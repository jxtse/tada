"""Supervisor-level operator jobs, per-call cancellation, budget view and observation fields."""
import json
import threading
import time

import pytest

import tada_runtime.supervisor as supervisor_module
from tada_runtime.rpc_contracts import KernelRequest, Observation
from test_cancellation import body_for, make_owner, request_for, response


def shutdown(owner):
    for job in owner.jobs.values():
        job.token.set()
    for job in owner.jobs.values():
        if job.thread:
            job.thread.join(10)
    owner.audit.close()


def tag_job_request(rows=4):
    request = request_for(rows=rows)
    return KernelRequest(op='job.submit', job_op='tag', snapshot=request.snapshot, query=request.query)


def wait_for(job_view, owner, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        view = owner.kernel_call(KernelRequest(op='job.status', job_id=job_view['job_id']))
        if view['status'] in ('done', 'failed', 'cancelled'):
            return view
        time.sleep(.05)
    raise AssertionError('job did not finish')


def slow_handler(delay):
    def handler(request):
        time.sleep(delay)
        return response(body_for(json.loads(request.content)['input']))
    return handler


def test_async_job_runs_in_slices_and_yields_the_operator_slot(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor_module, 'JOB_SLICE_SECONDS', 0.15)
    owner = make_owner(tmp_path, 1, slow_handler(0.1))
    owner.engine.progress = owner._on_progress
    try:
        view = owner.kernel_call(tag_job_request(rows=6))
        # A synchronous call submitted while the job runs gets the slot between slices.
        started = time.monotonic()
        result = owner.kernel_call(request_for(rows=1))
        assert result['kind'] == 'series' and time.monotonic() - started < 10
        final = wait_for(view, owner)
        assert final['status'] == 'done' and final['slices'] >= 2
        value = owner.kernel_call(KernelRequest(op='job.result', job_id=view['job_id']))['value']
        assert value['column']['values'] == ['Topic'] * 6
        # Resumed slices reuse cached batches: one request per row plus intents, never a full re-run.
        roles = [e.payload.role for e in owner.audit.events if e.type == 'budget/reserve']
        assert roles.count('tag-batch') == 6 + 1
    finally:
        shutdown(owner)


def test_job_cancel_stops_server_work(tmp_path):
    owner = make_owner(tmp_path, 1, slow_handler(0.2))
    try:
        view = owner.kernel_call(tag_job_request(rows=8))
        time.sleep(.3)
        owner.kernel_call(KernelRequest(op='job.cancel', job_id=view['job_id']))
        final = wait_for(view, owner)
        assert final['status'] == 'cancelled'
        dispatched = sum(e.type == 'budget/reserve' for e in owner.audit.events)
        time.sleep(.5)
        assert sum(e.type == 'budget/reserve' for e in owner.audit.events) == dispatched
    finally:
        shutdown(owner)


def test_client_call_cancel_stops_the_matching_server_call(tmp_path):
    owner = make_owner(tmp_path, 1, slow_handler(0.2))
    errors = []
    def run():
        try:
            request = request_for(rows=8).model_copy(update={'call_id': 'client-1'})
            owner.kernel_call(request)
        except Exception as error:
            errors.append(error)
    thread = threading.Thread(target=run)
    thread.start()
    time.sleep(.4)
    assert owner.kernel_call(KernelRequest(op='call.cancel', call_id='client-1')) == {'cancelled': True}
    thread.join(10)
    assert errors and type(errors[0]).__name__ == 'ExecutionCancelled'
    shutdown(owner)


def test_budget_view_warns_when_low(tmp_path):
    owner = make_owner(tmp_path, 1, slow_handler(0))
    try:
        view = owner.kernel_call(KernelRequest(op='budget'))
        assert view['left'] == view['limit'] == 40 and view['warning'] is None
        for index in range(33):
            owner.budget.reserve('agent', 'm', f'r{index}')
        assert 'requests remain' in owner.kernel_call(KernelRequest(op='budget'))['warning']
        budget, operators = owner._cell_status('fixture-cell')
        assert budget['left'] == 7 and operators == []
    finally:
        shutdown(owner)


def test_observation_carries_budget_operators_and_images():
    observation = Observation.model_validate({'execution_id': 'e', 'status': 'timeout', 'stdout': '', 'stderr': '',
        'full_output_ref': {'path': 'blobs/x', 'sha256': '0' * 64, 'size': 1},
        'budget': {'used': 1, 'limit': 2, 'left': 1, 'seconds_to_deadline': 3.0},
        'operators': [{'atom_call_id': 'a', 'op': 'tag', 'stage': 'batches', 'done': 3, 'total': 9, 'status': 'running'}],
        'images': [{'media_type': 'image/png', 'data_base64': 'AAAA'}]})
    assert observation.operators[0].done == 3 and observation.images[0].media_type == 'image/png'


def test_provider_receives_configured_concurrency(tmp_path):
    owner = make_owner(tmp_path, 2, slow_handler(0))
    try:
        assert owner.provider.concurrency == 2
    finally:
        shutdown(owner)
