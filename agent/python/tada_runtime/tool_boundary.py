"""Admission of original tool bytes, independent of any SDK parameter coercion."""
from __future__ import annotations
from typing import Literal
from pydantic import ValidationError
from .contracts import StrictModel, ExecuteArgs
from .audit_contracts import ErrorInfo
from .rpc_contracts import ToolAdmission, ToolObservationRecord


class ToolName(StrictModel):
    name: Literal['execute_python']


class ToolBoundary:
    def __init__(self, owner):
        self.owner = owner
        self.admissions = {}
        self.used = set()
        self.observations = {}

    def prepare(self, request):
        if request.tool_call_id in self.admissions:
            raise ValueError('duplicate tool call identity; no implicit re-execution')
        error = None
        code = None
        try:
            ToolName.model_validate({'name': request.name})
            arguments = ExecuteArgs.model_validate_json(request.arguments)
            code = arguments.code
        except ValidationError as failure:
            error = self.owner.error(failure)
        if error is None and request.sdk_arguments_error:
            error = ErrorInfo(kind='SDKArgumentsError', message=request.sdk_arguments_error)
        admission = ToolAdmission(tool_call_id=request.tool_call_id,
                                  status='rejected' if error else 'accepted',
                                  error=error, code=None if error else code)
        self.admissions[request.tool_call_id] = admission
        ids=request.ids.model_dump(exclude_none=True)
        ids['tool_call_id']=request.tool_call_id
        self.owner.audit.append('tool/call', {'name':request.name,'arguments':request.arguments,
            'code':admission.code,'error':error.model_dump(mode='json') if error else None},
            **ids)
        return admission.model_dump(mode='json')

    def execute(self, request):
        admission = self.admissions.get(request.tool_call_id)
        if admission is None or admission.status != 'accepted' or request.tool_call_id in self.used:
            raise ValueError('execution requires one unused strictly validated original tool call')
        self.used.add(request.tool_call_id)
        # Never consume SDK params: only the original Pydantic-validated code is executable.
        observation=self.owner.kernel.execute(admission.code)
        self.observations[request.tool_call_id]=observation
        return observation.model_dump(mode='json')

    def observe(self, request):
        admission = self.admissions.get(request.tool_call_id)
        record = ToolObservationRecord(tool_call_id=request.tool_call_id,
            is_error=request.is_error, stage='execution' if request.tool_call_id in self.used else 'pre_execution',
            error=admission.error if admission else ErrorInfo(kind='UnadmittedTool',message='SDK rejected an unadmitted tool call'),
            provider_raw=request.provider_raw)
        ref = self.owner.audit.blob(record.model_dump(mode='json'))
        observation=self.observations.get(request.tool_call_id)
        self.owner.audit.append('tool/result',{'status':'error' if request.is_error else 'ok',
            'observation_ref':ref.model_dump(),
            'full_output_ref':observation.full_output_ref.model_dump() if observation else None,
            'truncated':observation.truncated if observation else False},
            tool_call_id=request.tool_call_id,execution_id=observation.execution_id if observation else None,source='agent')
        return {'status':'recorded'}
