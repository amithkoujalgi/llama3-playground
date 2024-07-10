import json
import logging
import os
import subprocess
import uuid
from pathlib import Path

import pydantic
from fastapi import APIRouter
from pydantic.main import BaseModel
from starlette.responses import JSONResponse

from server.config import Config
from server.routers.utils import ResponseHandler
from server.routers.utils import is_infer_process_running

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()


class InferenceWithFileContextParams(BaseModel):
    model_name: str = pydantic.Field(default=None, description="Name of the model")
    context_data_file: str = pydantic.Field(default=None, description="Path to context data file")
    question_text: str = pydantic.Field(default="Who are you?", description="Question to the LLM")


class InferenceWithOCRRunIDParams(BaseModel):
    model_name: str = pydantic.Field(default=None, description="Name of the model")
    ocr_run_id: str = pydantic.Field(default=None, description="Run ID of an OCR run")
    question_text: str = pydantic.Field(default="Who are you?", description="Question to the LLM")


class InferenceWithTextContextParams(BaseModel):
    model_name: str = pydantic.Field(default=None, description="Name of the model")
    context_data: str = pydantic.Field(default=None, description="Context data string")
    question_text: str = pydantic.Field(default="Who are you?", description="Question to the LLM")


def _run_inference_process_and_collect_result(run_id: str, model_name: str, context_data_file: str,
                                              question_text: str) -> JSONResponse:
    cmd_arr = ['python', '/app/core/infer.py', run_id, model_name, context_data_file, question_text]
    out = ""
    err = ""
    p = subprocess.Popen(cmd_arr, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in p.stdout:
        out = out + "\n" + line.decode("utf-8")
    for line in p.stderr:
        err = err + "\n" + line.decode("utf-8")
    p.wait()
    return_code = p.returncode

    if return_code == 0:
        inference_dir = f'{Config.inferences_dir}/{run_id}'
        status_file = os.path.join(inference_dir, 'RUN-STATUS')
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status = f.read()
            if 'success' in status:
                response_file = os.path.join(inference_dir, 'response.txt')
                if os.path.exists(response_file):
                    with open(response_file, 'r') as rf:
                        response = rf.read()
                        return ResponseHandler.success(
                            data={"response": response, 'run_id': run_id, 'status': 'success',
                                  'exit_code': return_code})
                else:
                    return ResponseHandler.error(data='Response file not found!')
            else:
                err_file = os.path.join(inference_dir, 'error.log')
                if os.path.exists(err_file):
                    with open(err_file, 'r') as rf:
                        err = rf.read()
                        return ResponseHandler.error(data=f'Inference failed! Reason: {err}')
                else:
                    return ResponseHandler.error(data=f'Error log file not found!')
        else:
            return ResponseHandler.error(data=f'Run status file not found!')
    else:
        return ResponseHandler.error(
            data={"response": err, 'run_id': run_id, 'status': 'failed', 'exit_code': return_code})


@router.get('/status', summary='Get status of inference',
            description='API to get status of inference. Tells you if the inference process is ongoing or not.')
async def inference_status():
    return ResponseHandler.success(data={"running": is_infer_process_running()})


@router.post('/sync/ctx-file', summary="Run inference in sync mode with context file input",
             description="API to run inference in sync mode. Does not return a response until it is obtained from the LLM.")
async def run_inference_sync_ctx_file(inference_params: InferenceWithFileContextParams):
    inference_run_id = str(uuid.uuid4())
    return _run_inference_process_and_collect_result(
        run_id=inference_run_id,
        model_name=inference_params.model_name,
        context_data_file=inference_params.context_data_file,
        question_text=inference_params.question_text
    )


@router.post('/sync/ctx-text', summary="Run inference in sync mode with context text data input",
             description="API to run inference in sync mode. Does not return a response until it is obtained from the LLM.")
async def run_inference_sync_ctx_text(inference_params: InferenceWithTextContextParams):
    inference_run_id = str(uuid.uuid4())

    ctxs_dir = os.path.join(str(Path.home()), 'temp-data', 'llm-contexts')
    os.makedirs(ctxs_dir, exist_ok=True)
    ctx_data_file = os.path.join(ctxs_dir, f'ctx-{inference_run_id}.txt')
    with open(ctx_data_file, 'w') as f:
        f.write(inference_params.context_data)

    return _run_inference_process_and_collect_result(
        run_id=inference_run_id,
        model_name=inference_params.model_name,
        context_data_file=ctx_data_file,
        question_text=inference_params.question_text
    )


@router.post('/sync/ocr-run', summary="Run inference in sync mode for an OCR run ID input",
             description="API to run inference in sync mode for an OCR run. Does not return a response until it is obtained from the LLM.")
async def run_inference_sync_ocr_run_file(inference_params: InferenceWithOCRRunIDParams):
    inference_run_id = str(uuid.uuid4())

    ocr_run_dir = f'{Config.ocr_runs_dir}/{inference_params.ocr_run_id}'
    status_file = os.path.join(ocr_run_dir, 'RUN-STATUS')

    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            status = f.read()
        if 'success' in status:
            text_result_file = os.path.join(ocr_run_dir, 'text-result.txt')
            if os.path.exists(text_result_file):
                return _run_inference_process_and_collect_result(
                    run_id=inference_run_id,
                    model_name=inference_params.model_name,
                    context_data_file=text_result_file,
                    question_text=inference_params.question_text
                )
            else:
                return ResponseHandler.error(
                    data=f'Text result file not found for OCR run ID: {inference_params.ocr_run_id}!')
        else:
            return ResponseHandler.error(data=f'OCR has failed for the run ID: {inference_params.ocr_run_id}!')
    else:
        return ResponseHandler.error(data=f'Run status file not found for OCR run ID: {inference_params.ocr_run_id}')


@router.get('/async/ctx-file', summary="Run inference in async mode",
            description="API to run inference in async mode. Returns a run ID to monitor the status with.")
async def run_inference_async(inference_params: InferenceWithFileContextParams):
    inference_run_id = str(uuid.uuid4())

    ctxs_dir = os.path.join(str(Path.home()), 'temp-data', 'llm-contexts')
    os.makedirs(ctxs_dir, exist_ok=True)
    ctx_data_file = os.path.join(ctxs_dir, f'ctx-{inference_run_id}.txt')
    with open(ctx_data_file, 'w') as f:
        f.write(inference_params.context_data)

    # _run_inference_process_and_collect_result(
    #     run_id=inference_run_id,
    #     model_name=inference_params.model_name,
    #     context_data_file=ctx_data_file,
    #     question_text=inference_params.question_text
    # )

    return ResponseHandler.success(
        data={
            'run_id': inference_run_id,
            'status': 'initiated'
        }
    )


@router.get('/models', summary="List all models", description="API to list all models")
async def list_models():
    models_dir = "/app/data/trained-models"
    try:
        return ResponseHandler.success(data=os.listdir(models_dir))
    except FileNotFoundError as e:
        return ResponseHandler.success(data=[])
