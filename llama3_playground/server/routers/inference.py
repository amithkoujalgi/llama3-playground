import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path
from threading import Thread

import pydantic
from fastapi import APIRouter, UploadFile, File, Depends
from pydantic.main import BaseModel
from starlette.responses import JSONResponse

from llama3_playground.core.config import Config
from llama3_playground.core.utils import ModelManager
from llama3_playground.server.routers.utils import ResponseHandler
from llama3_playground.server.routers.utils import is_infer_process_running

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()


class InferenceWithFileUploadContextParams(BaseModel):
    model_name: str = pydantic.Field(default=ModelManager.get_latest_model(lora_adapters_only=True),
                                     description="Name of the model")
    question_text: str = pydantic.Field(default="Who are you?", description="Question to the LLM")
    prompt_text: str = pydantic.Field(
        default='You are a helpful assistant. You answer the questions precisely and concisely and if you do not know the answer you promptly say that you do not know. You do not respond with any extra text apart from what has been asked.',
        description="Custom prompt text for the model")
    max_new_tokens: int = pydantic.Field(default=128, description="Max new tokens to generate. Default is 128")
    embedding_model: str = pydantic.Field(default='Alibaba-NLP/gte-base-en-v1.5',
                                          description="Embedding model to use. Note: new embedding models would be downloaded")


class InferenceWithFileContextParams(BaseModel):
    model_name: str = pydantic.Field(default=None, description="Name of the model")
    context_data_file: str = pydantic.Field(default=None, description="Path to context data file")
    question_text: str = pydantic.Field(default="Who are you?", description="Question to the LLM")
    prompt_text: str = pydantic.Field(default='', description="Custom prompt text for the model")
    max_new_tokens: int = pydantic.Field(default=128, description="Max new tokens to generate. Default is 128")


class InferenceWithOCRRunIDParams(BaseModel):
    model_name: str = pydantic.Field(default=None, description="Name of the model")
    ocr_run_id: str = pydantic.Field(default=None, description="Run ID of an OCR run")
    question_text: str = pydantic.Field(default="Who are you?", description="Question to the LLM")
    prompt_text: str = pydantic.Field(default='', description="Custom prompt text for the model")
    max_new_tokens: int = pydantic.Field(default=128, description="Max new tokens to generate. Default is 128")


class InferenceWithTextContextParams(BaseModel):
    model_name: str = pydantic.Field(default=None, description="Name of the model")
    context_data: str = pydantic.Field(default=None, description="Context data string")
    question_text: str = pydantic.Field(default="Who are you?", description="Question to the LLM")
    prompt_text: str = pydantic.Field(default='', description="Custom prompt text for the model")
    max_new_tokens: int = pydantic.Field(default=128, description="Max new tokens to generate. Default is 128")


def _run_inference_process_and_collect_result(run_id: str, model_name: str, context_data_file: str,
                                              question_text: str, prompt_text: str,
                                              max_new_tokens: int,
                                              embedding_model: str) -> JSONResponse:
    tmp_questions_dir = os.path.join(str(Path.home()), 'temp-data', 'questions')
    os.makedirs(tmp_questions_dir, exist_ok=True)

    tmp_question_file = os.path.join(tmp_questions_dir, f'{run_id}.txt')
    with open(tmp_question_file, 'w') as f:
        f.write(question_text)

    import llama3_playground
    module_path = llama3_playground.__file__.replace('__init__.py', '')
    module_path = os.path.join(module_path, 'core', 'infer_rag.py')

    inference_dir = f'{Config.inferences_dir}/{run_id}'
    cmd_arr = [
        sys.executable, module_path,
        '-m', model_name,
        '-d', context_data_file,
        '-r', run_id,
        '-t', str(max_new_tokens),
        '-e', embedding_model,
        '-p', prompt_text,
        '-q', tmp_question_file,
    ]
    out = ""
    err = ""
    p = subprocess.Popen(cmd_arr, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in p.stdout:
        out = out + "\n" + line.decode("utf-8")
    for line in p.stderr:
        err = err + "\n" + line.decode("utf-8")
    p.wait()
    return_code = p.returncode

    with open(f"{inference_dir}/process-out.log", 'w') as lf:
        lf.write(out)
        lf.write('\n\n')
        lf.write(err)
        lf.write('\n\n')
        lf.write(f'Exit code: {return_code}')

    if return_code == 0:
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
        return ResponseHandler.error(data=f'Inference failed! Exit code: [{return_code}]. Error log: {err}')


@router.get('/status', summary='Get status of inference',
            description='API to get status of inference. Tells you if the inference process is ongoing or not.')
async def inference_status():
    return ResponseHandler.success(data={"running": is_infer_process_running()})


@router.post('/sync/with-ctx-file', summary="Run inference in sync mode with by uploading a context file",
             description="API to run inference in sync mode. Does not return a response until it is obtained from the LLM.")
async def run_inference_sync_ctx_file_upload(
        inference_params: InferenceWithFileUploadContextParams = Depends(),
        context_data_file: UploadFile = File(...)
):
    uploads_dir = os.path.join(str(Path.home()), 'temp-data', 'file-uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    uploaded_ctx_file = os.path.join(uploads_dir, context_data_file.filename)
    try:
        contents = context_data_file.file.read()
        with open(uploaded_ctx_file, 'wb') as f:
            f.write(contents)

        inference_run_id = str(uuid.uuid4())
        return _run_inference_process_and_collect_result(
            run_id=inference_run_id,
            model_name=inference_params.model_name,
            context_data_file=uploaded_ctx_file,
            question_text=inference_params.question_text,
            prompt_text=inference_params.prompt_text,
            max_new_tokens=inference_params.max_new_tokens,
            embedding_model=inference_params.embedding_model
        )
    except Exception as e:
        return ResponseHandler.error(data="Error running inference", exception=e)
    finally:
        context_data_file.file.close()


@router.post('/async/with-ctx-file', summary="Run inference in sync mode with by uploading a context file",
             description="API to run inference in sync mode. Does not return a response until it is obtained from the LLM.")
async def run_inference_sync_ctx_file_upload(
        inference_params: InferenceWithFileUploadContextParams = Depends(),
        context_data_file: UploadFile = File(...)
):
    uploads_dir = os.path.join(str(Path.home()), 'temp-data', 'file-uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    uploaded_ctx_file = os.path.join(uploads_dir, context_data_file.filename)
    try:
        contents = context_data_file.file.read()
        with open(uploaded_ctx_file, 'wb') as f:
            f.write(contents)

        inference_run_id = str(uuid.uuid4())

        thread = Thread(
            name=inference_run_id,
            target=_run_inference_process_and_collect_result,
            kwargs={
                "run_id": inference_run_id,
                "model_name": inference_params.model_name,
                "context_data_file": uploaded_ctx_file,
                "question_text": inference_params.question_text,
                "prompt_text": inference_params.prompt_text,
                "max_new_tokens": inference_params.max_new_tokens,
                "embedding_model": inference_params.embedding_model
            }
        )
        thread.start()

        return {
            "run_id": inference_run_id
        }
    except Exception as e:
        return ResponseHandler.error(data="Error running inference", exception=e)
    finally:
        context_data_file.file.close()


@router.get('/status/{run_id}', summary='Get status of inference',
            description='API to get details of inference run.')
async def get_inference_run_details(run_id: str):
    inference_run_dir = f'{Config.inferences_dir}/{run_id}'
    if os.path.exists(inference_run_dir):
        status_file = os.path.join(inference_run_dir, 'RUN-STATUS')
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status = f.read()
            if 'success' in status:
                response_file = os.path.join(inference_run_dir, 'response.txt')
                if os.path.exists(response_file):
                    with open(response_file, 'r') as rf:
                        response = rf.read()
                        return ResponseHandler.success(
                            data={"response": response, 'run_id': run_id, 'status': 'success'})
                else:
                    return ResponseHandler.error(data='Response file not found!')
            else:
                err_file = os.path.join(inference_run_dir, 'error.log')
                if os.path.exists(err_file):
                    with open(err_file, 'r') as rf:
                        err = rf.read()
                        return ResponseHandler.error(data=f'Inference failed! Reason: {err}')
                else:
                    return ResponseHandler.error(data=f'Error log file not found!')
        else:
            return ResponseHandler.success(
                data={"response": None, 'run_id': run_id, 'status': 'running'})
    else:
        return ResponseHandler.error(data=f"Couldn't get inference run details for run ID {run_id}")

# @router.get('/models', summary="List all models", description="API to list all models")
# async def list_models():
#     models_dir = "/app/data/trained-models"
#     try:
#         return ResponseHandler.success(data=os.listdir(models_dir))
#     except FileNotFoundError as e:
#         return ResponseHandler.success(data=[])

# @router.post('/sync/ctx-text', summary="Run inference in sync mode with context text data input",
#              description="API to run inference in sync mode. Does not return a response until it is obtained from the LLM.")
# async def run_inference_sync_ctx_text(inference_params: InferenceWithTextContextParams):
#     inference_run_id = str(uuid.uuid4())
#
#     ctxs_dir = os.path.join(str(Path.home()), 'temp-data', 'llm-contexts')
#     os.makedirs(ctxs_dir, exist_ok=True)
#     ctx_data_file = os.path.join(ctxs_dir, f'ctx-{inference_run_id}.txt')
#     with open(ctx_data_file, 'w') as f:
#         f.write(inference_params.context_data)
#
#     return _run_inference_process_and_collect_result(
#         run_id=inference_run_id,
#         model_name=inference_params.model_name,
#         context_data_file=ctx_data_file,
#         question_text=inference_params.question_text,
#         prompt_text=inference_params.prompt_text,
#         max_new_tokens=inference_params.max_new_tokens
#     )


# @router.post('/sync/ocr-run', summary="Run inference in sync mode for an OCR run ID input",
#              description="API to run inference in sync mode for an OCR run. Does not return a response until it is obtained from the LLM.")
# async def run_inference_sync_ocr_run_file(inference_params: InferenceWithOCRRunIDParams):
#     inference_run_id = str(uuid.uuid4())
#
#     ocr_run_dir = f'{Config.ocr_runs_dir}/{inference_params.ocr_run_id}'
#     status_file = os.path.join(ocr_run_dir, 'RUN-STATUS')
#
#     if os.path.exists(status_file):
#         with open(status_file, 'r') as f:
#             status = f.read()
#         if 'success' in status:
#             text_result_file = os.path.join(ocr_run_dir, 'text-result.txt')
#             if os.path.exists(text_result_file):
#                 return _run_inference_process_and_collect_result(
#                     run_id=inference_run_id,
#                     model_name=inference_params.model_name,
#                     context_data_file=text_result_file,
#                     question_text=inference_params.question_text,
#                     prompt_text=inference_params.prompt_text,
#                     max_new_tokens=inference_params.max_new_tokens
#                 )
#             else:
#                 return ResponseHandler.error(
#                     data=f'Text result file not found for OCR run ID: {inference_params.ocr_run_id}!')
#         else:
#             return ResponseHandler.error(data=f'OCR has failed for the run ID: {inference_params.ocr_run_id}!')
#     else:
#         return ResponseHandler.error(data=f'Run status file not found for OCR run ID: {inference_params.ocr_run_id}')
# @router.post('/sync/ctx-file-path', summary="Run inference in sync mode with context file path input",
#              description="API to run inference in sync mode. Does not return a response until it is obtained from the LLM.")
# async def run_inference_sync_ctx_file_path(inference_params: InferenceWithFileContextParams):
#     inference_run_id = str(uuid.uuid4())
#     return _run_inference_process_and_collect_result(
#         run_id=inference_run_id,
#         model_name=inference_params.model_name,
#         context_data_file=inference_params.context_data_file,
#         question_text=inference_params.question_text,
#         prompt_text=inference_params.prompt_text,
#         max_new_tokens=inference_params.max_new_tokens
#     )
