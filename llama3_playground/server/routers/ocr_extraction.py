import json
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path
from threading import Thread

from fastapi import APIRouter, UploadFile, File
from starlette.responses import JSONResponse

from llama3_playground.core.config import Config
from llama3_playground.server.routers.utils import ResponseHandler
from llama3_playground.server.routers.utils import is_ocr_process_running

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()


def _run_ocr_process_and_collect_result(run_id: str, pdf_file: str) -> JSONResponse:
    import llama3_playground
    module_path = llama3_playground.__file__.replace('__init__.py', '')
    module_path = os.path.join(module_path, 'core', 'ocr.py')

    ocr_run_dir = f'{Config.ocr_runs_dir}/{run_id}'

    cmd_arr = [
        sys.executable, module_path,
        '-r', run_id,
        '-f', pdf_file
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

    with open(f"{ocr_run_dir}/process-out.log", 'w') as lf:
        lf.write(out)
        lf.write('\n\n')
        lf.write(err)
        lf.write('\n\n')
        lf.write(f'Exit code: {return_code}')

    if return_code == 0:
        status_file = os.path.join(ocr_run_dir, 'RUN-STATUS')
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status = f.read()
            if 'success' in status:
                json_result_file = os.path.join(ocr_run_dir, 'ocr-result.json')
                text_result_file = os.path.join(ocr_run_dir, 'text-result.txt')
                json_response = ''
                text_response = ''
                if os.path.exists(json_result_file):
                    with open(json_result_file, 'r') as rf:
                        json_response = rf.read()
                        json_response = json.loads(json_response)
                else:
                    return ResponseHandler.error(data='JSON result file not found!')
                if os.path.exists(text_result_file):
                    with open(text_result_file, 'r') as rf:
                        text_response = rf.read()
                        text_response = text_response
                else:
                    return ResponseHandler.error(data='Text result file not found!')
                return ResponseHandler.success(
                    data={
                        "ocr_json_data": json_response,
                        "ocr_text_data": text_response,
                        'run_id': run_id,
                        'status': 'success',
                        'exit_code': return_code
                    }
                )
            else:
                err_file = os.path.join(ocr_run_dir, 'error.log')
                if os.path.exists(err_file):
                    with open(err_file, 'r') as rf:
                        err = rf.read()
                        return ResponseHandler.error(data=f'OCR failed for run ID: {run_id}! Reason: {err}')
                else:
                    return ResponseHandler.error(data=f'Error log file not found for run ID: {run_id}')
        else:
            return ResponseHandler.error(data=f'Run status file not found for run ID: {run_id}')
    else:
        return ResponseHandler.error(
            data=f'Failed to run OCR process for run ID: {run_id}! Process exit ccode: {return_code}.')


def _run_ocr_process_and_collect_result_yolo(run_id: str, pdf_file: str) -> JSONResponse:
    import llama3_playground
    module_path = llama3_playground.__file__.replace('__init__.py', '')
    module_path = os.path.join(module_path, 'core', 'ocr_yolo.py')

    ocr_run_dir = f'{Config.ocr_runs_dir}/{run_id}'

    cmd_arr = [
        sys.executable, module_path,
        '-r', run_id,
        '-f', pdf_file
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

    with open(f"{ocr_run_dir}/process-out.log", 'w') as lf:
        lf.write(out)
        lf.write('\n\n')
        lf.write(err)
        lf.write('\n\n')
        lf.write(f'Exit code: {return_code}')

    if return_code == 0:
        status_file = os.path.join(ocr_run_dir, 'RUN-STATUS')
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status = f.read()
            if 'success' in status:
                json_result_file = os.path.join(ocr_run_dir, 'ocr-result.json')
                text_result_file = os.path.join(ocr_run_dir, 'text-result.txt')
                json_response = ''
                text_response = ''
                if os.path.exists(json_result_file):
                    with open(json_result_file, 'r') as rf:
                        json_response = rf.read()
                        json_response = json.loads(json_response)
                else:
                    return ResponseHandler.error(data='JSON result file not found!')
                if os.path.exists(text_result_file):
                    with open(text_result_file, 'r') as rf:
                        text_response = rf.read()
                        text_response = text_response
                else:
                    return ResponseHandler.error(data='Text result file not found!')
                return ResponseHandler.success(
                    data={
                        "ocr_json_data": json_response,
                        "ocr_text_data": text_response,
                        'run_id': run_id,
                        'status': 'success',
                        'exit_code': return_code
                    }
                )
            else:
                err_file = os.path.join(ocr_run_dir, 'error.log')
                if os.path.exists(err_file):
                    with open(err_file, 'r') as rf:
                        err = rf.read()
                        return ResponseHandler.error(data=f'OCR failed for run ID: {run_id}! Reason: {err}')
                else:
                    return ResponseHandler.error(data=f'Error log file not found for run ID: {run_id}')
        else:
            return ResponseHandler.error(data=f'Run status file not found for run ID: {run_id}')
    else:
        return ResponseHandler.error(
            data=f'Failed to run OCR process for run ID: {run_id}! Process exit ccode: {return_code}.')


@router.get('/status', summary='Get status of OCR',
            description='API to get status of OCR. Tells you if the OCR process is ongoing or not.')
async def ocr_status():
    return ResponseHandler.success(data={"running": is_ocr_process_running()})


@router.post('/sync/pdf/old', summary="Run OCR in sync mode on a PDF file", deprecated=True,
             description="API to run OCR in sync mode on a PDF file and return a response. Does not return a response until OCR process is completed.")
async def run_ocr_sync_pdf(file: UploadFile = File(...)):
    ocr_run_id = str(uuid.uuid4())
    uploads_dir = os.path.join(str(Path.home()), 'temp-data', 'file-uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    upload_file = os.path.join(uploads_dir, file.filename)
    try:
        contents = file.file.read()
        with open(upload_file, 'wb') as f:
            f.write(contents)
        return _run_ocr_process_and_collect_result(run_id=ocr_run_id, pdf_file=upload_file)
    except Exception as e:
        return ResponseHandler.error(data="Error running OCR", exception=e)
    finally:
        file.file.close()


@router.post('/sync/pdf',
             summary="Run OCR in sync mode on a PDF file. Uses YOLOv8 model from [ultralytics](https://github.com/ultralytics)",
             description="API to run OCR in sync mode on a PDF file and return a response. Does not return a response until OCR process is completed.")
async def run_ocr_sync_pdf_yolo(file: UploadFile = File(...)):
    ocr_run_id = str(uuid.uuid4())
    uploads_dir = os.path.join(str(Path.home()), 'temp-data', 'file-uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    upload_file = os.path.join(uploads_dir, file.filename)
    try:
        contents = file.file.read()
        with open(upload_file, 'wb') as f:
            f.write(contents)
        return _run_ocr_process_and_collect_result_yolo(run_id=ocr_run_id, pdf_file=upload_file)
    except Exception as e:
        return ResponseHandler.error(data="Error running OCR", exception=e)
    finally:
        file.file.close()


@router.post('/async/pdf',
             summary="Run OCR in async mode on a PDF file. Uses YOLOv8 model from [ultralytics](https://github.com/ultralytics)",
             description="API to run OCR in async mode on a PDF file")
async def run_ocr_async_pdf_yolo(file: UploadFile = File(...)):
    ocr_run_id = str(uuid.uuid4())
    uploads_dir = os.path.join(str(Path.home()), 'temp-data', 'file-uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    upload_file = os.path.join(uploads_dir, file.filename)
    try:
        contents = file.file.read()
        with open(upload_file, 'wb') as f:
            f.write(contents)

        thread = Thread(
            name=ocr_run_id,
            target=_run_ocr_process_and_collect_result_yolo,
            kwargs={
                "run_id": ocr_run_id,
                "pdf_file": upload_file
            }
        )
        thread.start()

        return {
            "run_id": ocr_run_id
        }
    except Exception as e:
        return ResponseHandler.error(data="Error running OCR", exception=e)
    finally:
        file.file.close()


@router.get('/async/pdf/status/{run_id}', summary='Get status of OCR run',
            description='API to get details of OCR run.')
async def get_ocr_yolo_run_details(run_id: str):
    ocr_run_dir = f'{Config.ocr_runs_dir}/{run_id}'
    if os.path.exists(ocr_run_dir):
        status_file = os.path.join(ocr_run_dir, 'RUN-STATUS')
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status = f.read()
            if 'success' in status:
                text_result_file = os.path.join(ocr_run_dir, 'text-result.txt')
                ocr_result_file = os.path.join(ocr_run_dir, 'ocr-result.json')
                if os.path.exists(text_result_file) and os.path.exists(ocr_result_file):
                    with open(text_result_file, 'r') as rf:
                        text_result = rf.read()
                    with open(ocr_result_file, 'r') as rf:
                        ocr_result = rf.read()
                        ocr_result_dict = json.loads(ocr_result)
                    return ResponseHandler.success(
                        data={"text_result": text_result, "ocr_result": ocr_result_dict, 'run_id': run_id,
                              'status': 'success'})
                else:
                    return ResponseHandler.error(data='Result file/s not found!')
            else:
                err_file = os.path.join(ocr_run_dir, 'error.log')
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
