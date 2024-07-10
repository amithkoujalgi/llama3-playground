import subprocess
import traceback

from pydantic.main import BaseModel
from starlette.responses import JSONResponse
from http import HTTPStatus

from typing import Any


def is_any_process_running() -> bool:
    if is_infer_process_running() or is_ocr_process_running() or is_training_process_running():
        return True
    return False


def is_training_process_running() -> bool:
    result = subprocess.run(['ps', '-ef'], stdout=subprocess.PIPE, text=True)
    if 'train.py' in result.stdout:
        return True
    else:
        return False


def is_ocr_process_running() -> bool:
    result = subprocess.run(['ps', '-ef'], stdout=subprocess.PIPE, text=True)
    if 'ocr.py' in result.stdout:
        return True
    else:
        return False


def is_infer_process_running() -> bool:
    result = subprocess.run(['ps', '-ef'], stdout=subprocess.PIPE, text=True)
    if 'infer.py' in result.stdout:
        return True
    else:
        return False


class ResponseHandler(BaseModel):
    @staticmethod
    def success(
            http_status: HTTPStatus = HTTPStatus.OK, data: Any = None, message: str = "Success"
    ):
        res = dict(
            data=data, message=message, status=True, httpStatus=http_status.phrase, httpStatusCode=http_status.value
        )

        return JSONResponse(status_code=http_status.value, content=res)

    @staticmethod
    def error(
            data: str = "Something went wrong!",
            http_status: HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR,
            exception: Exception = None,
    ):
        tb = None
        if exception is not None:
            tb = traceback.format_exc()
            print(tb)
        res = dict(
            additionalInfo=tb, data=data, status=False, httpStatus=http_status.phrase, httpStatusCode=http_status.value
        )
        return JSONResponse(status_code=http_status.value, content=res)
