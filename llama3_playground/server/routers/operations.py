import logging

from fastapi import APIRouter

from llama3_playground.server.routers.utils import ResponseHandler, is_infer_process_running, \
    is_training_process_running, is_ocr_process_running

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()


@router.get('/status', summary='Check ops status', description='API to get ops status')
async def operations_status():
    return ResponseHandler.success(
        data={
            "LLM-Inference": is_infer_process_running(),
            "LLM-Training": is_training_process_running(),
            "OCR-Inference": is_ocr_process_running(),
            "OCR-Model-Training": False,
        }
    )
