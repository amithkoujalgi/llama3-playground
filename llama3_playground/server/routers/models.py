import copy
import json
import logging
import os

from fastapi import APIRouter

from llama3_playground.core.config import Config
from llama3_playground.core.utils import ModelManager
from llama3_playground.server.routers.utils import ResponseHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()


@router.get('/list', summary="List all trained models and their details",
            description="API to list all trained models and their details")
async def get_models_and_stats():
    trainer_runs_dir = Config.trainer_runs_dir
    try:
        model_and_stats = []
        list_of_models = ModelManager.list_trained_models(include_lora_adapters=True)
        for trainer_run in os.listdir(trainer_runs_dir):
            status_file = os.path.join(trainer_runs_dir, trainer_run, 'RUN-STATUS')
            if os.path.exists(status_file):
                with open(status_file, "r") as f:
                    run_status = f.read()
                if 'success' in run_status:
                    with open(os.path.join(trainer_runs_dir, trainer_run, 'out.json'), "r") as f:
                        run_result = json.loads(f.read())
                        run_result['run_id'] = trainer_run
                        run_result['status'] = 'success'
                        run_result['is_adapter_model_only'] = False
                        run_result['labels'] = []
                    if run_result['model_name'] in list_of_models:
                        model_and_stats.append(run_result)
                        # also include LoRA adapters
                        lora_adapter_version = copy.deepcopy(run_result)
                        lora_adapter_version[
                            'model_name'] = f"{lora_adapter_version['model_name']}{Config.LORA_ADAPTERS_SUFFIX}"
                        lora_adapter_version[
                            'model_path'] = f"{lora_adapter_version['model_path']}{Config.LORA_ADAPTERS_SUFFIX}"
                        lora_adapter_version['is_adapter_model_only'] = True
                        model_and_stats.append(lora_adapter_version)
                else:
                    err_file = os.path.join(trainer_runs_dir, trainer_run, 'error.log')
                    if os.path.exists(err_file):
                        with open(err_file, "r") as f:
                            run_err = f.read()
                    else:
                        run_err = None
                    model_and_stats.append({"run_id": trainer_run, "status": "failed", "error": run_err})
        return ResponseHandler.success(data=model_and_stats)
    except FileNotFoundError as e:
        return ResponseHandler.success(data=[])
