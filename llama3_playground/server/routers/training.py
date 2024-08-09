import logging

from fastapi import APIRouter

from llama3_playground.server.routers.utils import ResponseHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()


@router.get('/llm/run', summary='Start training LLM',
            description='API to start training LLM')
async def llm_training_start():
    return ResponseHandler.success(data={"running": True})


@router.get('/llm/details', summary='Get details of LLM training run',
            description='API to get details of LLM training run')
async def llm_training_details():
    return ResponseHandler.success(data={"running": True})


@router.get('/ocr/run', summary='Start training OCR model',
            description='API to start training OCR model')
async def ocr_training_start():
    return ResponseHandler.success(data={"running": True})


@router.get('/ocr/details', summary='Get details of OCR training run',
            description='API to get details of OCR training run')
async def ocr_training_details():
    return ResponseHandler.success(data={"running": True})

# @router.get('/status', summary='Get status of training',
#             description='API to get status of training. Tells you if the training process is ongoing or not.')
# async def training_status():
#     return ResponseHandler.success(data={"running": is_training_process_running()})

# @router.get('/model-stats', summary="Get all trained models and their stats",
#             description="API to get all trained models and their stats")
# async def get_models_and_stats():
#     trainer_runs_dir = "/app/data/trainer-runs"
#     try:
#         model_and_stats = []
#         # for trainer_run in os.listdir(trainer_runs_dir):
#         #     status_file = os.path.join(trainer_runs_dir, trainer_run, 'RUN-STATUS')
#         #     if os.path.exists(status_file):
#         #         with open(status_file, "r") as f:
#         #             run_status = f.read()
#         #         if 'success' in run_status:
#         #             with open(os.path.join(trainer_runs_dir, trainer_run, 'out.json'), "r") as f:
#         #                 run_result = json.loads(f.read())
#         #                 run_result['run_id'] = trainer_run
#         #                 run_result['status'] = 'success'
#         #             model_and_stats.append(run_result)
#         #         else:
#         #             err_file = os.path.join(trainer_runs_dir, trainer_run, 'error.log')
#         #             if os.path.exists(err_file):
#         #                 with open(err_file, "r") as f:
#         #                     run_err = f.read()
#         #             else:
#         #                 run_err = None
#         #             model_and_stats.append({"run_id": trainer_run, "status": "failed", "error": run_err})
#         #     # else:
#         #     #     model_and_stats.append({"run_id": trainer_run, "status": "running"})
#
#         list_of_models = ModelManager.list_trained_models(include_lora_adapters=True)
#         for trainer_run in os.listdir(trainer_runs_dir):
#             status_file = os.path.join(trainer_runs_dir, trainer_run, 'RUN-STATUS')
#             if os.path.exists(status_file):
#                 with open(status_file, "r") as f:
#                     run_status = f.read()
#                 if 'success' in run_status:
#                     with open(os.path.join(trainer_runs_dir, trainer_run, 'out.json'), "r") as f:
#                         run_result = json.loads(f.read())
#                         run_result['run_id'] = trainer_run
#                         run_result['status'] = 'success'
#                         run_result['is_adapter_model_only'] = False
#                         run_result['labels'] = []
#                     if run_result['model_name'] in list_of_models:
#                         model_and_stats.append(run_result)
#                         # also include LoRA adapters
#                         lora_adapter_version = copy.deepcopy(run_result)
#                         lora_adapter_version['model_name'] = f"{lora_adapter_version['model_name']}{Config.LORA_ADAPTERS_SUFFIX}"
#                         lora_adapter_version['model_path'] = f"{lora_adapter_version['model_path']}{Config.LORA_ADAPTERS_SUFFIX}"
#                         lora_adapter_version['is_adapter_model_only'] = True
#                         model_and_stats.append(lora_adapter_version)
#                 else:
#                     err_file = os.path.join(trainer_runs_dir, trainer_run, 'error.log')
#                     if os.path.exists(err_file):
#                         with open(err_file, "r") as f:
#                             run_err = f.read()
#                     else:
#                         run_err = None
#                     model_and_stats.append({"run_id": trainer_run, "status": "failed", "error": run_err})
#         return ResponseHandler.success(data=model_and_stats)
#     except FileNotFoundError as e:
#         return ResponseHandler.success(data=[])
