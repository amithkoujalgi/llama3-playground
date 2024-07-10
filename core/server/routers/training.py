import json
import logging
import os

from fastapi import APIRouter

from server.routers.utils import ResponseHandler, is_training_process_running

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()


# Example usage
@router.get('/status', summary='Get status of training',
            description='API to get status of training. Tells you if the training process is ongoing or not.')
async def training_status():
    return ResponseHandler.success(data={"running": is_training_process_running()})


@router.get('/model-stats', summary="Get all trained models and their stats",
            description="API to get all trained models and their stats")
async def get_models_and_stats():
    trainer_runs_dir = "/app/data/trainer-runs"
    try:
        model_and_stats = []
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
                    model_and_stats.append(run_result)
                else:
                    err_file = os.path.join(trainer_runs_dir, trainer_run, 'error.log')
                    if os.path.exists(err_file):
                        with open(err_file, "r") as f:
                            run_err = f.read()
                    else:
                        run_err = None
                    model_and_stats.append({"run_id": trainer_run, "status": "failed", "error": run_err})
            else:
                model_and_stats.append({"run_id": trainer_run, "status": "running"})
        return ResponseHandler.success(data=model_and_stats)
    except FileNotFoundError as e:
        return ResponseHandler.success(data=[])
