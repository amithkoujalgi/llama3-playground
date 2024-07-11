import json


def get_config() -> dict:
    with open('/app/config.json') as jf:
        json_config = json.load(jf)
        return json_config


class Config:
    base_model = get_config()['base_model']
    training_dataset_dir_path = get_config()['training_dataset_dir_path']
    checkpoints_dir = get_config()['checkpoints_dir']
    models_dir = get_config()['models_dir']
    trainer_runs_dir = get_config()['trainer_runs_dir']
    ocr_runs_dir = get_config()['ocr_runs_dir']
    inferences_dir = get_config()['inferences_dir']
    fine_tuned_model_name_prefix = get_config()['fine_tuned_model_name_prefix']
