import os
from typing import Union

from llama3_playground.core.config import Config


class ModelManager:
    @staticmethod
    def list_trained_models(include_lora_adapters: bool = True) -> [str]:
        _models = os.listdir(Config.models_dir)
        if include_lora_adapters:
            _models = sorted([m for m in _models], reverse=True)
        else:
            _models = sorted([m for m in _models if Config.LORA_ADAPTERS_SUFFIX in m], reverse=True)

        # Items to remove
        to_remove = ['.ipynb_checkpoints']
        _filtered_models = [item for item in _models if item not in to_remove]

        return _filtered_models

    @staticmethod
    def get_latest_model(lora_adapters_only: bool = False) -> Union[str, None]:
        _models = ModelManager.list_trained_models()
        if len(_models) == 0:
            return None
        else:
            _latest = _models[0].replace(Config.LORA_ADAPTERS_SUFFIX, '')
            if lora_adapters_only:
                _lora_model = f'{_latest}{Config.LORA_ADAPTERS_SUFFIX}'
                _lora_model_dir = os.path.join(Config.models_dir, _lora_model)
                if os.path.exists(_lora_model_dir):
                    return None
                else:
                    return _lora_model
            else:
                return _latest
