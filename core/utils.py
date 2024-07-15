import os
from typing import Union

from config import Config


class ModelManager:
    @staticmethod
    def list_trained_models(lora_adapters_only: bool = False) -> [str]:
        """
        Lists the trained models available in the configured models' directory.

        Args:
            lora_adapters_only (bool, optional): If True, only models saved as LoRA adapters will be listed.
                                                 If False (default), full models (base model merged with LoRA adapters) 
                                                 will be listed.

        Returns:
            list: A list of strings representing the names of the trained models.
        """
        _models = os.listdir(Config.models_dir)
        if lora_adapters_only:
            _models = sorted([m for m in _models if Config.LORA_ADAPTERS_SUFFIX in m], reverse=True)
        else:
            _models = sorted([m for m in _models if Config.LORA_ADAPTERS_SUFFIX not in m], reverse=True)
        return _models

    @staticmethod
    def get_latest_model(lora_adapters_only: bool = False) -> Union[str, None]:
        """
        Fetches the latest model available in the configured models directory.

        Args:
            lora_adapters_only (bool, optional): If True, only the latest model saved as LoRA adapters will be returned.
                                                 If False (default), the latest model saved as full model (base model 
                                                 merged with LoRA adapters) will be returned.

        Returns:
            str or None: The name of the latest model, or None if no model is found.
        """
        _models = ModelManager.list_trained_models()
        if len(_models) == 0:
            return None
        else:
            _latest = _models[0]
            if lora_adapters_only:
                _lora_model = f'{_latest}{Config.LORA_ADAPTERS_SUFFIX}'
                _lora_model_dir = os.path.join(Config.models_dir, _lora_model)
                if os.path.exists(_lora_model_dir):
                    return None
                else:
                    return _lora_model
            else:
                return _latest
