import os
import subprocess
import sys
from pathlib import Path
from typing import Union

from starlette.responses import JSONResponse

from llama3_playground.core.config import Config
from llama3_playground.server.routers.utils import ResponseHandler


class OCRInferenceRunner:
    def run_ocr_process_and_collect_result(run_id: str, pdf_file: str) -> JSONResponse:
        import llama3_playground
        module_path = llama3_playground.__file__.replace('__init__.py', '')
        module_path = os.path.join(module_path, 'core', 'ocr_infer.py')

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


class LLMInferenceRunner:
    @staticmethod
    def run_inference_process_and_collect_result(
            run_id: str,
            model_name: str,
            context_json_data_file: str,
            question_file: str,
            prompt_text_file: str,
            max_new_tokens: int,
            embedding_model: str
    ) -> JSONResponse:
        tmp_questions_dir = os.path.join(str(Path.home()), 'temp-data', 'questions')
        os.makedirs(tmp_questions_dir, exist_ok=True)

        import llama3_playground
        module_path = llama3_playground.__file__.replace('__init__.py', '')
        module_path = os.path.join(module_path, 'core', 'llm_infer.py')

        inference_dir = f'{Config.inferences_dir}/{run_id}'
        os.makedirs(inference_dir, exist_ok=True)

        cmd_arr = [
            sys.executable, module_path,
            '-m', model_name,
            '-d', context_json_data_file,
            '-r', run_id,
            '-t', str(max_new_tokens),
            '-e', embedding_model,
            # '-p', prompt_text_file,
            '-q', question_file,
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
                    return _lora_model
                else:
                    return None
            else:
                return _latest
