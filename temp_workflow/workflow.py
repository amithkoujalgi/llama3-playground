import json
import os
import time
from pathlib import Path

import requests

prompt_txt = """You are a smart, logical and helpful assistant.
Use the given context and extract the required fields from it and provide the result in a valid JSON format. Do not create invalid JSON.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information.
If you cannot answer the question from the given documents, please do not answer.

Below is the context that represents a document excerpt (a section of a document), paired with a related question.
Write a suitable response to the question based on the given context.

### Context:
{}

### Question:
{}

### Response:
{}
"""
question_dict = {
    "Name of the Adopting Employer": "EMPLOYER_NAME",
    "Address of the Adopting Employer": "EMPLOYER_ADDRESS",
    "City of the Adopting Employer": "EMPLOYER_CITY",
    "State of the Adopting Employer": "EMPLOYER_STATE",
    "Adopting Employer Zip (in 5 digit format)": "EMPLOYER_ZIP",
    "Adopting Employer Telephone number": "EMPLOYER_TELEPHONE",
    "Adopting Employer TIN": "EMPLOYER_TIN",
    "Tax Year End of the Adopting Employer": "EMPLOYER_TAX_YEAR_END",
    "Type of Business of the Adopting Employer": "EMPLOYER_TYPE_OF_BUSINESS",
    "Plan Name of the Adopting Employer": "EMPLOYER_PLAN_NAME",
    "Plan Sequence Number of the Adopting Employer": "EMPLOYER_PLAN_SEQUENCE_NUMBER",
    "Trust ID of the Adopting Employer": "EMPLOYER_TRUST_ID",
    "Account Number of the Adopting Employer": "EMPLOYER_ACCOUNT_NUMBER",
    "Is it a new plan or existing plan (as true or false)": "NEW_PLAN",
    "Plan Effective Date": "PLAN_EFFECTIVE_DATE",
    "Effective Date of Amendment or Restatement": "AMENDMENT_EFFECTIVE_DATE",
    "Initial Plan Document Effective Date": "INITIAL_PLAN_DOCUMENT_EFFECTIVE_DATE",
    "Frozen Plan Effective Date": "FROZEN_PLAN_EFFECTIVE_DATE",
    "Employee Age requirement to become a Participant in the Plan (in months)": "AGE_REQUIREMENT_IN_MONTHS",
    "Employee Eligibility Service Requirement to become a Participant in the Plan (in months, 0 if not required)": "ELIGIBILITY_SERVICE_REQUIREMENT_IN_MONTHS",
    "Employee Employed Date": "EMPLOYEE_EMPLOYED_DATE",
    "Does the waiver apply to all employees (as true or false)": "EMPLOYEE_CLASSIFICATION_ALL",
    "Does the waiver apply to employees who are defined (as true or false)": "EMPLOYEE_CLASSIFICATION_DEFINED",
    "Employees Employeed Entry Date - Option: Should use the above specified date (as true or false)": "ENTRY_DATE_SAME_AS_ABOVE",
    "Employees Employeed Entry Date - Option: Specified date in month, day and year format": "ENTRY_DATE",
    "Will Elective Deferrals be permitted (as true or false)": "ELECTIVE_DEFERRALS_PERMITTED",
    "Will Roth Elective Deferrals be permitted (as true or false)": "ROTH_DEFERRALS_PERMITTED",
    "May a Contributing Participant make Nondeductible Employee Contributions pursuant to Plan Section 3.05 (as true or false)": "NONDEDUCTIBLE_CONTRIBUTIONS_PERMITTED",
    "Will a Participant be entitled to request a loan (as true or false)": "LOANS_PERMITTED",
    "Will life insurance investments be permitted (as true or false)": "LIFE_INSURANCE_INVESTMENTS_PERMITTED",
    "Will a Participant be allowed to purchase and distribute Qualifying Longevity Annuity Contracts (as true or false)": "QUALIFYING_LONGEVITY_ANNUITY_CONTRACTS_PERMITTED",
    "Name of the document provider": "DOCUMENT_PROVIDER",
    "Address of the document provider": "DOCUMENT_PROVIDER_ADDRESS",
    "Telephone of the document provider": "DOCUMENT_PROVIDER_TELEPHONE",
    "Authorized employer name": "SIGNATURE_TYPE_NAME",
    "Authorized employer title": "SIGNATURE_TITLE"
}


class WorkflowRunner:
    def __init__(self, host: str):
        self._host = host

    def trigger_ocr(self, pdf_file_path: str) -> str:
        """
        Returns run ID
        """
        headers = {
            'accept': 'application/json',
            # requests won't add a boundary if this header is set when you pass files=
            # 'Content-Type': 'multipart/form-data',
        }

        files = {
            'file': (os.path.basename(pdf_file_path), open(pdf_file_path, 'rb'), 'application/pdf'),
        }
        print("Triggering OCR run...")
        response = requests.post(f'{self._host}/api/ocr/async/run', headers=headers, files=files)
        if response.status_code == 200:
            res = response.json()
            _run_id = res['run_id']
            print(f"Triggered OCR run. Run ID: {_run_id}")
            return _run_id
        else:
            res = response.json()
            res = json.dumps(res, indent=4)
            print(res)
            raise Exception(res)

    def check_ocr_status(self, run_id: str, poll_interval_seconds: int = 1):
        start_time = time.time()
        while True:
            time.sleep(poll_interval_seconds)
            headers = {
                'accept': 'application/json',
            }
            response = requests.get(f'{self._host}/api/ocr/async/run/{run_id}', headers=headers)
            if response.status_code == 200:
                res = response.json()
                _status = res['data']['status']
                if _status == 'success':
                    print('OCR extraction complete.')
                    end_time = time.time()
                    print(f'Time taken: {end_time - start_time}')
                    return res['data']
                if _status == 'running':
                    print('OCR extraction: running...')
                    continue
                else:
                    print(f'OCR extraction failed with status: {_status}')
                    end_time = time.time()
                    print(f'Time taken: {end_time - start_time}')
                    return Exception(res)
            else:
                res = response.json()
                print('OCR extraction failed!')
                end_time = time.time()
                print(f'Time taken: {end_time - start_time}')
                raise Exception(res)

    def trigger_LLM_inference(self, context_json_data_file: str, question_file: str, prompt_text_file: str):
        headers = {
            'accept': 'application/json',
            # requests won't add a boundary if this header is set when you pass files=
            # 'Content-Type': 'multipart/form-data',
        }

        params = {
            'llm_identifier': 'llama-3-8b-instruct-custom-1724066533-lora-adapters',
            'max_new_tokens': '128',
            'embedding_model': 'Alibaba-NLP/gte-base-en-v1.5',
        }

        files = {
            'context_json_data_file': (
                os.path.basename(context_json_data_file), open(context_json_data_file, 'rb'), 'application/json'),
            'question_file': (os.path.basename(question_file), open(question_file, 'rb'), 'text/plain'),
            'prompt_text_file': (os.path.basename(prompt_text_file), open(prompt_text_file, 'rb'), 'application/json'),
        }
        print("Triggering LLM Inference run...")
        response = requests.post(f'{self._host}/api/infer/async/run', params=params, headers=headers, files=files)
        if response.status_code == 200:
            res = response.json()
            _run_id = res['run_id']
            print(f"Triggered LLM Inference run. Run ID: {_run_id}")
            return _run_id
        else:
            res = response.json()
            res = json.dumps(res, indent=4)
            print(res)
            raise Exception(res)

    def check_llm_inference_status(self, run_id: str, poll_interval_seconds: int = 1):
        start_time = time.time()
        while True:
            time.sleep(poll_interval_seconds)
            headers = {
                'accept': 'application/json',
            }
            response = requests.get(f'{self._host}/api/infer/async/run/{run_id}', headers=headers)
            if response.status_code == 200:
                res = response.json()
                _status = res['data']['status']
                if _status == 'success':
                    print('LLM Inference complete.')
                    end_time = time.time()
                    print(f'Time taken: {end_time - start_time}')
                    return res['data']
                if _status == 'running':
                    print('LLM Inference: running...')
                    continue
                else:
                    print(f'LLM Inference failed with status: {_status}')
                    end_time = time.time()
                    print(f'Time taken: {end_time - start_time}')
                    return Exception(res)
            else:
                res = response.json()
                print('LLM Inference failed!')
                print(res['data'])
                end_time = time.time()
                print(f'Time taken: {end_time - start_time}')
                raise Exception(res)


workflow = WorkflowRunner(host='http://localhost:8883')
try:
    pdf_file = '/path/to/document.pdf'

    tmp_dir = os.path.join(str(Path.home()), 'temp-data')
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, 'question.json'), 'w') as f:
        f.write(json.dumps(question_dict, indent=True))
    with open(os.path.join(tmp_dir, 'prompt.txt'), 'w') as f:
        f.write(prompt_txt)

    question_file = os.path.join(tmp_dir, 'question.json')
    prompt_text_file = os.path.join(tmp_dir, 'prompt.txt')

    ocr_run_id = workflow.trigger_ocr(pdf_file)
    ocr_data = workflow.check_ocr_status(run_id=ocr_run_id, poll_interval_seconds=3)

    ctx_data_file = os.path.join(tmp_dir, 'context-data.json')
    with open(ctx_data_file, 'w') as f:
        json.dump(ocr_data, f, indent=4)

    llm_inference_run_id = workflow.trigger_LLM_inference(
        context_json_data_file=ctx_data_file,
        question_file=question_file,
        prompt_text_file=prompt_text_file
    )
    llm_inference_data = workflow.check_llm_inference_status(run_id=llm_inference_run_id, poll_interval_seconds=3)
    llm_response_json_file = os.path.join(tmp_dir, 'llm_response.json')
    with open(llm_response_json_file, 'w') as f:
        f.write(json.dumps(llm_inference_data, indent=True))
    print(f'LLM response written to: {llm_response_json_file}')
except Exception as e:
    print(e)
