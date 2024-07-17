import json
import os
import re

import easyocr
import numpy as np
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStore
from pdf2image import convert_from_path


class OCR:

    def __init__(self, pdf_file_path: str):
        self._pdf_file_path = pdf_file_path

    def extract_text(self, output_file_path: str, overwrite: bool = False, line_threshold=20,
                     paragraph_threshold=50) -> str:
        """
        Extract text from a PDF file and align it into paragraphs.
        :param output_file_path: Path to save the aligned text file.
        :param line_threshold: Threshold for considering two bounding boxes as part of the same line.
        :param paragraph_threshold: Threshold for considering two lines as part of the same paragraph.
        :return: Aligned text as a string.
        """

        if not overwrite and os.path.exists(output_file_path):
            with open(output_file_path, 'r') as _f:
                return _f.read()

        _reader = easyocr.Reader(['en'])
        _images = convert_from_path(self._pdf_file_path)

        _aligned_text = ""
        _paragraph_gaps = []

        for _img in _images:
            _img = np.array(_img)  # Convert PIL Image to numpy array for EasyOCR
            _results = _reader.readtext(_img, detail=1)

            # Group results by lines based on vertical position
            _lines = []
            for _bbox, _text, _ in _results:
                _top_left = _bbox[0]
                _found = False
                for _line in _lines:
                    if abs(_line['top'] - _top_left[1]) < line_threshold:  # Line threshold
                        _line['content'].append((_bbox, _text))
                        _found = True
                        break
                if not _found:
                    _lines.append({'top': _top_left[1], 'content': [(_bbox, _text)]})

            # Sort each line by horizontal position and detect paragraph breaks
            _prev_bottom = 0
            for _line in sorted(_lines, key=lambda x: x['top']):
                _line['content'].sort(key=lambda x: x[0][0][0])  # Sort by X coordinate
                _text_line = ' '.join([_text for _, _text in _line['content']])

                # Check for paragraph breaks
                _paragraph_gap = _line['top'] - _prev_bottom
                _paragraph_gaps.append(_paragraph_gap)
                if _paragraph_gap > np.percentile(_paragraph_gaps, 90):  # Use a higher percentile for paragraph breaks
                    _aligned_text += "\n\n\n"
                    _aligned_text += "---PAGE---"
                    _aligned_text += "\n\n\n"
                _aligned_text += _text_line + "\n"

                _prev_bottom = _line['content'][-1][0][2][1]  # Update bottom position from the last element of the line

        with open(output_file_path, 'w') as _f:
            _f.write(_aligned_text)
        return _aligned_text


class DocAI:

    def __init__(self, text_file_path: str, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self._text_file_path = text_file_path
        self._load_doc()
        self._model = model
        self._base_url = base_url
        self._llm = Ollama(model=self._model, base_url=self._base_url)

    def _load_doc(self):
        _loader = TextLoader(self._text_file_path)
        _documents = _loader.load()
        _split_texts = _documents[0].page_content.split('---PAGE')
        _docs = [Document(page_content=text) for text in _split_texts if text.strip()]
        _embeddings = HuggingFaceEmbeddings()
        self._vector_store: VectorStore = FAISS.from_documents(_docs, _embeddings)

    def _find_chunks_for_text_from_vector_store(self, query: str) -> []:
        _retriever = self._vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        _docs = _retriever.invoke(query)
        _texts = []
        for _doc in _docs:
            _texts.append(_doc.page_content)
        return _texts

    def prepare_refined_chunks_for_llm(self, data, verbose: bool = False):
        _contextual_chunks = []

        for _item in data:
            _lookup_text = _item["lookup-text"]
            _chunks_found = self._find_chunks_for_text_from_vector_store(query=_lookup_text)

            _chunk_info = {
                "lookup-texts": [_lookup_text],
                "chunks-found": sorted(_chunks_found)
            }

            _contextual_chunks.append(_chunk_info)

        def _merge_entries(_entries):
            _merged_entries = {}
            for _entry in _entries:
                _chunks_tuple = tuple(_entry['chunks-found'])
                if _chunks_tuple in _merged_entries:
                    _merged_entries[_chunks_tuple]['lookup-texts'].extend(_entry['lookup-texts'])
                else:
                    _merged_entries[_chunks_tuple] = {
                        'lookup-texts': _entry['lookup-texts'][:],
                        'chunks-found': _entry['chunks-found']
                    }
            return list(_merged_entries.values())

        _merged_chunks = _merge_entries(_contextual_chunks)

        for _chunk in _merged_chunks:
            for _lookup_text in _chunk['lookup-texts']:
                for _item in data:
                    if _lookup_text == _item['lookup-text']:
                        if 'question-texts' not in _chunk:
                            _chunk['question-texts'] = []
                        _chunk['question-texts'].append(_item['question-text'])

        for i in _merged_chunks:
            i['lookup-texts'] = list(set(i['lookup-texts']))
            i['question-texts'] = list(set(i['question-texts']))

        if verbose:
            print(json.dumps(_merged_chunks, indent=4))
        return _merged_chunks

    def ask_llm(self, refined_chunks: [], verbose: bool = False) -> []:
        qna = []
        for i in refined_chunks:
            q_template = """<<ctx>>


The above data is the about a section of a retirement plan document.
Below are the list of fields specified and beside each field is the key that needs to be used to represent the field in a short form.
Make sure to use the exactly the specified field keys.
From the above data, give me the following fields and their values as JSON:

<<qtn>>
"""
            all_fields_question = ""
            full_ctx = ""

            for q in i['question-texts']:
                all_fields_question += f'{q}\n'
            for c in i['chunks-found']:
                full_ctx += f'{c}\n\n'

            all_fields_question = re.sub(r'\n+', '\n', all_fields_question)
            full_ctx = re.sub(r'\n+', '\n', full_ctx)

            question_to_llm = q_template.replace('<<ctx>>', full_ctx).replace('<<qtn>>', all_fields_question)
            answer = self._llm.invoke(question_to_llm)
            qna.append({
                'question': question_to_llm,
                'answer': answer
            })
            if verbose:
                print('======PROMPT======')
                print(question_to_llm)
                print('======RESPONSE======')
                print(answer)
        return qna


lookup_data = [
    {
        "lookup-text": 'Employer Information, adopting employer name, address, city, state, zip, telephone, Federal Tax Identification Number, Plan Sequence Number, Trust Identification Number',
        "question-text": "\n\nName of Adopting Employer as 'EMPLOYER_NAME' \nEmployer Address as 'EMPLOYER_ADDRESS'\nEmployer City as 'EMPLOYER_CITY'\nEmployer State as 'EMPLOYER_STATE'\nEmployer Zip as 'EMPLOYER_ZIP'\nEmployer Telephone as 'EMPLOYER_TELEPHONE'"
    },
    {
        "lookup-text": 'Employer Information, adopting employer name, address, city, state, zip, telephone, Federal Tax Identification Number, Plan Sequence Number, Trust Identification Number',
        "question-text": "Employer TIN as 'EMPLOYER_TIN' \nEmployer Trust ID as 'EMPLOYER_TRUST_ID' \nEmployer Account Number as 'EMPLOYER_ACCOUNT_NUMBER' \nPlan Sequence Number as 'EMPLOYER_PLAN_SEQUENCE_NUMBER' \nPlan Name as 'PLAN_NAME'"
    },
    {
        "lookup-text": 'roth deferrals, eligibility service requirement',
        "question-text": "Age requirement in months as 'AGE_REQUIREMENT_IN_MONTHS' \nEligibility Service Requirement in months (0 if not required) as 'ELIGIBILITY_SERVICE_REQUIREMENT_IN_MONTHS' \nWill Elective Deferrals be permitted (as true or false) as 'ELECTIVE_DEFERRALS_PERMITTED' \nWill Roth Elective Deferrals be permitted (as true or false) as 'ROTH_DEFERRALS_PERMITTED'"
    }
]

out_file = '/Users/amithkoujalgi/Downloads/llm-extraction/new-ocr-data.txt'
# ocr = OCR(pdf_file_path='/Users/amithkoujalgi/Downloads/llm-extraction/XPAA Demo3 IK.pdf')
# ocr.extract_text(output_file_path=out_file, overwrite=False)

doc_ai = DocAI(
    text_file_path=out_file,
    base_url='http://192.168.29.223:11434',
    model='llama3'
)
chunks_obtained = doc_ai.prepare_refined_chunks_for_llm(data=lookup_data, verbose=True)
doc_ai.ask_llm(refined_chunks=chunks_obtained, verbose=True)
