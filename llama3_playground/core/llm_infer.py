# Installs Unsloth, Xformers (Flash Attention) and all other packages!
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
import json
import os
import sys
import time

import numpy as np
# noinspection PyUnresolvedReferences
import pysqlite3

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_MODE'] = 'disabled'

import uuid
import re
import shutil
import argparse
import traceback
from argparse import RawTextHelpFormatter
from prettytable import PrettyTable
from unsloth import FastLanguageModel
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from typing import List

from llama3_playground.core.config import Config
from llama3_playground.core.utils import ModelManager


class MyEmbeddings:
    def __init__(self, model: str):
        self.model = SentenceTransformer(model, trust_remote_code=True, device="cuda")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]

    def embed_query(self, query: str) -> List[float]:
        # return self.model.encode([query])
        return self.model.encode(query).tolist()


def get_embedding_function(embed_model_Path: str):
    embedding_function = MyEmbeddings(embed_model_Path)
    return embedding_function


class CreateChromaDB:
    # CHUNK_SIZE = 1024
    # CHUNK_OVERLAP = 100
    # TOP_K = 5

    def __init__(self, db_path: str, embed_model_path: str, chunk_size: int, chunk_overlap: int, top_k: int,
                 clear_db: bool = False):
        self.chroma_path = f"{db_path}"
        self.clear_db = clear_db
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embedding_function = get_embedding_function(embed_model_path)
        self.query_chunk_map = {}
        self.query_chunk_id_map = {}

    def populate_database(self, ocr_text: str, pdf_file_path: str, ocr_coordinates: dict):
        if self.clear_db:
            print("Clearing vector database...")
            self.clear_database()
        if not os.path.exists(self.chroma_path):
            chunks = self.generate_chunks_from_ocr_text(ocr_text=ocr_text, pdf_file_path=pdf_file_path,
                                                        ocr_coordinates=ocr_coordinates)
            self.add_to_chroma(chunks=chunks)
        else:
            print("Vector database already created.")

    def generate_chunks_from_ocr_text(self, ocr_text: str, pdf_file_path: str, ocr_coordinates: dict) -> List[Document]:
        split_pages = re.split(r'\n\n---PAGE \d+---\n\n', ocr_text)
        # split_pages = re.split(r'\n\n---PAGE-SEPARATOR---\n\n', ocr_text)
        split_pages = [page.strip() for page in split_pages if page.strip()]
        chunks = []
        for idx, page in enumerate(split_pages):
            page_ocr_coordinates = json.dumps(ocr_coordinates[str(idx)])
            # page_ocr_coordinates = str([])
            doc = Document(page_content=page,
                           metadata={"source": pdf_file_path, "page": idx, "coordinates": page_ocr_coordinates})
            chunks.append(doc)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                       chunk_overlap=self.chunk_overlap)
        chunks = text_splitter.split_documents(chunks)
        return chunks

    def add_to_chroma(self, chunks):
        db = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embedding_function
        )

        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        chunks_with_ids = self.calculate_chunk_ids(chunks=chunks)
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            db.persist()
        else:
            print("No new documents to add")
        print("\n")

    def calculate_chunk_ids(self, chunks) -> List[Document]:
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            chunk.metadata["id"] = chunk_id

        return chunks

    def clear_database(self):
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)

    def db_similarity_search(self, query, top_k, filter_dict=None):
        db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_function)
        results = db.similarity_search_with_score(query, k=top_k, filter=filter_dict)
        # print(results)
        # print("\n")
        return results

    def get_similar_chunks_metadata(self, query, top_k, filter_in_list):
        filter_dict = {"id": {"$in": filter_in_list}}
        results = self.db_similarity_search(query=query, top_k=top_k, filter_dict=filter_dict)
        print(query)
        # print(results)
        print(filter_dict)
        page_meta = {doc.metadata.get('page'): doc.metadata.get('coordinates') for doc, _score in results}
        return page_meta

    def retrieve_relevant_chunks(self, query_text: dict) -> str:
        for field_query, field_name in query_text.items():
            temp_start_time = time.time()
            results = self.db_similarity_search(query=field_query, top_k=self.top_k)

            sources = [doc.metadata.get('id', None) for doc, _score in results]
            print(f"{field_query}: {time.time() - temp_start_time} seconds :\n{sources}\n")
            self.query_chunk_id_map[f"{field_query} :as: {field_name}"] = sources

            context_text = [doc.page_content for doc, _score in results]
            self.query_chunk_map[f"{field_query} :as: {field_name}"] = context_text

        query_chunk_combined = self.combine_keys_if_match()
        query_chunk_combined = self.format_query(query_chunk_combined)
        return query_chunk_combined

    def format_query(self, query_dict):
        template = "Extract fields:  {query}. provide the result in a json format."
        return {template.replace('{query}', query): "\n\n---\n\n".join(context) for query, context in
                query_dict.items()}

    def combine_keys_if_match(self, threshold=0.6):
        combined_data = {}
        keys = list(self.query_chunk_map.keys())
        visited = set()

        for i, key1 in enumerate(keys):
            if key1 in visited:
                continue
            combined_keys = [key1]
            combined_values = set(self.query_chunk_map[key1])

            for key2 in keys[i + 1:]:
                if key2 in visited:
                    continue
                values1 = set(self.query_chunk_map[key1])
                values2 = set(self.query_chunk_map[key2])

                common_values = values1.intersection(values2)
                if len(common_values) >= np.floor(threshold * min(len(values1), len(values2))):
                    combined_keys.append(key2)
                    combined_values.update(values2)
                    visited.add(key2)

            combined_key = '; '.join(combined_keys)
            print(f"combined keys: {combined_key}")
            combined_data[combined_key] = list(combined_values)
            visited.add(key1)
        print("\n")
        return combined_data


def extract_json_from_string(input_str, query_text):
    # json_pattern = re.compile(r'\{.*?\}', re.DOTALL)
    json_pattern = re.compile(r'\{.*\}', re.DOTALL)
    json_match = json_pattern.search(input_str)

    query_fields = query_text.split("Extract fields:  ")[1].split(". provide the result in a json format.")[
        0].split("; ")
    query_fields_map = {field.split(" :as: ")[0]: field.split(" :as: ")[1] for field in query_fields}
    query_fields = [field.split(" :as: ")[1] for field in query_fields]
    try:
        if json_match:
            json_str = json_match.group()
            res_json = json.loads(json_str)
            res_json = {(query_fields_map[field] if field in query_fields_map else field): value for field, value in res_json.items()}
            res_json = {field: (res_json[field] if field in res_json else None) for field in query_fields}
            return res_json
        else:
            raise ValueError("No JSON object found in the input string")
    except Exception as e:
        print(f"Error: {str(e)}")
        return {field: None for field in query_fields}


def run_inference(model_path: str,
                  embed_model_path: str,
                  question_text: dict,
                  # prompt_text: str,
                  prompt_text_file: str,
                  ctx_json_file: str,
                  resp_file: str,
                  result_file: str,
                  rag_db_path: str,
                  max_new_tokens: int,
                  chunks_text_file: str,
                  chunk_size: int,
                  chunk_overlap: int,
                  top_k: int,
                  clear_db: bool):
    start_time = time.time()

    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    print(f"\n================== Loading the LLM model from {model_path} ==================")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    print(f"Model Loaded : {time.time() - start_time} seconds")

    with open(prompt_text_file, 'r') as f:
        prompt = f.read()

    with open(ctx_json_file, 'r') as f:
        ctx_dict = json.loads(f.read())
        text_data = ctx_dict['text_result']
        ocr_data = ctx_dict['ocr_result']

    print(f"\n================== Creating Vector DB ==================")
    chroma_db = CreateChromaDB(db_path=rag_db_path, embed_model_path=embed_model_path, clear_db=clear_db,
                               chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k)
    chroma_db.populate_database(ocr_text=text_data, pdf_file_path=ctx_json_file, ocr_coordinates=ocr_data['ocr-data'])
    print(f"Populated Vector DB: {time.time() - start_time} seconds")
    query_chunk_map = chroma_db.retrieve_relevant_chunks(query_text=question_text)
    print(f"Retrieved relevant chunks: {time.time() - start_time} seconds")


    with open(chunks_text_file, 'w') as f:
        f.write(json.dumps(query_chunk_map))
    print(f'Wrote the contextual chunks data fetched from vector database into file: {chunks_text_file}')

    final_response = ""
    final_result = {}
    final_result_raw = []
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    print(f"\n================== Genreating model response ==================")
    model_start_time = time.time()
    for query_text, context_text in query_chunk_map.items():
        temp_start_time = time.time()
        inputs = tokenizer(
            [
                prompt.format(
                    context_text,
                    query_text,
                    "",  # output - leave this blank for generation!
                )
            ], return_tensors="pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
        response = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        response = response.strip()
        final_response = final_response + "---------\nQuestion:\n---------\n" + query_text + "\n---------\nResponse:\n---------\n" + response + "\n---------\n" + "\n\n\n\n----Separator----\n\n\n\n"
        print("\n")
        print("---------")
        print("Question:")
        print(query_text)
        print("---------")
        print("Response:")
        print("---------")
        print(response)
        response = extract_json_from_string(response, query_text)
        print("Response formatted:")
        print("---------")
        print(response)
        print(f"Time taken: {time.time() - temp_start_time} seconds")
        print("---------")
        final_result.update(response)
        final_result_raw.append({"query_text": query_text, "response": response})

    print("\n")
    print("---------")
    print("Final Result:")
    print("---------")
    print(final_result)
    print("---------")
    print("\n")
    print(f"Model response Genreated: {time.time() - model_start_time} seconds")

    final_result_json = format_result_json(result_json=final_result_raw, db_obj=chroma_db)

    print(f"Time taken: {time.time() - start_time} seconds")

    print("\n")
    print("---------")
    print("Final Result:")
    print("---------")
    print(final_result_json)
    print("---------")

    with open(resp_file, 'w') as f:
        f.write(final_response)
    print(f'Wrote the response to {resp_file}')

    final_result_json = json.dumps(final_result_json)
    with open(result_file, 'w') as f:
        f.write(final_result_json)
    print(f'Wrote the response to {result_file}')
    print(f"Time taken: {time.time() - start_time} seconds")


# def get_coordinates_for_response(value, coord_dict):
#     if value:
#         value_tokens = value.split(" ")
#         start_coordinates = None
#         end_coordinates = None
#         try:
#             if value in coord_dict:
#                 final_coordinates = coord_dict[value]
#                 return final_coordinates
#             # else:
#             #     for s_idx, (s_field, s_coord) in enumerate(list(coord_dict.items())):
#             #         if s_field[0] == value_tokens[0]:
#             #             start_coordinates = s_coord
#             #             for e_idx, (e_field, e_coord) in enumerate(list(coord_dict.items())[s_idx+1:]):
#             #                 if e_field == value_tokens[-1]:
#             #                     end_coordinates = e_coord
#             #                     break
#             #         else:
#
#             #     return final_coordinates
#
#         except Exception as e:
#             print(str(e))
#             return []
#     else:
#         return []


def format_result_json(result_json, db_obj):
    page_coordinates_map = {}
    for result in result_json:
        query_text, response = result['query_text'], result['response']
        print(query_text)
        print("\n")
        query_fields = query_text.split("Extract fields:  ")[1].split(". provide the result in a json format.")[
            0].split("; ")
        for field in query_fields:
            print(field)
            field_new_1 = field.split(" :as: ")[0]
            field_new_2 = field.split(" :as: ")[1].replace(" ", "_")
            value = response[field_new_2]
            value_token_list = None
            if isinstance(value, str):
                value_token_list = value.strip().split(" ")
            if value is not None:
                sources = db_obj.query_chunk_id_map.get(field, None)
                query = f"{field}: {value}"
                # query = f"{field_new_1}: {value}"
                # query = f"{value}"
                # if isinstance(value, bool):
                #     query = f"{field}: {value}"
                page_meta = db_obj.get_similar_chunks_metadata(query=query, top_k=3, filter_in_list=sources)
                pg_no = list(page_meta.keys())[0]
                pg_coord = json.loads(list(page_meta.values())[0])
                left, top, right, bottom = [None, None, None, None]
                if value_token_list:
                    for coord in pg_coord:
                        if left is None and top is None and value_token_list[0] in coord:
                            print(coord)
                            left, top, right, bottom = coord[value_token_list[0]]
                        if left is not None and top is not None and value_token_list[-1] in coord:
                            print(coord)
                            _, _, right, bottom = coord[value_token_list[-1]]
                            break
                if pg_no not in page_coordinates_map:
                    page_coordinates_map[pg_no] = []
                page_coordinates_map[pg_no].append(
                    {"field": field_new_2, "value": value, "coordinates": [left, top, right, bottom]})
            else:
                if -1 not in page_coordinates_map:
                    page_coordinates_map[-1] = []
                page_coordinates_map[-1].append({"field": field_new_2, "value": value, "coordinates": []})
            print("---------\n")


    final_result_json = {"result": []}
    for page in page_coordinates_map.keys():
        page_res_json = {'pageNo': page, 'Fields': [], 'document_type': ""}
        for field_dict in page_coordinates_map[page]:
            field_res_json = {}
            field_res_json['key'] = field_dict['field']
            field_res_json['validation_rules'] = []
            field_res_json['field_type'] = "TEXT"
            field_res_json['valueSet'] = [
                {"value": field_dict['value'], "coordinates": field_dict['coordinates'], "is_validated": False,
                 "source": "Llama3-RAG"}]
            field_res_json['validation_status'] = "VALID_VALUE"
            field_res_json['confidence_score'] = 100
            field_res_json['is_mandatory'] = False
            field_res_json['is_user_edited'] = False
            field_res_json['validationResults'] = []
            page_res_json['Fields'].append(field_res_json)
        final_result_json["result"].append(page_res_json)
    return final_result_json


def print_cli_args(cli_args: argparse.Namespace):
    print("Using the following config:")
    t = PrettyTable(['Config Key', 'Specified Value'])
    t.align["Config Key"] = "r"
    t.align["Specified Value"] = "l"
    for k, v in cli_args.__dict__.items():
        t.add_row([k, v])
    print(t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A utility to infer from an LLM using a text file as context data.',
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        '-r',
        '--run-id',
        type=str,
        dest='run_id',
        help='A run ID (string) to create a folder with the inference run data. Example: "123"',
        required=False,
        default=str(uuid.uuid4())
    )
    parser.add_argument(
        '-p',
        '--prompt-text-file-path',
        type=str,
        dest='prompt_text_file_path',
        help='Path to the .txt data file to use as prompt for the model.',
        required=False,
        default=None
    )
    parser.add_argument(
        '-t',
        '--max-new-tokens',
        type=int,
        dest='max_new_tokens',
        help='Maximum number of new tokens to generate. Example: 1024. Default is 128.',
        required=False,
        default=128
    )
    parser.add_argument(
        '-m',
        '--model-name',
        type=str,
        dest='model_name',
        help=f'Name of the model to use for inference. Uses the models available at: "{Config.models_dir}". If the argument is not specified, it will try to use the latest model available. Example: "llama3-8b-custom-1720545601"',
        required=False,
        default=None
    )
    parser.add_argument(
        '-e',
        '--embedding-model',
        type=str,
        dest='embedding_model_name',
        help='Embedding Model Name. Example: "Alibaba-NLP/gte-base-en-v1.5", "BAAI/bge-base-en-v1.5", "Snowflake/snowflake-arctic-embed-l". Default is "Alibaba-NLP/gte-base-en-v1.5". Refer: https://huggingface.co/spaces/mteb/leaderboard',
        required=False,
        default="Alibaba-NLP/gte-base-en-v1.5"
    )
    parser.add_argument(
        '-l',
        '--prefer-lora-adapter-model',
        type=bool,
        dest='prefer_lora_adapter_model',
        help=f'Prefers using LoRA adapter model (if available) over full model.',
        required=False,
        default=False
    )
    parser.add_argument(
        '-cs',
        '--chunk-size',
        type=int,
        dest='chunk_size',
        help=f'maximum chunk size for the text splitter',
        required=False,
        default=1024
    )
    parser.add_argument(
        '-co',
        '--chunk-overlap',
        type=int,
        dest='chunk_overlap',
        help=f'maximum chunk overlap size for the text splitter',
        required=False,
        default=100
    )
    parser.add_argument(
        '-k',
        '--top-k-chunks',
        type=int,
        dest='top_k',
        help=f'top k chunks to be fetched from db based on the query',
        required=False,
        default=5
    )
    parser.add_argument(
        '-c',
        '--clear-db',
        type=bool,
        dest='clear_db',
        help=f'wherther to clear vector db or not',
        required=False,
        default=True
    )

    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument(
        '-d',
        '--json-data-file-path',
        type=str,
        dest='context_json_data_file_path',
        help='Path to the JSON data file to use as context for the model.',
        required=True
    )
    required_args.add_argument(
        '-q',
        '--question-file-path',
        type=str,
        dest='question_file_path',
        help='Path to the file that contains a question to be asked to the LLM. Example: "/app/question.txt"',
        required=True
    )

    args: argparse.Namespace = parser.parse_args()
    print_cli_args(cli_args=args)

    runId = args.run_id
    model_name = args.model_name
    embedding_model_name = args.embedding_model_name
    context_json_data_file_path = args.context_json_data_file_path
    question_file_path = args.question_file_path
    prompt_text_file_path = args.prompt_text_file_path
    max_new_tokens = args.max_new_tokens
    prefer_lora_adapter_model = args.prefer_lora_adapter_model
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    top_k = args.top_k
    clear_db = args.clear_db

    if model_name is None:
        model_name = ModelManager.get_latest_model(lora_adapters_only=False)
        print(f"Latest model is: {model_name}")
    else:
        print(f"Specified model: {model_name}")

    model_path = f'{Config.models_dir}/{model_name}'

    if prefer_lora_adapter_model is not None and prefer_lora_adapter_model is True and Config.LORA_ADAPTERS_SUFFIX not in model_name and os.path.exists(
            f'{model_path}{Config.LORA_ADAPTERS_SUFFIX}'):
        model_path = f'{model_path}{Config.LORA_ADAPTERS_SUFFIX}'
        print(
            f"Using model with LoRA adapters instead of the full model as `--prefer-lora-adapter-model` is selected. [{model_path}]")
    else:
        print(f"Using the full model:[{model_path}]")

    inference_dir = f'{Config.inferences_dir}/{runId}'
    os.makedirs(inference_dir, exist_ok=True)

    inference_run_question_file = f'{inference_dir}/question.json'
    ctx_data_file = f'{inference_dir}/context-data.json'
    resp_file = f'{inference_dir}/response.txt'
    result_file = f'{inference_dir}/result.json'
    prompt_text_file = f'{inference_dir}/prompt.txt'
    chunks_text_file = f'{inference_dir}/chunks-picked.txt'
    rag_db_path = f'{inference_dir}/db'

    model_path = f'{Config.models_dir}/{model_name}'
    embed_model_path = embedding_model_name

    if prompt_text_file_path is None:
        prompt = """
        You are a smart, logical and helpful assistant. Use the given context and extract the required fields from it and provide the result in a json format. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. If you cannot answer the question from the given documents, please dont answer.

    Below is the context that represents a document excerpt (a section of a document), paired with a related question. Write a suitable response to the question based on the given context.

    ### Context:
    {}

    ### Question:
    {}

    ### Response:
    {}"""
        with open(prompt_text_file, 'w') as f:
            f.write(prompt)
        print(f'Wrote prompt text to: {prompt_text_file}')
    else:
        shutil.copyfile(prompt_text_file_path, prompt_text_file)
        print(f'Wrote prompt text to: {prompt_text_file}')

    shutil.copyfile(context_json_data_file_path, ctx_data_file)
    shutil.copyfile(question_file_path, inference_run_question_file)
    print(f'Wrote question text to: {inference_run_question_file}')

    with open(inference_run_question_file, 'r') as f:
        question_text = json.loads(f.read())
    try:
        run_inference(
            model_path=model_path,
            question_text=question_text,
            # prompt_text=prompt_text,
            ctx_json_file=context_json_data_file_path,
            resp_file=resp_file,
            result_file=result_file,
            prompt_text_file=prompt_text_file,
            max_new_tokens=max_new_tokens,
            chunks_text_file=chunks_text_file,
            embed_model_path=embed_model_path,
            rag_db_path=rag_db_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            clear_db=clear_db
        )

        with open(os.path.join(inference_dir, 'RUN-STATUS'), 'w') as f:
            f.write("success")
    except Exception as e:
        error_str = traceback.format_exc()
        print(f"Infer RAG Error: {e}. Cause: {error_str}")
        with open(os.path.join(inference_dir, 'error.log'), 'w') as f:
            f.write(error_str)
        with open(os.path.join(inference_dir, 'RUN-STATUS'), 'w') as f:
            f.write("failure")
