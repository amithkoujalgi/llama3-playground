# Installs Unsloth, Xformers (Flash Attention) and all other packages!
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
import json
import os
import sys

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

    def populate_database(self, ocr_text: str, pdf_file_path: str):
        if self.clear_db:
            print("Clearing vector database...")
            self.clear_database()
        if not os.path.exists(self.chroma_path):
            chunks = self.generate_chunks_from_ocr_text(ocr_text=ocr_text, pdf_file_path=pdf_file_path)
            self.add_to_chroma(chunks=chunks)
        else:
            print("Vector database already created.")

    def generate_chunks_from_ocr_text(self, ocr_text: str, pdf_file_path: str) -> List[Document]:
        split_pages = re.split(r'---PAGE \d+---', ocr_text)
        split_pages = [page.strip() for page in split_pages if page.strip()]
        chunks = []
        for idx, page in enumerate(split_pages):
            doc = Document(page_content=page, metadata={"source": pdf_file_path, "page": idx})
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

    def retrieve_relevant_chunks(self, query_text: dict) -> str:
        db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_function)
        # query_text = extract_json_from_string(query_text)
        for field_query, field_name in query_text.items():
            results = db.similarity_search_with_score(field_query, k=self.top_k)
            # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            # print(f"Retrieved relevant context chunks: \n{context_text}\n")
            sources = [doc.metadata.get('id', None) for doc, _score in results]
            print(f"{field_query}:\n{sources}\n")
            context_text = [doc.page_content for doc, _score in results]
            self.query_chunk_map[field_name] = context_text
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

            combined_key = ', '.join(combined_keys)
            print(f"combined keys: {combined_key}")
            combined_data[combined_key] = list(combined_values)
            visited.add(key1)
        print("\n")
        return combined_data


def extract_json_from_string(input_str, query_text):
    # json_pattern = re.compile(r'\{.*?\}', re.DOTALL)
    json_pattern = re.compile(r'\{.*\}', re.DOTALL)
    json_match = json_pattern.search(input_str)

    if json_match:
        json_str = json_match.group()
        return json.loads(json_str)
    else:
        print("No JSON object found in the input string")
        query_fields = query_text.split("Extract fields:  ")[1].split(". provide the result in a json format.")[
            0].split(", ")
        return {field: None for field in query_fields}


def run_inference(model_path: str,
                  embed_model_path: str,
                  question_text: dict,
                  prompt_text: str,
                  prompt_text_file: str,
                  ctx_json_file: str,
                  resp_file: str,
                  rag_db_path: str,
                  max_new_tokens: int,
                  chunks_text_file: str,
                  chunk_size: int,
                  chunk_overlap: int,
                  top_k: int,
                  clear_db: bool):
    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    prompt = """
    You are a smart, logical and helpful assistant.
    [PROMPT_PLACEHOLDER]

    Below is the context that represents a document excerpt (a section of a document), paired with a related question. Write a suitable response to the question based on the given context.

    ### Context:
    {}

    ### Question:
    {}

    ### Response:
    {}"""

    if prompt_text is not None and type(prompt_text) == str:
        prompt = prompt.replace('[PROMPT_PLACEHOLDER]', prompt_text)

    with open(prompt_text_file, 'w') as f:
        f.write(prompt)
    print(f'Wrote prompt text to: {prompt_text_file}')

    with open(ctx_json_file, 'r') as f:
        ctx_dict = json.loads(f.read())
        text_data = ctx_dict['text_result']
        ocr_data = ctx_dict['ocr_result']

    chroma_db = CreateChromaDB(db_path=rag_db_path, embed_model_path=embed_model_path, clear_db=clear_db,
                               chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k)
    chroma_db.populate_database(ocr_text=text_data, pdf_file_path=ctx_json_file)
    query_chunk_map = chroma_db.retrieve_relevant_chunks(query_text=question_text)

    with open(chunks_text_file, 'w') as f:
        f.write(json.dumps(query_chunk_map))
    print(f'Wrote the contextual chunks data fetched from vector database into file: {chunks_text_file}')

    final_response = {}
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    for query_text, context_text in query_chunk_map.items():
        inputs = tokenizer(
            [
                prompt.format(
                    context_text,
                    query_text,
                    "",  # output - leave this blank for generation!
                )
            ], return_tensors="pt").to("cuda")

        # from transformers import TextStreamer
        # text_streamer = TextStreamer(tokenizer)
        # outputs = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True)
        # response = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]

        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
        response = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        response = response.strip()
        # print(response)
        response = extract_json_from_string(response, query_text)
        print("\n")
        print("---------")
        print("Question:")
        print(query_text)
        print("---------")
        print("Response:")
        print("---------")
        print(response)
        print("---------")
        final_response.update(response)

    print("\n")
    print("---------")
    print("Final Response:")
    print("---------")
    print(final_response)
    print("---------")
    final_response = json.dumps(final_response)
    with open(resp_file, 'w') as f:
        f.write(final_response)
    print(f'Wrote the response to {resp_file}')


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
        '--prompt-text',
        type=str,
        dest='prompt_text',
        help='Custom prompt to be set to the LLM. Example: "Always respond in a JSON format"',
        required=False,
        default=""
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
    prompt_text = args.prompt_text
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
    prompt_text_file = f'{inference_dir}/prompt.txt'
    chunks_text_file = f'{inference_dir}/chunks-picked.txt'
    rag_db_path = f'{inference_dir}/db'

    model_path = f'{Config.models_dir}/{model_name}'
    embed_model_path = embedding_model_name

    shutil.copyfile(context_json_data_file_path, ctx_data_file)
    shutil.copyfile(question_file_path, inference_run_question_file)
    print(f'Wrote question text to: {inference_run_question_file}')

    with open(inference_run_question_file, 'r') as f:
        question_text = json.loads(f.read())
    try:
        run_inference(
            model_path=model_path,
            question_text=question_text,
            prompt_text=prompt_text,
            ctx_json_file=context_json_data_file_path,
            resp_file=resp_file,
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
