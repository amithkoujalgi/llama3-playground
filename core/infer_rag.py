# Installs Unsloth, Xformers (Flash Attention) and all other packages!
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
import os
import sys

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
from config import Config
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from typing import List


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
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 100
    TOP_K = 5

    def __init__(self, db_path: str, embed_model_path: str, clear_db: bool = False):
        self.chroma_path = f"{db_path}"
        self.clear_db = clear_db
        self.embedding_function = get_embedding_function(embed_model_path)

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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CreateChromaDB.CHUNK_SIZE,
                                                       chunk_overlap=CreateChromaDB.CHUNK_OVERLAP)
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

    def retrieve_relevant_chunks(self, query_text: str) -> str:
        db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_function)
        results = db.similarity_search_with_score(query_text, k=CreateChromaDB.TOP_K)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        return context_text


def run_inference(model_path: str,
                  embed_model_path: str,
                  question_text: str,
                  prompt_text: str,
                  prompt_text_file: str,
                  ctx_file: str,
                  resp_file: str,
                  rag_db_path: str,
                  max_new_tokens: int,
                  chunks_text_file: str):
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

    with open(ctx_file, 'r') as f:
        context = f.read()

    chroma_db = CreateChromaDB(db_path=rag_db_path, embed_model_path=embed_model_path, clear_db=True)
    chroma_db.populate_database(ocr_text=context, pdf_file_path=ctx_file)
    context_text = chroma_db.retrieve_relevant_chunks(query_text=question_text)

    with open(chunks_text_file, 'w') as f:
        f.write(context_text)
    print(f'Wrote the contextual chunks data fetched from vector database into file: {chunks_text_file}')

    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    inputs = tokenizer(
        [
            prompt.format(
                context_text,
                question_text,
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
    print("\n")
    print("---------")
    print("Response:")
    print("---------")
    print(response)
    print("---------")
    with open(resp_file, 'w') as f:
        f.write(response)
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
        '-e',
        '--embedding-model',
        type=str,
        dest='embedding_model_name',
        help='Embedding Model Name. Example: "Alibaba-NLP/gte-base-en-v1.5", "Snowflake/snowflake-arctic-embed-l". Default is "Alibaba-NLP/gte-base-en-v1.5". Refer: https://huggingface.co/spaces/mteb/leaderboard',
        required=False,
        default="Alibaba-NLP/gte-base-en-v1.5"
    )
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument(
        '-m',
        '--model-name',
        type=str,
        dest='model_name',
        help=f'Name of the model to use for inference. Uses the models available at: "{Config.models_dir}". Example: "llama3-8b-custom-1720545601"',
        required=True
    )
    required_args.add_argument(
        '-d',
        '--data-file-path',
        type=str,
        dest='context_data_file_path',
        help='Path to the data file to use as context for the model. Example: "llama3-8b-custom-1720545601"',
        required=True
    )
    required_args.add_argument(
        '-q',
        '--question-text',
        type=str,
        dest='question_text',
        help='Question to be asked to the LLM. Example: "Who is Jashpal?"',
        required=True
    )

    args: argparse.Namespace = parser.parse_args()
    print_cli_args(cli_args=args)

    runId = args.run_id
    model_name = args.model_name
    embedding_model_name = args.embedding_model_name
    context_data_file_path = args.context_data_file_path
    question_text = args.question_text
    prompt_text = args.prompt_text
    max_new_tokens = args.max_new_tokens

    inference_dir = f'{Config.inferences_dir}/{runId}'
    os.makedirs(inference_dir, exist_ok=True)

    question_file = f'{inference_dir}/question.txt'
    ctx_data_file = f'{inference_dir}/context-data.txt'
    resp_file = f'{inference_dir}/response.txt'
    prompt_text_file = f'{inference_dir}/prompt.txt'
    chunks_text_file = f'{inference_dir}/chunks-picked.txt'
    rag_db_path = f'{inference_dir}/db'

    with open(question_file, 'w') as f:
        f.write(question_text)

    shutil.copyfile(context_data_file_path, ctx_data_file)

    model_path = f'{Config.models_dir}/{model_name}'
    embed_model_path = embedding_model_name

    try:
        run_inference(model_path=model_path, embed_model_path=embed_model_path, question_text=question_text,
                      prompt_text=prompt_text, prompt_text_file=prompt_text_file,
                      ctx_file=context_data_file_path, resp_file=resp_file, rag_db_path=rag_db_path,
                      max_new_tokens=max_new_tokens, chunks_text_file=chunks_text_file)

        with open(os.path.join(inference_dir, 'RUN-STATUS'), 'w') as f:
            f.write("success")
    except Exception as e:
        error_str = traceback.format_exc()
        print(f"OCR Error: {e}. Cause: {error_str}")
        with open(os.path.join(inference_dir, 'error.log'), 'w') as f:
            f.write(error_str)
        with open(os.path.join(inference_dir, 'RUN-STATUS'), 'w') as f:
            f.write("failure")
