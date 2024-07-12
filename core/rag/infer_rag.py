# Installs Unsloth, Xformers (Flash Attention) and all other packages!
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import argparse
import os
import shutil
import sys
import traceback

from prettytable import PrettyTable
from create_rag_db import CreateChromaDB

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_MODE'] = 'disabled'

from unsloth import FastLanguageModel
from config import Config


def run_inference(model_path: str, embed_model_path: str, question_text: str, ctx_file: str, resp_file: str, rag_db_path: str, 
                 max_new_tokens: int):
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

    prompt = """Below is the context that represents a document excerpt (a section of a document), paired with a related question. Write a suitable response to the question based on the given context.

    ### Context:
    {}

    ### Question:
    {}

    ### Response:
    {}"""

    # prompt = """You are a smart and logical assistant. Use the given context and extract the required fields from it and provide the result in a json format. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. If you cannot answer the question from the given documents, please state that you do not have an answer.

    # Context:
    # {}

    # Question: 
    # {}

    # Response:
    # {}"""

    with open(ctx_file, 'r') as f:
        context = f.read()

    chroma_db = CreateChromaDB(db_path=rag_db_path, embed_model_path=embed_model_path, clear_db=True)
    chroma_db.populate_database(ocr_text=context, pdf_file_path=ctx_file)
    context_text = chroma_db.retrieve_relevant_chunks(query_text=question_text)

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
    print(response)
    with open(resp_file, 'w') as f:
        f.write(response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ask LLM.')
    parser.add_argument('runId', type=str,
                        help='Run ID')
    parser.add_argument('modelName', type=str,
                        help='Model Name')
    parser.add_argument('embeddingmodelName', type=str,
                    help='Embedding Model Name')
    parser.add_argument('contextDataFile', type=str,
                        help='Context Data File')
    parser.add_argument('ragDBPath', type=str,
                        help='Rag DB Path')
    parser.add_argument('questionText', type=str,
                        help='Question text')
    parser.add_argument('max_new_tokens', type=int,
                    help='Maximum number of new tokens to generate. Example: 1024. Default is 128.')
    args = parser.parse_args()

    if len(sys.argv) != len(vars(args)) + 1:
        print("Invalid arguments!")
        print("Arguments needed: runId, modelName, embeddingmodelName, contextDataFile, ragDBPath, questionText, max_new_tokens")
        print("Example:")
        print(
            f'python3 {os.path.basename(__file__)} "123" "llama3-8b-custom-1720545601" "Snowflake/snowflake-arctic-embed-l" "/app/ocr/pdf-text-data.txt" "/app/ocr" "what is the name of the employer?"')
        exit(1)

    runId = args.runId
    modelName = args.modelName
    embeddingmodelName = args.embeddingmodelName
    contextDataFile = args.contextDataFile
    ragDBPath = args.ragDBPath
    questionText = args.questionText
    max_new_tokens = args.max_new_tokens

    print("Using the following config:")
    t = PrettyTable(['Config', 'Value'])
    t.align["Config"] = "r"
    t.align["Value"] = "l"
    t.add_row(['runId', runId])
    t.add_row(['modelName', modelName])
    t.add_row(['embeddingmodelName', embeddingmodelName])
    t.add_row(['contextDataFile', contextDataFile])
    t.add_row(['ragDBPath', ragDBPath])
    t.add_row(['questionText', questionText])
    t.add_row(['max_new_tokens', max_new_tokens])
    print(t)

    inference_dir = f'{Config.inferences_dir}/{runId}'
    os.makedirs(inference_dir, exist_ok=True)

    question_file = f'{inference_dir}/question.txt'
    ctx_data_file = f'{inference_dir}/context-data.txt'
    resp_file = f'{inference_dir}/response.txt'
    rag_db_path = f'{ragDBPath}/{runId}'

    with open(question_file, 'w') as f:
        f.write(questionText)

    shutil.copyfile(contextDataFile, ctx_data_file)

    model_path = f'{Config.models_dir}/{modelName}'
    # embed_model_path = f'{Config.models_dir}/{embeddingmodelName}'
    embed_model_path = embeddingmodelName

    try:
        run_inference(model_path=model_path, embed_model_path=embed_model_path, question_text=questionText, ctx_file=contextDataFile, resp_file=resp_file, rag_db_path=rag_db_path, max_new_tokens=max_new_tokens)

        with open(os.path.join(inference_dir, 'RUN-STATUS'), 'w') as f:
            f.write("success")
    except Exception as e:
        error_str = traceback.format_exc()
        print(f"OCR Error: {e}. Cause: {error_str}")
        with open(os.path.join(inference_dir, 'error.log'), 'w') as f:
            f.write(error_str)
        with open(os.path.join(inference_dir, 'RUN-STATUS'), 'w') as f:
            f.write("failure")
