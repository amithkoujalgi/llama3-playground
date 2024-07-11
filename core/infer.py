# Installs Unsloth, Xformers (Flash Attention) and all other packages!
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
import argparse
import os
import shutil
import traceback
import uuid
from argparse import RawTextHelpFormatter

from prettytable import PrettyTable

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_MODE'] = 'disabled'

from unsloth import FastLanguageModel
from config import Config


def run_inference(model_path: str, question_text: str, prompt_text: str, ctx_file: str, resp_file: str,
                  prompt_text_file: str, max_new_tokens: int = 128):
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
    You are a helpful assistant.
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

    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    inputs = tokenizer(
        [
            prompt.format(
                context,
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
    print(f'Wrote LLM response to: {resp_file}')


def print_cli_args(cli_args: argparse.Namespace):
    print("Using the following config:")
    t = PrettyTable(['Config Key', 'Specified Value'])
    t.align["Config Key"] = "r"
    t.align["Specified Value"] = "l"
    for k, v in cli_args.__dict__.items():
        t.add_row([k, v])
    print(t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A utility to infer from an LLM using a text file as context data.',
                                     formatter_class=RawTextHelpFormatter)
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
    modelName = args.model_name
    contextDataFile = args.context_data_file_path
    questionText = args.question_text
    prompt_text = args.prompt_text
    max_new_tokens = args.max_new_tokens

    inference_dir = f'{Config.inferences_dir}/{runId}'
    os.makedirs(inference_dir, exist_ok=True)

    question_file = f'{inference_dir}/question.txt'
    ctx_data_file = f'{inference_dir}/context-data.txt'
    resp_file = f'{inference_dir}/response.txt'
    prompt_text_file = f'{inference_dir}/prompt.txt'

    with open(question_file, 'w') as f:
        f.write(questionText)
    print(f'Wrote question text to: {question_file}')

    shutil.copyfile(contextDataFile, ctx_data_file)

    model_path = f'{Config.models_dir}/{modelName}'

    try:
        run_inference(model_path=model_path, question_text=questionText, prompt_text=prompt_text,
                      ctx_file=contextDataFile, resp_file=resp_file, prompt_text_file=prompt_text_file,
                      max_new_tokens=max_new_tokens)

        with open(os.path.join(inference_dir, 'RUN-STATUS'), 'w') as f:
            f.write("success")
    except Exception as e:
        error_str = traceback.format_exc()
        print(f"OCR Error: {e}. Cause: {error_str}")
        with open(os.path.join(inference_dir, 'error.log'), 'w') as f:
            f.write(error_str)
        with open(os.path.join(inference_dir, 'RUN-STATUS'), 'w') as f:
            f.write("failure")
