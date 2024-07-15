# Installs Unsloth, Xformers (Flash Attention) and all other packages!
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
import datetime
import json
import os
import sys
import time
import traceback

from prettytable import PrettyTable

os.environ['WANDB_MODE'] = 'disabled'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

from llama3_playground.core.config import Config
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported


def _do_train(dataset_dir: str, target_model_name: str):
    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=Config.base_model,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    prompt = """Below is the context that represents a document excerpt (a section of a document), paired with a related question. Write a suitable response to the question based on the given context.

    ### Context:
    {}

    ### Question:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        contexts = examples["context"]
        inputs = examples["question"]
        outputs = examples["response"]
        texts = []
        for context, input_txt, output_txt in zip(contexts, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = prompt.format(context, input_txt, output_txt) + EOS_TOKEN
            texts.append(text)
        return {"text": texts, }

    # dataset = load_dataset("csv", data_files=Config.training_dataset_file_path, split="all")
    dataset = load_dataset("csv", data_dir=dataset_dir, split="all")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=1,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=Config.checkpoints_dir,
        ),
    )

    trainer_stats = trainer.train()

    # Local saving of LoRA adapters only!
    target_model_name_lora_adapters = f'{target_model_name}{Config.LORA_ADAPTERS_SUFFIX}'
    model.save_pretrained(target_model_name_lora_adapters)
    tokenizer.save_pretrained(target_model_name_lora_adapters)
    print(f"Saved LoRA adapters at: {target_model_name_lora_adapters}")

    # Local saving of full model - in float16!
    model.save_pretrained_merged(target_model_name, tokenizer, save_method="merged_16bit", )

    # Local saving of full model - in int4!
    # model.save_pretrained_merged(target_model_name, tokenizer, save_method="merged_4bit", )

    print(f"Fine-tuning completed. Saved new model at: {target_model_name}")


def run_train():
    start_date_time = datetime.datetime.utcnow()
    time_now = time.time()
    trainer_run_dir = f'{Config.trainer_runs_dir}/{int(time_now)}'
    _target_model_name = f'{Config.fine_tuned_model_name_prefix}-{int(time_now)}'
    target_model_dir = f'{Config.models_dir}/{_target_model_name}'
    _dataset_dir = Config.training_dataset_dir_path

    if not os.path.isdir(_dataset_dir):
        os.makedirs(_dataset_dir)

    if len(os.listdir(_dataset_dir)) == 0:
        print(f"Training dataset directory '{_dataset_dir}' is empty. Please add some data here before proceeding.")
        sys.exit(1)

    if not os.path.isdir(trainer_run_dir):
        print(f"Creating trainer run directory: {trainer_run_dir}")
        os.makedirs(trainer_run_dir)

    print("Using the following config:")
    t = PrettyTable(['Config', 'Value'])
    t.align["Config"] = "r"
    t.align["Value"] = "l"
    t.add_row(['trainer_run_dir', trainer_run_dir])
    t.add_row(['target_model_name', _target_model_name])
    t.add_row(['target_model_dir', target_model_dir])
    t.add_row(['dataset_dir', _dataset_dir])
    print(t)

    try:
        start_time = time.time()
        _do_train(dataset_dir=_dataset_dir, target_model_name=target_model_dir)
        end_time = time.time()
        with open(f"{trainer_run_dir}/out.json", 'w') as f:
            model_train_meta = {
                "model_name": _target_model_name,
                "run_at": str(start_date_time),
                "model_path": target_model_dir,
                "train_duration": end_time - start_time
            }
            json.dump(model_train_meta, f)

        with open(os.path.join(trainer_run_dir, 'RUN-STATUS'), 'w') as f:
            f.write("success")
    except Exception as e:
        error_str = traceback.format_exc()
        print(f"OCR Error: {e}. Cause: {error_str}")
        with open(os.path.join(trainer_run_dir, 'error.log'), 'w') as f:
            f.write(error_str)
        with open(os.path.join(trainer_run_dir, 'RUN-STATUS'), 'w') as f:
            f.write("failure")


if __name__ == '__main__':
    run_train()
